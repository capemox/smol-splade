"""SAE and SAE-SPLADE model definitions (pure PyTorch, no framework dependencies)."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder.

    Learns a dictionary of ``sae_width`` directions in the backbone's hidden space.
    For each token vector, keeps the ``k`` most-activated directions and reconstructs
    the input from them.  Dead neurons are recovered via an auxiliary loss.
    """

    def __init__(
        self,
        hidden_size: int,
        sae_width: int,
        k: int,
        aux_k: int = 0,
        dead_steps_threshold: int = 10_000_000,
        normalize_input: bool = False,
    ):
        super().__init__()
        self.k = k
        self.sae_width = sae_width
        self.hidden_size = hidden_size
        self.aux_k = aux_k if aux_k > 0 else k * 2
        self.dead_steps_threshold = dead_steps_threshold
        self.normalize_input = normalize_input

        self.W_enc = nn.Parameter(torch.empty(hidden_size, sae_width))
        self.b_enc = nn.Parameter(torch.zeros(sae_width))
        self.W_dec = nn.Parameter(torch.empty(sae_width, hidden_size))
        self.b_dec = nn.Parameter(torch.zeros(hidden_size))

        if normalize_input:
            self.register_buffer("mean_norm", torch.ones(1))
            self.register_buffer("mean_bias", torch.zeros(hidden_size))

        self.register_buffer("steps_since_active", torch.zeros(sae_width, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5), mode="fan_out")
        self.W_dec.data = F.normalize(self.W_enc.data.T.clone().contiguous(), dim=-1)
        self.W_enc.data = self.W_enc.data.contiguous()

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _pre_encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            return (x - self.mean_bias) / (self.mean_norm + 1e-8)
        return x

    @torch.no_grad()
    def init_normalisation(self, vecs: torch.Tensor):
        """Compute mean bias and mean norm from a sample of token vectors."""
        self.mean_bias.data = vecs.mean(0)
        centred = vecs - self.mean_bias
        self.mean_norm.data = centred.norm(dim=-1).mean().unsqueeze(0)

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode(
        self, x: torch.Tensor, override_k: int = 0
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: ``[N, hidden_size]`` token vectors.
            override_k: use this k instead of ``self.k`` (0 = use default).
        Returns:
            ``(inds, vals)`` where ``inds`` is ``[N, k]`` (or None for dense)
            and ``vals`` is ``[N, k]`` (or ``[N, sae_width]`` for dense).
        """
        k = override_k if override_k > 0 else self.k
        xn = self._pre_encode(x)
        latents = (xn - self.b_dec) @ self.W_enc + self.b_enc  # [N, sae_width]

        if k >= self.sae_width:
            return None, F.relu(latents)

        vals, inds = torch.topk(latents, k, dim=-1)
        return inds, F.relu(vals)

    def decode(
        self, inds: Optional[torch.Tensor], vals: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inds: ``[N, k]`` or None (dense case).
            vals: ``[N, k]`` or ``[N, sae_width]``.
        Returns:
            Reconstructed vectors ``[N, hidden_size]``.
        """
        if inds is None:
            return vals @ self.W_dec + self.b_dec
        W_sel = self.W_dec[inds]  # [N, k, hidden_size]
        return (W_sel * vals.unsqueeze(-1)).sum(1) + self.b_dec

    # ------------------------------------------------------------------
    # SAE pretraining forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass for pretraining.

        Returns a dict with reconstruction, auxiliary reconstruction, and
        logging stats.
        """
        xn = self._pre_encode(x)
        latents = (xn - self.b_dec) @ self.W_enc + self.b_enc  # [N, sae_width]

        # ── Main TopK ──────────────────────────────────────────────────
        top_vals, inds = torch.topk(latents, self.k, dim=-1)
        vals = F.relu(top_vals)

        # Track dead neurons
        with torch.no_grad():
            self.steps_since_active.add_(x.shape[0])
            active = inds[vals > 0].unique()
            if active.numel() > 0:
                self.steps_since_active[active] = 0

        # ── Auxiliary TopK (dead neuron recovery) ──────────────────────
        with torch.no_grad():
            dead_mask = (self.steps_since_active > self.dead_steps_threshold).float()

        aux_latents = latents.detach() * dead_mask.unsqueeze(0)
        aux_top_vals, aux_inds = torch.topk(aux_latents, self.aux_k, dim=-1)
        aux_vals = F.relu(aux_top_vals)

        # ── Decode ─────────────────────────────────────────────────────
        x_rec = self.decode(inds, vals)
        x_aux_rec = self.decode(aux_inds, aux_vals)

        return {
            "inds": inds,
            "vals": vals,
            "x_rec": x_rec,
            "x_aux_rec": x_aux_rec,
            "xn": xn,
            "sparsity": (vals > 0).float().sum(-1).mean(),
            "dead_ratio": dead_mask.mean(),
        }

    # ------------------------------------------------------------------
    # Post-step hooks (call after every optimiser step)
    # ------------------------------------------------------------------

    def remove_parallel_gradient(self):
        """Remove the gradient component parallel to W_dec columns (call before optimizer.step)."""
        with torch.no_grad():
            if self.W_dec.grad is not None:
                W_unit = F.normalize(self.W_dec.data, dim=-1)
                parallel = (self.W_dec.grad * W_unit).sum(-1, keepdim=True)
                self.W_dec.grad.sub_(parallel * W_unit)

    def post_step(self):
        """Normalise W_dec columns to unit norm (call after optimizer.step)."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# SAE pretraining wrapper
# ──────────────────────────────────────────────────────────────────────────────

class SAEPretrainModel(nn.Module):
    """Backbone + SAE for the SAE pretraining stage."""

    def __init__(self, hf_id: str, sae: TopKSAE, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(hf_id)
        self.sae = sae

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def get_token_vecs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return valid (non-padding) token vectors ``[N_tokens, hidden_size]``."""
        out = self.backbone(input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, H]
        mask = attention_mask.bool()
        return hidden[mask]

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        out = self.backbone(input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, H]
        mask = attention_mask.bool()
        tokens = hidden[mask]           # [N_tokens, H]
        return self.sae(tokens)


def sae_loss(
    output: Dict[str, torch.Tensor],
    rcst_coeff: float = 1.0,
    aux_coeff: float = 0.0625,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Reconstruction + auxiliary (dead-neuron) loss for SAE pretraining."""
    xn = output["xn"]
    x_rec = output["x_rec"]
    x_aux_rec = output["x_aux_rec"]

    rcst = ((xn.detach() - x_rec) ** 2).sum(-1).mean()
    # Auxiliary target: rec_main.detach() + aux_contribution
    aux = ((xn.detach() - (x_rec.detach() + x_aux_rec)) ** 2).sum(-1).mean().nan_to_num(0)

    total = rcst_coeff * rcst + aux_coeff * aux
    return total, {
        "loss": total.item(),
        "rcst": rcst.item(),
        "aux": aux.item(),
        "sparsity": output["sparsity"].item(),
        "dead_ratio": output["dead_ratio"].item(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SAE-SPLADE dual encoder
# ──────────────────────────────────────────────────────────────────────────────

class SAESPLADEModel(nn.Module):
    """SAE-SPLADE dual sparse retrieval encoder.

    Uses the backbone to get token embeddings, then the pre-trained SAE to map
    each token to a sparse set of latent dimensions.  Per-document aggregation
    is log1p(amax over tokens), matching the original SPLADE formula.
    """

    def __init__(self, hf_id: str, sae: TopKSAE, scale: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(hf_id)
        self.sae = sae
        self.scale = scale
        if scale:
            self.alpha = nn.Parameter(torch.ones(1))

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        override_k: int = 0,
    ) -> torch.Tensor:
        """Encode a batch of texts to sparse SPLADE vectors ``[B, sae_width]``."""
        out = self.backbone(input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, H]
        B, L, H = hidden.shape

        # Flatten to token level
        tokens = hidden.reshape(B * L, H)
        inds, vals = self.sae.encode(tokens, override_k=override_k)

        # Zero-out padding tokens
        flat_mask = attention_mask.reshape(B * L).float()
        if inds is None:
            # Dense path
            vals = vals * flat_mask.unsqueeze(-1)
            vals = vals.reshape(B, L, self.sae.sae_width)
            doc_rep = vals.max(dim=1).values
        else:
            # Sparse scatter-max path
            k = inds.shape[-1]
            vals = vals * flat_mask.unsqueeze(-1)  # [B*L, k]

            doc_ids = torch.arange(B, device=inds.device).repeat_interleave(L)  # [B*L]
            doc_ids = doc_ids.unsqueeze(1).expand_as(inds).reshape(-1)           # [B*L*k]
            lat_ids = inds.reshape(-1)                                             # [B*L*k]
            vals_flat = vals.reshape(-1)                                           # [B*L*k]

            combined = doc_ids * self.sae.sae_width + lat_ids
            doc_rep_flat = torch.zeros(
                B * self.sae.sae_width, device=vals.device, dtype=vals.dtype
            )
            doc_rep_flat.scatter_reduce_(0, combined, vals_flat, reduce="amax", include_self=True)
            doc_rep = doc_rep_flat.reshape(B, self.sae.sae_width)

        doc_rep = torch.log1p(doc_rep)
        if self.scale:
            doc_rep = doc_rep * self.alpha
        return doc_rep

    def score(self, q_vecs: torch.Tensor, d_vecs: torch.Tensor) -> torch.Tensor:
        """Dot product of sparse query and document vectors."""
        return q_vecs @ d_vecs.T  # [Bq, Bd]


def splade_loss(
    model: SAESPLADEModel,
    q_ids: torch.Tensor,
    q_mask: torch.Tensor,
    d_ids: torch.Tensor,
    d_mask: torch.Tensor,
    teacher_scores: Optional[torch.Tensor],
    lambda_d: float,
    lambda_q: float,
    flops_scale: float,
    override_k: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute SAE-SPLADE training loss.

    Supports two modes:
    * **With teacher scores** (``teacher_scores`` is not None): KL divergence +
      MSE delta distillation from a teacher (e.g. ColBERTv2).
    * **Without teacher scores**: softmax cross-entropy in-batch negatives.

    Both modes add FLOPs regularisation.
    """
    B = q_ids.shape[0]
    nway = d_ids.shape[0] // B

    q_vecs = model.encode(q_ids, q_mask, override_k=override_k)           # [B, W]
    d_vecs = model.encode(d_ids, d_mask, override_k=override_k)           # [B*nway, W]

    # Scores: [B, B*nway]
    scores = model.score(q_vecs, d_vecs)

    # ── Retrieval loss ────────────────────────────────────────────────
    if teacher_scores is not None:
        # teacher_scores: [B, nway]
        # scores is [B, B*nway]; extract each query's own nway-block (diagonal)
        student = scores.reshape(B, B, nway)[torch.arange(B, device=scores.device), torch.arange(B, device=scores.device)]

        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)(
            F.log_softmax(student, dim=-1),
            F.log_softmax(teacher_scores, dim=-1),
        )
        pos_student = student[:, 0:1]
        pos_teacher = teacher_scores[:, 0:1]
        mse_loss = F.mse_loss(
            pos_student - student[:, 1:],
            pos_teacher - teacher_scores[:, 1:],
        )
        retr_loss = kl_loss + 0.05 * mse_loss
    else:
        # In-batch negatives: first document of each group is the positive
        # labels: [0, nway, 2*nway, ...]
        labels = torch.arange(B, device=scores.device) * nway
        retr_loss = F.cross_entropy(scores, labels)

    # ── FLOPs regularisation ──────────────────────────────────────────
    avg_d_nnz = (d_vecs > 0).float().mean(0).sum()   # expected non-zeros per doc
    avg_q_nnz = (q_vecs > 0).float().mean(0).sum()   # expected non-zeros per query
    flops = flops_scale * (lambda_d * avg_d_nnz + lambda_q * avg_q_nnz)

    total = retr_loss + flops
    return total, {
        "loss": total.item(),
        "retr": retr_loss.item(),
        "flops": flops.item(),
        "avg_d_nnz": avg_d_nnz.item(),
        "avg_q_nnz": avg_q_nnz.item(),
    }
