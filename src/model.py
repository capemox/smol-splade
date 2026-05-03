"""SAE and SAE-SPLADE model definitions (pure PyTorch, no framework dependencies)."""

import math
from typing import Dict, List, Optional, Tuple

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


# ──────────────────────────────────────────────────────────────────────────────
# Frozen document SPLADE encoder (asymmetric retrieval)
# ──────────────────────────────────────────────────────────────────────────────

class FrozenDocSPLADE(nn.Module):
    """Frozen SPLADE document encoder from any HuggingFace MLM checkpoint.

    Applies the standard SPLADE-max aggregation:
        log1p(relu(max_pool_over_tokens(MLM_logits)))

    All parameters are frozen — this encoder is never updated during training.
    It handles its own tokenization internally so the query encoder can use a
    completely different tokenizer.
    """

    def __init__(self, hf_id: str):
        super().__init__()
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.mlm = AutoModelForMaskedLM.from_pretrained(hf_id)
        self.vocab_size: int = self.mlm.config.vocab_size
        for p in self.parameters():
            p.requires_grad_(False)

    def encode(self, texts: List[str], max_length: int, no_grad: bool = True) -> torch.Tensor:
        """Encode document texts to SPLADE vectors ``[N, vocab_size]``.

        The output tensor lives on the same device as the model weights.
        Set ``no_grad=False`` when the doc encoder has trainable LoRA adapters.
        """
        import contextlib
        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            device = next(self.parameters()).device
            enc = self.tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = self.mlm(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()   # [B, L, 1]
            logits = out.logits * mask                            # zero padding [B, L, V]
            return torch.log1p(torch.relu(logits).max(dim=1).values)  # [B, V]


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


def _extract_mlm_head(mlm_model: nn.Module) -> nn.Module:
    """Extract and freeze the vocabulary projection head from a HuggingFace MLM model.

    Supports DistilBERT, BERT, and RoBERTa families.
    """
    model_type = mlm_model.config.model_type

    if model_type == "distilbert":
        class _DistilBertHead(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.vocab_transform = m.vocab_transform
                self.vocab_layer_norm = m.vocab_layer_norm
                self.vocab_projector = m.vocab_projector
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.vocab_projector(self.vocab_layer_norm(F.gelu(self.vocab_transform(x))))
        head = _DistilBertHead(mlm_model)
    elif model_type == "bert":
        head = mlm_model.cls.predictions
    elif model_type in ("roberta", "xlm-roberta"):
        head = mlm_model.lm_head
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}' for MLM head extraction. "
            "Supported: distilbert, bert, roberta, xlm-roberta."
        )

    for p in head.parameters():
        p.requires_grad_(False)
    return head


class ProjectedQuerySPLADE(nn.Module):
    """Small query encoder that routes through a frozen doc-SPLADE's MLM head.

    Architecture::

        backbone (small)
            → Linear(backbone_H → splade_H) → GELU → LayerNorm → Linear(splade_H → splade_H)
            → frozen MLM head
            → SPLADE max-pool  [log1p(relu(max_over_tokens))]

    The two-layer projection gives the model enough capacity to bridge architecturally
    different backbone representation spaces.  Both query and doc vectors live in the
    same vocabulary space, so dot products are valid regardless of tokenizer.
    """

    def __init__(self, backbone_hf_id: str, doc_splade_hf_id: str):
        super().__init__()
        from transformers import AutoModelForMaskedLM

        self.backbone = AutoModel.from_pretrained(backbone_hf_id)
        backbone_hidden = self.backbone.config.hidden_size

        doc_mlm = AutoModelForMaskedLM.from_pretrained(doc_splade_hf_id)
        splade_hidden = doc_mlm.config.hidden_size
        self.vocab_size: int = doc_mlm.config.vocab_size

        self.proj = nn.Sequential(
            nn.Linear(backbone_hidden, splade_hidden),
            nn.GELU(),
            nn.LayerNorm(splade_hidden),
            nn.Linear(splade_hidden, splade_hidden),
        )
        self.mlm_head = _extract_mlm_head(doc_mlm)
        del doc_mlm

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        override_k: int = 0,  # unused; accepted for interface compatibility with SAESPLADEModel
    ) -> torch.Tensor:
        """Encode queries to SPLADE vectors in vocabulary space ``[B, vocab_size]``."""
        hidden = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.mlm_head(self.proj(hidden)) * attention_mask.unsqueeze(-1).float()
        return torch.log1p(torch.relu(logits).max(dim=1).values)


def projected_alignment_loss(
    query_model: ProjectedQuerySPLADE,
    doc_splade: "FrozenDocSPLADE",
    q_ids: torch.Tensor,
    q_mask: torch.Tensor,
    texts: List[str],
    doc_max_length: int,
) -> torch.Tensor:
    """Cosine alignment loss between query encoder output and doc SPLADE output on the same texts.

    Used for the alignment warm-up phase: teaches the projection layer to produce
    splade-v3-like vocabulary representations before ranking fine-tuning begins,
    avoiding the cold-start gradient stagnation caused by a random projection.

    Cosine similarity is used rather than MSE because SPLADE vectors are ~99% zeros;
    MSE would be dominated by the zero dimensions and the model could collapse to
    outputting all-zeros.  Cosine only measures directional alignment over the
    non-zero dimensions, which is exactly what matters for retrieval.
    """
    q_vecs = query_model.encode(q_ids, q_mask)
    with torch.no_grad():
        d_vecs = doc_splade.encode(texts, doc_max_length)
    return (1.0 - F.cosine_similarity(q_vecs, d_vecs)).mean()


class VocabTransplantQuerySPLADE(nn.Module):
    """Query SPLADE encoder built from a vocab-transplanted MLM model.

    After ``_run_tokensurgeon`` transplants the doc SPLADE's vocabulary onto
    the small query encoder, this class wraps the result for standard SPLADE
    encoding:  log1p(relu(max_over_tokens(MLM_logits))).

    Both query and doc vectors live in the same vocabulary space, so their
    dot product is valid.  The full model (backbone + MLM head) is trainable.
    """

    def __init__(self, model_path: str):
        super().__init__()
        from transformers import AutoModelForMaskedLM
        self.mlm = AutoModelForMaskedLM.from_pretrained(model_path)
        self.vocab_size: int = self.mlm.config.vocab_size

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        override_k: int = 0,  # unused; accepted for interface compatibility
    ) -> torch.Tensor:
        """Encode queries to SPLADE vectors in vocabulary space ``[B, vocab_size]``."""
        out = self.mlm(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits * attention_mask.unsqueeze(-1).float()
        return torch.log1p(torch.relu(logits).max(dim=1).values)


def vocab_transplant_splade_loss(
    query_model: "VocabTransplantQuerySPLADE",
    q_ids: torch.Tensor,
    q_mask: torch.Tensor,
    doc_vecs: torch.Tensor,
    teacher_scores: Optional[torch.Tensor],
    lambda_q: float,
    flops_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Training loss for vocab-transplant query encoder.

    Same retrieval objective as asymmetric_splade_loss but with a differentiable
    L1 FLOPs term: ``q_vecs.mean(dim=0).sum()``.  Unlike the step-function
    indicator ``(q_vecs > 0).float()`` used in asymmetric_splade_loss, this
    carries a non-zero gradient that sparsifies a model starting from a dense
    (no top-k gate, no frozen sparse head) initialisation.
    """
    B = q_ids.shape[0]
    nway = doc_vecs.shape[0] // B

    q_vecs = query_model.encode(q_ids, q_mask)  # [B, vocab_size]
    scores = q_vecs @ doc_vecs.T                # [B, B*nway]

    # ── Retrieval loss ────────────────────────────────────────────────
    if teacher_scores is not None:
        idx = torch.arange(B, device=scores.device)
        student = scores.reshape(B, B, nway)[idx, idx]      # [B, nway]
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)(
            F.log_softmax(student, dim=-1),
            F.log_softmax(teacher_scores, dim=-1),
        )
        mse_loss = F.mse_loss(
            student[:, 0:1] - student[:, 1:],
            teacher_scores[:, 0:1] - teacher_scores[:, 1:],
        )
        retr_loss = kl_loss + 0.05 * mse_loss
    else:
        labels = torch.arange(B, device=scores.device) * nway
        retr_loss = F.cross_entropy(scores, labels)

    # ── Squared FLOPS regularisation (sentence-transformers / SPLADE paper) ──
    # sum(mean(dim=0)**2): gradient ∝ current activation → soft sparsity that
    # kills large activations strongly while leaving small ones to stabilise.
    # This is strictly differentiable; L1 (uniform gradient) was too aggressive.
    flops = flops_scale * lambda_q * (q_vecs.mean(dim=0) ** 2).sum()
    avg_q_nnz = (q_vecs > 0).float().mean(0).sum()  # monitoring only, no grad

    total = retr_loss + flops
    return total, {
        "loss": total.item(),
        "retr": retr_loss.item(),
        "flops": flops.item(),
        "avg_q_nnz": avg_q_nnz.item(),
    }


def vocab_transplant_joint_loss(
    query_model: "VocabTransplantQuerySPLADE",
    q_ids: torch.Tensor,
    q_mask: torch.Tensor,
    doc_vecs: torch.Tensor,
    teacher_q_vecs: torch.Tensor,
    lambda_q: float,
    flops_scale: float,
    align_coeff: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Joint CE ranking + cosine alignment loss for vocab-transplant query encoder.

    Replaces self-distillation KL, which fails because naver/splade-v3 is a doc
    encoder and produces near-zero / near-uniform score distributions on short
    query texts, giving the model no useful gradient to follow.

    CE ranking:  cross-entropy over the [B, B*nway] query×doc score matrix;
                 first doc per query is the positive, rest are in-batch hard negatives.
    Alignment:   cosine similarity between query_model(query) and doc_splade(query).
                 Anchors the query encoder to splade-v3's vocabulary space without
                 requiring valid teacher score distributions.
    FLOPS:       squared mean regularisation for sparsity.
    """
    B = q_ids.shape[0]
    nway = doc_vecs.shape[0] // B

    q_vecs = query_model.encode(q_ids, q_mask)  # [B, vocab_size]
    scores = q_vecs @ doc_vecs.T                # [B, B*nway]

    labels = torch.arange(B, device=q_ids.device) * nway
    ranking_loss = F.cross_entropy(scores, labels)

    # Cast teacher to match q_vecs dtype (fp32 outside autocast, fp16 inside).
    align_loss = (1.0 - F.cosine_similarity(q_vecs, teacher_q_vecs.to(q_vecs.dtype))).mean()

    flops = flops_scale * lambda_q * (q_vecs.mean(dim=0) ** 2).sum()
    avg_q_nnz = (q_vecs > 0).float().mean(0).sum()

    total = ranking_loss + align_coeff * align_loss + flops
    return total, {
        "loss": total.item(),
        "ranking": ranking_loss.item(),
        "align": align_loss.item(),
        "flops": flops.item(),
        "avg_q_nnz": avg_q_nnz.item(),
    }


def asymmetric_splade_loss(
    query_model: SAESPLADEModel,
    q_ids: torch.Tensor,
    q_mask: torch.Tensor,
    doc_vecs: torch.Tensor,
    teacher_scores: Optional[torch.Tensor],
    lambda_q: float,
    flops_scale: float,
    override_k: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Training loss for asymmetric retrieval.

    ``doc_vecs`` are pre-encoded by the frozen :class:`FrozenDocSPLADE` — no
    gradient flows through them.  Only the query encoder is updated.
    FLOPs regularisation is applied to the query side only.

    Args:
        query_model: the SAE-SPLADE query encoder being trained.
        q_ids / q_mask: tokenised queries ``[B, Lq]``.
        doc_vecs: pre-encoded document vectors ``[B*nway, vocab_size]``.
        teacher_scores: ColBERTv2 scores ``[B, nway]``, or None for CE loss.
        lambda_q: query FLOPs regularisation coefficient.
        flops_scale: ramp factor (0→1) applied to the FLOPs term.
        override_k: if >0, use this k for SAE encoding instead of model default.
    """
    B = q_ids.shape[0]
    nway = doc_vecs.shape[0] // B

    q_vecs = query_model.encode(q_ids, q_mask, override_k=override_k)  # [B, vocab_size]

    # [B, vocab_size] @ [vocab_size, B*nway] → [B, B*nway]
    scores = q_vecs @ doc_vecs.T

    # ── Retrieval loss ────────────────────────────────────────────────
    if teacher_scores is not None:
        # Extract each query's nway scores from the diagonal block
        idx = torch.arange(B, device=scores.device)
        student = scores.reshape(B, B, nway)[idx, idx]          # [B, nway]
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)(
            F.log_softmax(student, dim=-1),
            F.log_softmax(teacher_scores, dim=-1),
        )
        mse_loss = F.mse_loss(
            student[:, 0:1] - student[:, 1:],
            teacher_scores[:, 0:1] - teacher_scores[:, 1:],
        )
        retr_loss = kl_loss + 0.05 * mse_loss
    else:
        labels = torch.arange(B, device=scores.device) * nway
        retr_loss = F.cross_entropy(scores, labels)

    # ── Query FLOPs regularisation only ──────────────────────────────
    avg_q_nnz = (q_vecs > 0).float().mean(0).sum()
    flops = flops_scale * lambda_q * avg_q_nnz

    total = retr_loss + flops
    return total, {
        "loss": total.item(),
        "retr": retr_loss.item(),
        "flops": flops.item(),
        "avg_q_nnz": avg_q_nnz.item(),
    }
