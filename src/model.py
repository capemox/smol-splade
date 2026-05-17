"""SAE and SAE-SPLADE model definitions (pure PyTorch, no framework dependencies)."""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen Linear layer.

    forward(x) = frozen(x) + lora_B(lora_A(x))

    lora_B is zero-initialised so the adapter starts as identity at init.
    Only lora_A and lora_B have requires_grad=True; the base weight stays frozen.
    """

    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        out_features, in_features = linear.weight.shape
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        dtype = linear.weight.dtype
        self.lora_A = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(rank, out_features, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_B(self.lora_A(x))


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

    def encode_logits(self, texts: List[str], max_length: int) -> torch.Tensor:
        """Return max-pooled MLM logits BEFORE relu and log1p — ``[N, vocab_size]``.

        Used for KD warmup: provides dense gradients to all vocabulary dimensions
        regardless of their current activation state, avoiding the dead-dim collapse
        that occurs when training cosine alignment from a cold-start BERT model.
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            enc = self.tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = self.mlm(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            logits = out.logits * mask                           # [B, L, V]
            return logits.max(dim=1).values                      # [B, V]


class ShallowSpladeQuery(nn.Module):
    """SPLADE query encoder built by truncating a doc SPLADE's body to N layers.

    The query model shares the doc encoder's tokenizer, embeddings, and MLM
    head — only the body is shallower. This sidesteps every cross-architecture
    mismatch problem the vocab-transplant approach has to fight (vocab overlap,
    BPE convention, kNN-interpolation quality, head alignment): everything
    *except* the body's intermediate representations is already exactly correct,
    because the query model is literally the doc model with later layers
    removed.

    Training task: re-learn the body so its layer-N output, when fed through the
    same frozen MLM head, produces SPLADE vectors that align with the full doc
    encoder's outputs. Standard layer-distillation, with strong priors.

    Currently supports BERT-style architectures (BertForMaskedLM, including
    naver/splade-v3 which is bert-base-uncased + SPLADE training). Layers live
    at ``self.mlm.bert.encoder.layer``; head at ``self.mlm.cls``.

    Args:
        hf_id: HuggingFace ID of the doc SPLADE checkpoint. The query model
            is constructed by loading this and chopping its body.
        n_layers: number of body layers to keep (counted from the input side).
            Must be ``≤`` the model's ``num_hidden_layers``.
    """

    def __init__(self, hf_id: str, n_layers: int, layer_indices: Optional[List[int]] = None):
        super().__init__()
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.mlm = AutoModelForMaskedLM.from_pretrained(hf_id)
        self.vocab_size: int = self.mlm.config.vocab_size

        # Locate the body and truncate. Keep this guarded so accidental use on
        # other architectures fails loudly rather than silently no-op'ing.
        if not hasattr(self.mlm, "bert") or not hasattr(self.mlm.bert, "encoder"):
            raise ValueError(
                f"ShallowSpladeQuery currently supports BERT-style MLMs "
                f"(bert.encoder.layer). Got {type(self.mlm).__name__}."
            )
        original_n = len(self.mlm.bert.encoder.layer)
        if n_layers <= 0 or n_layers > original_n:
            raise ValueError(
                f"n_layers={n_layers} must be in [1, {original_n}]"
            )
        if layer_indices is not None:
            if len(layer_indices) != n_layers:
                raise ValueError(
                    f"layer_indices length ({len(layer_indices)}) must match n_layers={n_layers}"
                )
            if any(idx < 0 or idx >= original_n for idx in layer_indices):
                raise ValueError(
                    f"layer_indices={layer_indices} must all be in [0, {original_n - 1}]"
                )
            selected_layers = [self.mlm.bert.encoder.layer[idx] for idx in layer_indices]
            self.layer_indices: List[int] = list(layer_indices)
        else:
            selected_layers = list(self.mlm.bert.encoder.layer)[:n_layers]
            self.layer_indices = list(range(n_layers))
        self.mlm.bert.encoder.layer = nn.ModuleList(
            selected_layers
        )
        self.mlm.config.num_hidden_layers = n_layers
        self.n_layers: int = n_layers
        self.original_n_layers: int = original_n

    # ── Trainability control ──────────────────────────────────────────
    def freeze_for_warmup(self) -> None:
        """Freeze embeddings and the MLM head; only body layers train.

        Used during the warmup phase: the head is the doc encoder's, already
        perfectly tuned for the SPLADE distribution. Letting it drift while the
        body is also moving is unstable. Keep it pinned, let the body adapt to
        producing layer-N hidden states the head can interpret, then unfreeze.
        """
        # First freeze everything ...
        for p in self.parameters():
            p.requires_grad_(False)
        # ... then unfreeze just the body.
        for p in self.mlm.bert.encoder.layer.parameters():
            p.requires_grad_(True)

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter for global fine-tuning."""
        for p in self.parameters():
            p.requires_grad_(True)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Encoding ──────────────────────────────────────────────────────
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        override_k: int = 0,  # unused; accepted for interface compatibility
    ) -> torch.Tensor:
        """Encode pre-tokenized inputs to SPLADE vectors ``[B, vocab_size]``.

        Matches the ``VocabTransplantQuerySPLADE`` interface so this drops into
        ``_encode_texts`` in eval without modification.
        """
        out = self.mlm(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits * attention_mask.unsqueeze(-1).float()
        return torch.log1p(torch.relu(logits).max(dim=1).values)


class FactorizedWordEmbeddings(nn.Module):
    """ALBERT-style factorized token embeddings for BERT.

    Stores a lexical table ``A`` with shape ``[vocab, factor_dim]`` and an
    up-projection ``B`` with shape ``[factor_dim, hidden]``. The effective
    embedding matrix is ``A @ B`` but it is never materialized during normal
    forward passes.
    """

    def __init__(self, vocab_size: int, hidden_size: int, factor_dim: int, padding_idx: int | None = None):
        super().__init__()
        self.lexical_embeddings = nn.Embedding(vocab_size, factor_dim, padding_idx=padding_idx)
        self.up_project = nn.Linear(factor_dim, hidden_size, bias=False)
        self.embedding_dim = hidden_size
        self.num_embeddings = vocab_size
        self.padding_idx = padding_idx

    @property
    def weight(self) -> torch.Tensor:
        return self.lexical_embeddings.weight @ self.up_project.weight.T

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.up_project(self.lexical_embeddings(input_ids))


class FactorizedBertDecoder(nn.Module):
    """Decoder tied to :class:`FactorizedWordEmbeddings` in factorized form."""

    def __init__(self, factorized_embeddings: FactorizedWordEmbeddings, bias: torch.Tensor | None = None):
        super().__init__()
        self.factorized_embeddings = factorized_embeddings
        vocab_size = factorized_embeddings.num_embeddings
        if bias is None:
            self.bias = nn.Parameter(torch.zeros(vocab_size))
        elif isinstance(bias, nn.Parameter):
            self.bias = bias
        else:
            self.bias = nn.Parameter(bias.detach().clone())

    @property
    def weight(self) -> torch.Tensor:
        return self.factorized_embeddings.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bottleneck = F.linear(hidden_states, self.factorized_embeddings.up_project.weight.T)
        return F.linear(bottleneck, self.factorized_embeddings.lexical_embeddings.weight, self.bias)


def _factorize_embedding_matrix(weight: torch.Tensor, factor_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``A, B`` such that ``A @ B`` approximates ``weight``.

    ``weight`` is ``[vocab, hidden]``. ``A`` is ``[vocab, factor_dim]`` and
    ``B`` is ``[factor_dim, hidden]``. Uses truncated SVD with the square root
    of singular values split evenly between the two factors.
    """
    full = weight.detach().float().cpu()
    vocab_size, hidden_size = full.shape
    if factor_dim <= 0 or factor_dim > min(vocab_size, hidden_size):
        raise ValueError(f"factor_dim={factor_dim} must be in [1, {min(vocab_size, hidden_size)}]")
    U, S, Vh = torch.linalg.svd(full, full_matrices=False)
    sqrt_s = S[:factor_dim].sqrt()
    A = U[:, :factor_dim] * sqrt_s.unsqueeze(0)
    B = sqrt_s.unsqueeze(1) * Vh[:factor_dim, :]
    return A.contiguous(), B.contiguous()


def _factorize_embedding_matrix_lowrank(
    weight: torch.Tensor,
    factor_dim: int,
    oversample: int = 16,
    niter: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-friendlier randomized low-rank SVD for very large lexical matrices."""
    full = weight.detach().float().cpu()
    vocab_size, hidden_size = full.shape
    if factor_dim <= 0 or factor_dim > min(vocab_size, hidden_size):
        raise ValueError(f"factor_dim={factor_dim} must be in [1, {min(vocab_size, hidden_size)}]")
    q = min(min(vocab_size, hidden_size), factor_dim + oversample)
    U, S, V = torch.svd_lowrank(full, q=q, niter=niter)
    sqrt_s = S[:factor_dim].sqrt()
    A = U[:, :factor_dim] * sqrt_s.unsqueeze(0)
    B = sqrt_s.unsqueeze(1) * V[:, :factor_dim].T
    return A.contiguous(), B.contiguous()


class ShallowFactorizedSpladeQuery(ShallowSpladeQuery):
    """Shallow SPLADE query with ALBERT-style factorized embedding/head matrix.

    The factorized lexical table is shared between input embeddings and the MLM
    decoder:

    ``input: token_id -> A[token_id] -> B -> hidden``
    ``output: hidden -> B.T -> A.T -> vocab logits``

    This preserves the original SPLADE vocabulary/output dimensionality while
    reducing the parameter floor from the tied ``[vocab, hidden]`` matrix.
    """

    def __init__(
        self,
        hf_id: str,
        n_layers: int,
        factorized_embedding_dim: int = 128,
        init: str = "svd",
        layer_indices: Optional[List[int]] = None,
    ):
        super().__init__(hf_id, n_layers, layer_indices=layer_indices)
        self.factorized_embedding_dim = int(factorized_embedding_dim)
        self.factorization_init = init
        self._install_factorized_embeddings(init=init)

    def _install_factorized_embeddings(self, init: str = "svd") -> None:
        old_embeddings = self.mlm.bert.embeddings.word_embeddings
        old_weight = old_embeddings.weight.detach()
        vocab_size, hidden_size = old_weight.shape
        padding_idx = getattr(old_embeddings, "padding_idx", None)

        factorized = FactorizedWordEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            factor_dim=self.factorized_embedding_dim,
            padding_idx=padding_idx,
        )
        if init == "svd":
            A, B = _factorize_embedding_matrix(old_weight, self.factorized_embedding_dim)
            factorized.lexical_embeddings.weight.data.copy_(A.to(old_weight.device, dtype=old_weight.dtype))
            factorized.up_project.weight.data.copy_(B.T.to(old_weight.device, dtype=old_weight.dtype))
        elif init == "random":
            nn.init.normal_(factorized.lexical_embeddings.weight, mean=0.0, std=self.mlm.config.initializer_range)
            nn.init.normal_(factorized.up_project.weight, mean=0.0, std=self.mlm.config.initializer_range)
        else:
            raise ValueError(f"Unknown factorized embedding init: {init}")

        self.mlm.bert.embeddings.word_embeddings = factorized

        old_bias = getattr(self.mlm.cls.predictions, "bias", None)
        decoder = FactorizedBertDecoder(factorized, bias=old_bias)
        self.mlm.cls.predictions.decoder = decoder
        self.mlm.cls.predictions.bias = decoder.bias

    def factorized_param_count(self) -> int:
        emb = self.mlm.bert.embeddings.word_embeddings
        return emb.lexical_embeddings.weight.numel() + emb.up_project.weight.numel() + emb.num_embeddings


class FrozenLionSPLADE(nn.Module):
    """Frozen Lion-SP SPLADE encoder (decoder-only Llama, bidirectional).

    Loads hzeng/Lion-SP-*-llama3-marco-mntp (LoRA adapter on Llama-3) and applies
    the Lion SPLADE formula:
        log1p(relu(max_over_tokens(logits * hidden_size**-0.25)))

    Uses transformers 5.x ``config.is_causal = False`` to enable bidirectional
    attention — no custom class needed.  All parameters are frozen.
    """

    def __init__(self, hf_id: str):
        super().__init__()
        import json
        from huggingface_hub import hf_hub_download
        from transformers import LlamaForCausalLM, AutoTokenizer
        from peft import PeftModel, LoraConfig

        print(f"[FrozenLionSPLADE] Loading {hf_id} …")
        adapter_cfg_path = hf_hub_download(hf_id, "adapter_config.json")
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_model_path = adapter_cfg["base_model_name_or_path"]
        print(f"[FrozenLionSPLADE] Base model: {base_model_path}")

        base = LlamaForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        base.config.is_causal = False  # transformers 5.x bidirectional switch

        lora_cfg = LoraConfig.from_pretrained(hf_id)
        peft_model = PeftModel.from_pretrained(base, hf_id, config=lora_cfg, is_trainable=False)
        merged = peft_model.merge_and_unload()
        merged.config.is_causal = False  # preserve after merge
        self.model = merged

        self.hidden_size: int = self.model.config.hidden_size
        self.vocab_size: int = self.model.config.vocab_size

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        for p in self.parameters():
            p.requires_grad_(False)

    def _tokenize(self, texts: List[str], max_length: int) -> dict:
        device = next(self.parameters()).device
        enc = self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    def encode(self, texts: List[str], max_length: int, no_grad: bool = True) -> torch.Tensor:
        """Encode texts to SPLADE vectors ``[N, vocab_size]``."""
        import contextlib
        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            enc = self._tokenize(texts, max_length)
            logits = self.model(**enc, use_cache=False).logits  # [B, L, V]
            logits = logits * (self.hidden_size ** -0.25)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            logits = logits + (1.0 - mask) * -1e6
            return torch.log1p(torch.relu(logits.max(dim=1).values))

    def encode_logits(self, texts: List[str], max_length: int) -> torch.Tensor:
        """Return max-pooled logits BEFORE relu/log1p — ``[N, vocab_size]``."""
        with torch.no_grad():
            enc = self._tokenize(texts, max_length)
            logits = self.model(**enc, use_cache=False).logits
            logits = logits * (self.hidden_size ** -0.25)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            logits = logits + (1.0 - mask) * -1e6
            return logits.max(dim=1).values


class ShallowLionQuery(nn.Module):
    """Lion-SP-1B truncated to N body layers for use as a lightweight query encoder.

    Same philosophy as ShallowSpladeQuery: keep the full model's embeddings and
    LM head pinned to the doc encoder's weights during warmup, train only the
    body layers, then fine-tune everything jointly.

    Layers live at ``self.model.model.layers``; LM head at ``self.model.lm_head``.
    SPLADE formula mirrors FrozenLionSPLADE:
        log1p(relu(max_pool(logits * hidden_size^-0.25)))
    """

    def __init__(self, hf_id: str, n_layers: int, layer_indices: Optional[List[int]] = None, head_lora_rank: int = 0):
        super().__init__()
        import json
        from huggingface_hub import hf_hub_download
        from transformers import LlamaForCausalLM, AutoTokenizer
        from peft import PeftModel, LoraConfig

        adapter_cfg_path = hf_hub_download(hf_id, "adapter_config.json")
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_model_path = adapter_cfg["base_model_name_or_path"]
        print(f"[ShallowLionQuery] Base: {base_model_path}")

        # bfloat16: same exponent range as float32 so no GradScaler needed,
        # but half the memory of float32 for weights and AdamW states.
        base = LlamaForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )
        base.config.is_causal = False
        lora_cfg = LoraConfig.from_pretrained(hf_id)
        peft_model = PeftModel.from_pretrained(base, hf_id, config=lora_cfg, is_trainable=False)
        merged = peft_model.merge_and_unload()
        merged.config.is_causal = False

        self.model = merged
        self.hidden_size: int = merged.config.hidden_size
        self.vocab_size: int = merged.config.vocab_size

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        original_n = len(self.model.model.layers)
        if n_layers <= 0 or n_layers > original_n:
            raise ValueError(f"n_layers={n_layers} must be in [1, {original_n}]")
        if layer_indices is not None:
            if len(layer_indices) != n_layers:
                raise ValueError(
                    f"layer_indices length ({len(layer_indices)}) must match n_layers={n_layers}"
                )
            resolved_layer_indices = [
                idx if idx >= 0 else original_n + idx
                for idx in layer_indices
            ]
            if any(idx < 0 or idx >= original_n for idx in resolved_layer_indices):
                raise ValueError(
                    f"layer_indices={layer_indices} must resolve to [0, {original_n - 1}]"
                )
            selected_layers = [self.model.model.layers[idx] for idx in resolved_layer_indices]
            self.layer_indices: List[int] = list(resolved_layer_indices)
        else:
            selected_layers = list(self.model.model.layers)[:n_layers]
            self.layer_indices = list(range(n_layers))

        self.model.model.layers = nn.ModuleList(selected_layers)
        self.model.config.num_hidden_layers = n_layers
        self.n_layers = n_layers
        self.original_n_layers = original_n

        if head_lora_rank > 0:
            self.model.lm_head = LoRALinear(self.model.lm_head, head_lora_rank)
            print(f"[ShallowLionQuery] Installed LoRA on lm_head (rank={head_lora_rank}, "
                  f"trainable={sum(p.numel() for p in self.model.lm_head.lora_A.parameters()) + sum(p.numel() for p in self.model.lm_head.lora_B.parameters()):,} params)")
        self.head_lora_rank = head_lora_rank

    def freeze_for_warmup(self) -> None:
        """Freeze embeddings and LM head; only the kept body layers train."""
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.model.model.layers.parameters():
            p.requires_grad_(True)

    def unfreeze_all(self) -> None:
        """Unfreeze body + head LoRA adapter (if any); keep embed_tokens and base lm_head frozen.

        embed_tokens (262M) and the full lm_head weight (262M) each cost ~2 GB of AdamW
        optimizer state on an 8 GB GPU — always keep them frozen. Only the lightweight
        LoRA adapter (if installed) trains in place of the full lm_head.
        """
        for p in self.parameters():
            p.requires_grad_(True)
        # embed_tokens stays frozen — 262M params × 8 bytes AdamW = 2 GB alone
        self.model.model.embed_tokens.weight.requires_grad_(False)
        # lm_head base weight stays frozen if LoRA is installed
        if isinstance(self.model.lm_head, LoRALinear):
            self.model.lm_head.linear.weight.requires_grad_(False)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        override_k: int = 0,
    ) -> torch.Tensor:
        """Encode pre-tokenized inputs to SPLADE vectors ``[B, vocab_size]``."""
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits * (self.hidden_size ** -0.25)
        mask = attention_mask.unsqueeze(-1).float()
        logits = logits + (1.0 - mask) * -1e6
        return torch.log1p(torch.relu(logits.max(dim=1).values))


class ShallowFactorizedLionQuery(ShallowLionQuery):
    """Shallow Lion query with ALBERT-style tied lexical factors.

    Replaces the huge Llama input embedding / LM-head lexical matrix with shared
    factors. The input path computes ``token_id -> A -> B`` and the output path
    computes ``hidden -> B.T -> A.T``. This keeps the Lion vocabulary unchanged
    while cutting the query-side lexical parameter floor.
    """

    def __init__(
        self,
        hf_id: str,
        n_layers: int,
        factorized_embedding_dim: int = 128,
        init: str = "svd",
        layer_indices: Optional[List[int]] = None,
    ):
        super().__init__(hf_id, n_layers, layer_indices=layer_indices)
        self.factorized_embedding_dim = int(factorized_embedding_dim)
        self._install_factorized_embeddings(init=init)

    def _install_factorized_embeddings(self, init: str = "svd") -> None:
        old_embed = self.model.model.embed_tokens
        old_lm_head = self.model.lm_head
        old_weight = old_lm_head.weight.detach()
        if old_embed.weight.shape == old_lm_head.weight.shape:
            # Prefer the output head: Lion retrieval quality depends directly on
            # this lexical basis, and many Llama checkpoints do not strictly tie
            # input/output weights in config even when shapes match.
            init_weight = old_weight
        else:
            raise ValueError(
                "Lion embed_tokens and lm_head shapes differ; cannot install tied factorization"
            )

        factorized = FactorizedWordEmbeddings(
            old_embed.num_embeddings,
            old_embed.embedding_dim,
            self.factorized_embedding_dim,
            padding_idx=old_embed.padding_idx,
        ).to(device=old_embed.weight.device, dtype=old_embed.weight.dtype)

        if init in {"svd", "svd_lowrank", "lowrank_svd", "randomized_svd"}:
            if init == "svd":
                A, B = _factorize_embedding_matrix(init_weight, self.factorized_embedding_dim)
            else:
                A, B = _factorize_embedding_matrix_lowrank(init_weight, self.factorized_embedding_dim)
            factorized.lexical_embeddings.weight.data.copy_(A.to(init_weight.device, dtype=init_weight.dtype))
            factorized.up_project.weight.data.copy_(B.T.to(init_weight.device, dtype=init_weight.dtype))
        elif init == "random":
            init_std = float(getattr(self.model.config, "initializer_range", 0.02))
            nn.init.normal_(factorized.lexical_embeddings.weight, mean=0.0, std=init_std)
            nn.init.normal_(factorized.up_project.weight, mean=0.0, std=init_std)
        else:
            raise ValueError(f"Unknown factorized embedding init: {init}")

        self.model.model.embed_tokens = factorized
        self.model.lm_head = FactorizedBertDecoder(factorized, bias=None).to(
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype,
        )
        self.model.config.tie_word_embeddings = True

    def unfreeze_all(self) -> None:
        """Unfreeze everything — factorized embed/head is small enough to train freely."""
        for p in self.parameters():
            p.requires_grad_(True)

    def factorized_param_count(self) -> int:
        emb = self.model.model.embed_tokens
        return emb.lexical_embeddings.weight.numel() + emb.up_project.weight.numel()


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

    def encode_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return max-pooled MLM logits BEFORE relu and log1p — [B, vocab_size]."""
        hidden = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.mlm_head(self.proj(hidden)) * attention_mask.unsqueeze(-1).float()
        return logits.max(dim=1).values


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
