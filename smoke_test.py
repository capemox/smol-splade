#!/usr/bin/env python
"""Smoke test: verifies the full training pipeline runs without errors.

Uses tiny dimensions and fake data — no GPU or HuggingFace downloads needed.
Runs in ~10 seconds on CPU.

    uv run smoke_test.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import TopKSAE, SAEPretrainModel, sae_loss, SAESPLADEModel, splade_loss, FrozenDocSPLADE, asymmetric_splade_loss, ProjectedQuerySPLADE, projected_alignment_loss, VocabTransplantQuerySPLADE, vocab_transplant_splade_loss

HIDDEN = 64
SAE_WIDTH = 128
K = 4
AUX_K = 8
BATCH = 4
SEQ_LEN = 16
NWAY = 2
DEVICE = torch.device("cpu")


def make_fake_tokens(n: int) -> torch.Tensor:
    return torch.randn(n, HIDDEN)


def make_fake_batch(b: int, seq: int):
    ids = torch.randint(0, 100, (b, seq))
    mask = torch.ones(b, seq, dtype=torch.long)
    return ids, mask


def make_fake_hidden(b: int, seq: int) -> torch.Tensor:
    return torch.randn(b, seq, HIDDEN)


# ── Helpers to monkey-patch backbone ─────────────────────────────────────────

class FakeBackboneOutput:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class FakeBackbone(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None):
        b, l = input_ids.shape
        return FakeBackboneOutput(torch.randn(b, l, HIDDEN))


# ── Test 1: TopKSAE forward / backward ───────────────────────────────────────

def test_sae_forward_backward():
    sae = TopKSAE(HIDDEN, SAE_WIDTH, K, AUX_K)
    x = make_fake_tokens(BATCH * SEQ_LEN)
    out = sae(x)

    assert out["x_rec"].shape == x.shape, "x_rec shape mismatch"
    assert out["inds"].shape == (BATCH * SEQ_LEN, K), "inds shape mismatch"
    assert out["sparsity"].item() <= K, "sparsity exceeds k"

    loss, metrics = sae_loss(out)
    loss.backward()

    assert sae.W_enc.grad is not None, "W_enc has no gradient"
    assert sae.W_dec.grad is not None, "W_dec has no gradient"
    print(f"  [PASS] SAE forward/backward — loss={metrics['loss']:.4f}, sparsity={metrics['sparsity']:.1f}")


# ── Test 2: remove_parallel_gradient + post_step ─────────────────────────────

def test_sae_post_step():
    sae = TopKSAE(HIDDEN, SAE_WIDTH, K, AUX_K)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=1e-4)

    x = make_fake_tokens(BATCH)
    out = sae(x)
    loss, _ = sae_loss(out)
    loss.backward()

    sae.remove_parallel_gradient()
    optimizer.step()
    sae.post_step()

    norms = F.normalize(sae.W_dec.data, dim=-1)
    diff = (sae.W_dec.data - norms).abs().max().item()
    assert diff < 1e-5, f"W_dec columns not unit norm after post_step (max_diff={diff})"
    print(f"  [PASS] remove_parallel_gradient + post_step — W_dec unit norm OK")


# ── Test 3: SAEPretrainModel with fake backbone ───────────────────────────────

def test_sae_pretrain_model():
    sae = TopKSAE(HIDDEN, SAE_WIDTH, K, AUX_K)
    model = SAEPretrainModel.__new__(SAEPretrainModel)
    torch.nn.Module.__init__(model)
    model.backbone = FakeBackbone()
    model.sae = sae

    ids, mask = make_fake_batch(BATCH, SEQ_LEN)
    out = model(ids, mask)
    loss, metrics = sae_loss(out)
    loss.backward()
    print(f"  [PASS] SAEPretrainModel forward/backward — loss={metrics['loss']:.4f}")


# ── Test 4: SAESPLADEModel encode + splade_loss ───────────────────────────────

def test_splade_model():
    sae = TopKSAE(HIDDEN, SAE_WIDTH, K, AUX_K)
    model = SAESPLADEModel.__new__(SAESPLADEModel)
    torch.nn.Module.__init__(model)
    model.backbone = FakeBackbone()
    model.sae = sae
    model.scale = True
    model.alpha = torch.nn.Parameter(torch.ones(1))

    B = BATCH
    q_ids, q_mask = make_fake_batch(B, 8)
    d_ids, d_mask = make_fake_batch(B * NWAY, SEQ_LEN)

    # Without teacher scores (CE loss)
    loss, metrics = splade_loss(
        model, q_ids, q_mask, d_ids, d_mask,
        teacher_scores=None,
        lambda_d=0.04, lambda_q=0.06, flops_scale=1.0,
    )
    loss.backward()
    assert loss.item() > 0, "loss should be positive"
    print(f"  [PASS] splade_loss (no teacher) — loss={metrics['loss']:.4f}, "
          f"d_nnz={metrics['avg_d_nnz']:.1f}, q_nnz={metrics['avg_q_nnz']:.1f}")

    # With teacher scores (KL + MSE loss)
    teacher_scores = torch.randn(B, NWAY)
    loss2, metrics2 = splade_loss(
        model, q_ids, q_mask, d_ids, d_mask,
        teacher_scores=teacher_scores,
        lambda_d=0.04, lambda_q=0.06, flops_scale=0.5,
    )
    loss2.backward()
    print(f"  [PASS] splade_loss (with teacher) — loss={metrics2['loss']:.4f}")


# ── Test 5: Full SAE training step (optimizer + schedulers) ──────────────────

def test_full_sae_step():
    from transformers import get_linear_schedule_with_warmup

    sae = TopKSAE(HIDDEN, SAE_WIDTH, K, AUX_K)
    model = SAEPretrainModel.__new__(SAEPretrainModel)
    torch.nn.Module.__init__(model)
    model.backbone = FakeBackbone()
    model.sae = sae

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-5
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, 10, 100)

    for step in range(3):
        ids, mask = make_fake_batch(BATCH, SEQ_LEN)
        optimizer.zero_grad()
        out = model(ids, mask)
        loss, metrics = sae_loss(out)
        loss.backward()
        model.sae.remove_parallel_gradient()
        optimizer.step()
        scheduler.step()
        model.sae.post_step()

    print(f"  [PASS] Full SAE training loop (3 steps) — final loss={metrics['loss']:.4f}")


# ── Test 6: Full SPLADE training step ────────────────────────────────────────

def test_full_splade_step():
    from transformers import get_linear_schedule_with_warmup

    sae = TopKSAE(HIDDEN, SAE_WIDTH, K, AUX_K)
    model = SAESPLADEModel.__new__(SAESPLADEModel)
    torch.nn.Module.__init__(model)
    model.backbone = FakeBackbone()
    model.sae = sae
    model.scale = True
    model.alpha = torch.nn.Parameter(torch.ones(1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, 10, 100)

    for step in range(3):
        flops_scale = min(1.0, step / 2)
        q_ids, q_mask = make_fake_batch(BATCH, 8)
        d_ids, d_mask = make_fake_batch(BATCH * NWAY, SEQ_LEN)
        optimizer.zero_grad()
        loss, metrics = splade_loss(
            model, q_ids, q_mask, d_ids, d_mask,
            teacher_scores=None,
            lambda_d=0.04, lambda_q=0.06, flops_scale=flops_scale,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.sae.post_step()

    print(f"  [PASS] Full SPLADE training loop (3 steps) — final loss={metrics['loss']:.4f}")


# ── Test 7: evaluate_nanobeir with fake model + fake data ────────────────────

def test_eval():
    from eval import _ndcg_at_k, evaluate_nanobeir
    from unittest.mock import MagicMock, patch

    # Test NDCG@10 computation
    qrels = {"q1": {"d1": 1, "d2": 0}, "q2": {"d3": 1}}
    ranked = [["d1", "d3", "d2"], ["d3", "d1", "d2"]]
    query_ids = ["q1", "q2"]
    ndcg = _ndcg_at_k(ranked, qrels, query_ids, k=10)
    assert 0.0 < ndcg <= 1.0, f"NDCG out of range: {ndcg}"
    assert abs(ndcg - 1.0) < 1e-6, f"Expected perfect NDCG=1.0, got {ndcg}"
    print(f"  [PASS] NDCG@10 computation — NDCG={ndcg:.4f}")

    # Test evaluate_nanobeir end-to-end with fake data
    sae = TopKSAE(HIDDEN, SAE_WIDTH, K, AUX_K)
    model = SAESPLADEModel.__new__(SAESPLADEModel)
    torch.nn.Module.__init__(model)
    model.backbone = FakeBackbone()
    model.sae = sae
    model.scale = True
    model.alpha = torch.nn.Parameter(torch.ones(1))

    fake_corpus = [{"_id": f"d{i}", "text": f"doc {i}"} for i in range(10)]
    fake_queries = [{"_id": f"q{i}", "text": f"query {i}"} for i in range(3)]
    fake_qrels = [{"query_id": "q0", "doc_id": "d0", "score": 1},
                  {"query_id": "q1", "doc_id": "d1", "score": 1}]

    def fake_load_dataset(name, split=None):
        from torch.utils.data import Dataset
        class FakeDS(list):
            def __getitem__(self, key):
                if isinstance(key, str):
                    return [row[key] for row in list.__iter__(self)]
                return list.__getitem__(self, key)
            def __iter__(self):
                return list.__iter__(self)
        if split == "corpus":  return FakeDS(fake_corpus)
        if split == "queries": return FakeDS(fake_queries)
        if split == "qrels":   return FakeDS(fake_qrels)

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {
        "input_ids": torch.randint(0, 100, (3, 8)),
        "attention_mask": torch.ones(3, 8, dtype=torch.long),
    }
    fake_tokenizer.side_effect = lambda texts, **kw: {
        "input_ids": torch.randint(0, 100, (len(texts), 8)),
        "attention_mask": torch.ones(len(texts), 8, dtype=torch.long),
    }

    cfg = {"splade": {"doc_max_length": 8, "query_max_length": 8},
           "eval": {"datasets": ["fake/dataset"], "batch_size": 4}}

    with patch("eval.load_nanobeir") as mock_load:
        mock_load.return_value = (
            [r["_id"] for r in fake_corpus],
            [r["text"] for r in fake_corpus],
            [r["_id"] for r in fake_queries],
            [r["text"] for r in fake_queries],
            {"q0": {"d0": 1}, "q1": {"d1": 1}},
        )
        results = evaluate_nanobeir(model, fake_tokenizer, cfg, DEVICE)

    assert "fake/dataset".split("/")[-1] in results, "dataset not in results"
    print(f"  [PASS] evaluate_nanobeir end-to-end — NDCG@10={list(results.values())[0]:.4f}")


# ── Test 8: Asymmetric loss + collate_asymmetric_batch ───────────────────────

def test_asymmetric():
    from data import collate_asymmetric_batch
    from unittest.mock import MagicMock

    VOCAB = 64   # plays the role of doc_splade vocab_size; matches SAE width

    # ── Fake FrozenDocSPLADE ──────────────────────────────────────────
    class FakeFrozenDocSPLADE(torch.nn.Module):
        vocab_size = VOCAB
        def encode(self, texts, max_length):
            return torch.relu(torch.randn(len(texts), VOCAB))

    # ── Query SAE-SPLADE with sae_width = VOCAB ───────────────────────
    sae = TopKSAE(HIDDEN, VOCAB, K, AUX_K)
    query_model = SAESPLADEModel.__new__(SAESPLADEModel)
    torch.nn.Module.__init__(query_model)
    query_model.backbone = FakeBackbone()
    query_model.sae = sae
    query_model.scale = True
    query_model.alpha = torch.nn.Parameter(torch.ones(1))

    doc_splade = FakeFrozenDocSPLADE()

    # Verify no doc_splade params require grad
    assert all(not p.requires_grad for p in doc_splade.parameters()), \
        "doc_splade params should be frozen"

    # ── collate_asymmetric_batch ──────────────────────────────────────
    fake_tokenizer = MagicMock()
    fake_tokenizer.side_effect = lambda texts, **kw: {
        "input_ids": torch.randint(0, 100, (len(texts), 8)),
        "attention_mask": torch.ones(len(texts), 8, dtype=torch.long),
    }

    items = [
        {
            "query": f"query {i}",
            "passages": [{"text": f"passage {i}_{j}"} for j in range(NWAY)],
            "teacher_scores": [float(j) for j in range(NWAY)],
        }
        for i in range(BATCH)
    ]
    q_ids, q_mask, doc_texts, teacher_scores = collate_asymmetric_batch(
        items, fake_tokenizer, query_max_length=8, device=DEVICE
    )
    assert q_ids.shape == (BATCH, 8), f"q_ids shape wrong: {q_ids.shape}"
    assert len(doc_texts) == BATCH * NWAY, f"expected {BATCH*NWAY} doc texts, got {len(doc_texts)}"
    assert teacher_scores.shape == (BATCH, NWAY), f"teacher_scores shape wrong"
    assert doc_texts[0] == "passage 0_0", f"unexpected doc text: {doc_texts[0]}"

    # ── asymmetric_splade_loss (with teacher) ─────────────────────────
    doc_vecs = doc_splade.encode(doc_texts, max_length=128)   # [B*nway, VOCAB]
    assert doc_vecs.shape == (BATCH * NWAY, VOCAB)

    loss, metrics = asymmetric_splade_loss(
        query_model, q_ids, q_mask, doc_vecs, teacher_scores,
        lambda_q=0.06, flops_scale=1.0,
    )
    loss.backward()
    assert loss.item() > 0
    assert "avg_q_nnz" in metrics
    assert "avg_d_nnz" not in metrics, "doc FLOPs should not be tracked in asymmetric loss"

    # ── asymmetric_splade_loss (no teacher, CE fallback) ─────────────
    loss2, metrics2 = asymmetric_splade_loss(
        query_model, q_ids, q_mask, doc_vecs, teacher_scores=None,
        lambda_q=0.06, flops_scale=0.5,
    )
    loss2.backward()
    assert loss2.item() > 0

    print(
        f"  [PASS] asymmetric loss (teacher) — loss={metrics['loss']:.4f}, "
        f"q_nnz={metrics['avg_q_nnz']:.1f}"
    )
    print(f"  [PASS] asymmetric loss (CE fallback) — loss={metrics2['loss']:.4f}")
    print(f"  [PASS] collate_asymmetric_batch — {len(doc_texts)} doc texts returned as strings")


# ── Test 9: ProjectedQuerySPLADE end-to-end ──────────────────────────────────

def test_projected():
    from unittest.mock import MagicMock, patch
    from eval import evaluate_asymmetric

    VOCAB = 64
    SPLADE_H = 32

    # ── Shared fakes ──────────────────────────────────────────────────
    class FakeMLMHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_transform = torch.nn.Linear(SPLADE_H, SPLADE_H)
            self.vocab_layer_norm = torch.nn.LayerNorm(SPLADE_H)
            self.vocab_projector = torch.nn.Linear(SPLADE_H, VOCAB)
        def forward(self, x):
            return self.vocab_projector(self.vocab_layer_norm(F.gelu(self.vocab_transform(x))))

    class FakeFrozenDocSPLADE(torch.nn.Module):
        vocab_size = VOCAB
        def encode(self, texts, max_length):
            return torch.relu(torch.randn(len(texts), VOCAB))

    def make_query_model():
        m = ProjectedQuerySPLADE.__new__(ProjectedQuerySPLADE)
        torch.nn.Module.__init__(m)
        m.backbone = FakeBackbone()
        m.proj = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN, SPLADE_H),
            torch.nn.GELU(),
            torch.nn.LayerNorm(SPLADE_H),
            torch.nn.Linear(SPLADE_H, SPLADE_H),
        )
        m.mlm_head = FakeMLMHead()
        for p in m.mlm_head.parameters():
            p.requires_grad_(False)
        m.vocab_size = VOCAB
        return m

    doc_splade = FakeFrozenDocSPLADE()
    query_model = make_query_model()
    q_ids, q_mask = make_fake_batch(BATCH, SEQ_LEN)
    texts = [f"text {i}" for i in range(BATCH)]

    # ── 1. encode: shape and non-negativity ──────────────────────────
    vecs = query_model.encode(q_ids, q_mask)
    assert vecs.shape == (BATCH, VOCAB), f"encode shape wrong: {vecs.shape}"
    assert (vecs >= 0).all(), "SPLADE output must be non-negative"
    print(f"  [PASS] encode — shape={tuple(vecs.shape)}, non-negative=True")

    # ── 2. alignment loss: value range, grad flow, frozen MLM head ───
    loss_align = projected_alignment_loss(query_model, doc_splade, q_ids, q_mask, texts, doc_max_length=128)
    loss_align.backward()
    assert 0.0 <= loss_align.item() <= 2.0, f"cosine loss out of [0,2]: {loss_align.item()}"
    for p in query_model.mlm_head.parameters():
        assert p.grad is None, "mlm_head must stay frozen (no grad)"
    first_linear = next(m for m in query_model.proj.modules() if isinstance(m, torch.nn.Linear))
    assert first_linear.weight.grad is not None, "proj must receive grad"
    print(f"  [PASS] alignment loss — cosine={loss_align.item():.4f}, proj has grad, mlm_head frozen")

    # ── 3. ranking loss (with teacher + CE fallback) ──────────────────
    query_model.zero_grad()
    doc_vecs = doc_splade.encode([f"p{i}" for i in range(BATCH * NWAY)], 128)
    teacher = torch.randn(BATCH, NWAY)
    loss_rank, metrics = asymmetric_splade_loss(
        query_model, q_ids, q_mask, doc_vecs, teacher, lambda_q=0.04, flops_scale=1.0
    )
    loss_rank.backward()
    assert loss_rank.item() > 0
    assert "avg_q_nnz" in metrics and "avg_d_nnz" not in metrics
    for p in query_model.mlm_head.parameters():
        assert p.grad is None, "mlm_head must stay frozen during ranking loss too"
    loss_ce, _ = asymmetric_splade_loss(
        query_model, q_ids, q_mask, doc_vecs, None, lambda_q=0.04, flops_scale=0.5
    )
    assert loss_ce.item() > 0
    print(f"  [PASS] ranking loss — teacher={loss_rank.item():.4f}, CE={loss_ce.item():.4f}")

    # ── 4. evaluate_asymmetric with ProjectedQuerySPLADE ─────────────
    query_model2 = make_query_model()
    fake_tokenizer = MagicMock()
    fake_tokenizer.side_effect = lambda texts, **kw: {
        "input_ids": torch.randint(0, 100, (len(texts), 8)),
        "attention_mask": torch.ones(len(texts), 8, dtype=torch.long),
    }
    cfg = {
        "projected": {"doc_max_length": 8, "query_max_length": 8},
        "eval": {"datasets": ["fake/ds"], "batch_size": 4},
    }
    fake_corpus  = [{"_id": f"d{i}", "text": f"doc {i}"}   for i in range(10)]
    fake_queries = [{"_id": f"q{i}", "text": f"query {i}"} for i in range(3)]

    with patch("eval.load_nanobeir") as mock_load:
        mock_load.return_value = (
            [r["_id"] for r in fake_corpus],
            [r["text"] for r in fake_corpus],
            [r["_id"] for r in fake_queries],
            [r["text"] for r in fake_queries],
            {"q0": {"d0": 1}, "q1": {"d1": 1}},
        )
        results = evaluate_asymmetric(
            query_model2, fake_tokenizer, doc_splade, cfg, DEVICE,
            run_doc_doc=True, override_k=0, section="projected",
        )

    assert "ds" in results, f"expected 'ds' key, got {list(results.keys())}"
    assert "query_doc" in results["ds"]
    assert "doc_doc"   in results["ds"]
    print(f"  [PASS] evaluate_asymmetric — query_doc={results['ds']['query_doc']:.4f}, "
          f"doc_doc={results['ds']['doc_doc']:.4f}")

    # ── 5. full alignment warm-up loop (3 steps) ─────────────────────
    query_model3 = make_query_model()
    align_opt = torch.optim.AdamW(query_model3.proj.parameters(), lr=1e-3)
    query_model3.backbone.requires_grad_(False)
    for _ in range(3):
        l = projected_alignment_loss(query_model3, doc_splade, q_ids, q_mask, texts, doc_max_length=8)
        align_opt.zero_grad()
        l.backward()
        align_opt.step()
    query_model3.backbone.requires_grad_(True)
    assert all(p.requires_grad for p in query_model3.backbone.parameters()), \
        "backbone should be unfrozen after warm-up"
    print(f"  [PASS] alignment warm-up loop (3 steps) — final cosine loss={l.item():.4f}")

    # ── 6. full ranking fine-tuning loop (3 steps) ───────────────────
    from transformers import get_linear_schedule_with_warmup
    query_model4 = make_query_model()
    opt = torch.optim.AdamW(
        [p for p in query_model4.parameters() if p.requires_grad], lr=2e-5
    )
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=1, num_training_steps=3)
    for step in range(3):
        dv = doc_splade.encode([f"p{i}" for i in range(BATCH * NWAY)], 8)
        ts = torch.randn(BATCH, NWAY)
        loss, _ = asymmetric_splade_loss(
            query_model4, q_ids, q_mask, dv, ts,
            lambda_q=0.04, flops_scale=min(1.0, step / 2)
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(query_model4.parameters(), 1.0)
        opt.step()
        sched.step()
    print(f"  [PASS] ranking fine-tuning loop (3 steps) — final loss={loss.item():.4f}")


# ── Test 10: _run_tokensurgeon transplant logic ───────────────────────────────

def test_tokensurgeon_transplant():
    import importlib.util
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    # ── Import _run_tokensurgeon from train.py without running main() ─
    spec = importlib.util.spec_from_file_location(
        "train_mod", Path(__file__).parent / "train.py"
    )
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    Q_V, D_V, H_q, H_d = 8, 6, 4, 8

    # Known vocabularies: 3 shared tokens, 3 donor-only tokens
    query_tokens = ["apple", "banana", "cherry", "dog", "elephant", "fox", "grape", "honey"]
    donor_tokens = ["apple", "banana", "cherry", "kiwi", "lemon", "mango"]
    # Shared:      apple(q=0,d=0)  banana(q=1,d=1)  cherry(q=2,d=2)
    # Donor-only:  kiwi(d=3)  lemon(d=4)  mango(d=5)

    # ── Fake MLM model that tracks embedding changes ──────────────────
    class FakeMLMForTransplant(torch.nn.Module):
        def __init__(self, vocab_size, hidden):
            super().__init__()
            self.config = SimpleNamespace(vocab_size=vocab_size)
            self._embed = torch.nn.Embedding(vocab_size, hidden)

        def get_input_embeddings(self):
            return self._embed

        def resize_token_embeddings(self, new_size):
            old = self._embed
            new_e = torch.nn.Embedding(new_size, old.embedding_dim)
            n = min(old.num_embeddings, new_size)
            new_e.weight.data[:n] = old.weight.data[:n]
            self._embed = new_e
            self.config.vocab_size = new_size

        def save_pretrained(self, path):
            import json
            Path(path).mkdir(parents=True, exist_ok=True)
            with open(Path(path) / "config.json", "w") as f:
                json.dump({"vocab_size": self.config.vocab_size}, f)

    class FakeTokenizerForTransplant:
        def __init__(self, tokens):
            self._vocab = {t: i for i, t in enumerate(tokens)}

        def get_vocab(self):
            return dict(self._vocab)

        def save_pretrained(self, path):
            pass

    torch.manual_seed(42)
    query_model_inst = FakeMLMForTransplant(Q_V, H_q)
    donor_model_inst = FakeMLMForTransplant(D_V, H_d)
    # Set query model's pad_token_id to a value that is valid in its own
    # vocab (Q_V=8) but out-of-range for the donor vocab (D_V=6).
    # After transplant the config must be updated to the donor's value.
    query_model_inst.config.pad_token_id = 7   # 7 < 8=Q_V but 7 >= 6=D_V → crash if not fixed
    query_tok_inst   = FakeTokenizerForTransplant(query_tokens)
    donor_tok_inst   = FakeTokenizerForTransplant(donor_tokens)
    donor_tok_inst.pad_token_id  = 0
    donor_tok_inst.mask_token_id = 4

    # Capture original query embedding before the transplant modifies it
    orig_query_embed = query_model_inst._embed.weight.data.clone()

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "transplanted"

        with patch("transformers.AutoModelForMaskedLM") as mock_mlm_cls, \
             patch("transformers.AutoTokenizer") as mock_tok_cls:
            mock_mlm_cls.from_pretrained.side_effect = [query_model_inst, donor_model_inst]
            mock_tok_cls.from_pretrained.side_effect = [query_tok_inst, donor_tok_inst]

            train_mod._run_tokensurgeon("fake/query", "fake/donor", str(out_path), k=3)

        final_embed = query_model_inst._embed.weight.data  # [D_V, H_q]

        # Shape
        assert final_embed.shape == (D_V, H_q), f"wrong shape: {final_embed.shape}"

        # config.json created (the skip-if-exists check relies on this)
        assert (out_path / "config.json").exists(), "config.json not written"

        # Exact-match tokens must be copied verbatim from the query embedding
        for token, d_idx, q_idx in [("apple", 0, 0), ("banana", 1, 1), ("cherry", 2, 2)]:
            assert torch.allclose(final_embed[d_idx], orig_query_embed[q_idx], atol=1e-6), \
                f"'{token}' was not copied exactly (d_idx={d_idx}, q_idx={q_idx})"

        # Donor-only tokens must be approximated (non-zero) not left at zero
        for token, d_idx in [("kiwi", 3), ("lemon", 4), ("mango", 5)]:
            assert final_embed[d_idx].abs().sum().item() > 0, \
                f"'{token}' embedding is all-zeros — approximation did not run"

        # Special-token IDs must be updated to the donor's values.
        # The original query model had pad_token_id=7, which is >= D_V=6
        # and would cause nn.Embedding to crash with "Padding_idx must be
        # within num_embeddings" when the transplanted model is reloaded.
        assert query_model_inst.config.pad_token_id == 0, \
            "pad_token_id not updated to donor value — reload will crash"
        assert query_model_inst.config.mask_token_id == 4, \
            "mask_token_id not updated to donor value"

    print(
        f"  [PASS] tokensurgeon — shape=({D_V},{H_q}), "
        f"3 exact copies verified, 3 donor-only tokens approximated, "
        f"special token IDs updated"
    )


# ── Test 11: VocabTransplantQuerySPLADE encode + gradients ───────────────────

def test_vocab_transplant_model():
    VOCAB = 64

    class FakeMLMOutput:
        def __init__(self, logits):
            self.logits = logits

    class FakeMLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(HIDDEN, VOCAB)

        def forward(self, input_ids, attention_mask=None):
            B, L = input_ids.shape
            # Deterministic hidden states driven by proj so grads flow
            h = torch.ones(B, L, HIDDEN)
            return FakeMLMOutput(self.proj(h))

    # Construct without downloading any real model (same __new__ pattern as above)
    vt = VocabTransplantQuerySPLADE.__new__(VocabTransplantQuerySPLADE)
    torch.nn.Module.__init__(vt)
    vt.mlm = FakeMLM()
    vt.vocab_size = VOCAB

    q_ids, q_mask = make_fake_batch(BATCH, SEQ_LEN)

    # ── 1. Shape + non-negativity (log1p(relu(...)) ≥ 0) ─────────────
    vecs = vt.encode(q_ids, q_mask)
    assert vecs.shape == (BATCH, VOCAB), f"wrong encode shape: {vecs.shape}"
    assert (vecs >= 0).all(), "SPLADE output must be non-negative"
    print(f"  [PASS] encode — shape={tuple(vecs.shape)}, non-negative=True")

    # ── 2. Padding mask zeros out masked positions ────────────────────
    zero_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long)
    vecs_zero = vt.encode(q_ids, zero_mask)
    assert (vecs_zero == 0).all(), "all-zero attention_mask should yield all-zero SPLADE vector"
    print(f"  [PASS] padding mask — zero mask → zero output")

    # ── 3. Gradient flows through the full model ──────────────────────
    vt.zero_grad()
    vt.encode(q_ids, q_mask).sum().backward()
    assert vt.mlm.proj.weight.grad is not None, "grad must reach mlm.proj.weight"
    print(f"  [PASS] gradient flow — grad reaches mlm.proj.weight")

    # ── 4. vocab_transplant_splade_loss (with teacher) ────────────────
    vt.zero_grad()
    doc_vecs = torch.relu(torch.randn(BATCH * NWAY, VOCAB))
    teacher  = torch.randn(BATCH, NWAY)
    loss, metrics = vocab_transplant_splade_loss(
        vt, q_ids, q_mask, doc_vecs, teacher, lambda_q=0.1, flops_scale=1.0
    )
    loss.backward()
    assert loss.item() > 0 and not torch.isnan(loss), f"invalid loss: {loss.item()}"
    assert "avg_q_nnz" in metrics and "avg_d_nnz" not in metrics
    assert vt.mlm.proj.weight.grad is not None, "grad must flow through ranking loss"
    # Key property: FLOPs term must carry a non-zero gradient to sparsify a dense model
    vt.zero_grad()
    q_vecs = vt.encode(q_ids, q_mask)
    flops_only = (q_vecs.mean(dim=0) ** 2).sum()
    flops_only.backward()
    assert vt.mlm.proj.weight.grad is not None, "FLOPs term must have gradient"
    assert vt.mlm.proj.weight.grad.abs().sum().item() > 0, \
        "FLOPs gradient is zero — sparsification will not work"
    print(f"  [PASS] vocab_transplant_splade_loss (teacher) — loss={loss.item():.4f}, "
          f"FLOPs grad is non-zero")

    # ── 5. vocab_transplant_splade_loss (CE fallback, no teacher) ─────
    vt.zero_grad()
    loss_ce, _ = vocab_transplant_splade_loss(
        vt, q_ids, q_mask, doc_vecs, None, lambda_q=0.1, flops_scale=0.5
    )
    loss_ce.backward()
    assert loss_ce.item() > 0
    print(f"  [PASS] vocab_transplant_splade_loss (CE fallback) — loss={loss_ce.item():.4f}")


# ── Test 12: Full vocab_transplant training loop + evaluate_asymmetric ────────

def test_full_vocab_transplant_loop():
    from transformers import get_linear_schedule_with_warmup
    from unittest.mock import MagicMock, patch
    from eval import evaluate_asymmetric
    from data import collate_asymmetric_batch

    VOCAB = 64

    class FakeMLMOutput:
        def __init__(self, logits):
            self.logits = logits

    class FakeMLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(HIDDEN, VOCAB)

        def forward(self, input_ids, attention_mask=None):
            B, L = input_ids.shape
            h = torch.ones(B, L, HIDDEN)
            return FakeMLMOutput(self.proj(h))

    class FakeFrozenDocSPLADE(torch.nn.Module):
        vocab_size = VOCAB

        def encode(self, texts, max_length):
            return torch.relu(torch.randn(len(texts), VOCAB))

    # ── Build models ──────────────────────────────────────────────────
    vt = VocabTransplantQuerySPLADE.__new__(VocabTransplantQuerySPLADE)
    torch.nn.Module.__init__(vt)
    vt.mlm = FakeMLM()
    vt.vocab_size = VOCAB

    doc_splade = FakeFrozenDocSPLADE()

    # ── collate_asymmetric_batch produces the right shapes ────────────
    fake_tok = MagicMock()
    fake_tok.side_effect = lambda texts, **kw: {
        "input_ids":      torch.randint(0, 100, (len(texts), 8)),
        "attention_mask": torch.ones(len(texts), 8, dtype=torch.long),
    }
    items = [
        {
            "query": f"q{i}",
            "passages": [{"text": f"p{i}_{j}"} for j in range(NWAY)],
            "teacher_scores": [float(j) for j in range(NWAY)],
        }
        for i in range(BATCH)
    ]
    q_ids, q_mask, doc_texts, teacher_scores = collate_asymmetric_batch(
        items, fake_tok, query_max_length=8, device=DEVICE
    )
    assert q_ids.shape   == (BATCH, 8),          f"q_ids shape wrong: {q_ids.shape}"
    assert len(doc_texts) == BATCH * NWAY,        "wrong number of doc texts"
    assert teacher_scores.shape == (BATCH, NWAY), f"teacher_scores shape wrong"
    print(f"  [PASS] collate_asymmetric_batch — {len(doc_texts)} doc texts, shapes OK")

    # ── 3-step optimizer loop (mirrors train_vocab_transplant internals) ─
    optimizer = torch.optim.AdamW(vt.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=3)

    vt.train()
    losses = []
    for step in range(3):
        flops_scale = min(1.0, step / 2.0)
        doc_vecs = doc_splade.encode(doc_texts, max_length=128)

        optimizer.zero_grad()
        loss, metrics = vocab_transplant_splade_loss(
            vt, q_ids, q_mask, doc_vecs, teacher_scores,
            lambda_q=0.1, flops_scale=flops_scale,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vt.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    assert all(not torch.isnan(torch.tensor(l)) for l in losses), "NaN loss during training"
    assert all(l > 0 for l in losses), "loss must be positive"
    print(f"  [PASS] training loop (3 steps) — losses={[f'{l:.4f}' for l in losses]}")

    # ── evaluate_asymmetric with VocabTransplantQuerySPLADE ───────────
    # Build a fresh model for eval to avoid any stale grad state
    vt_eval = VocabTransplantQuerySPLADE.__new__(VocabTransplantQuerySPLADE)
    torch.nn.Module.__init__(vt_eval)
    vt_eval.mlm = FakeMLM()
    vt_eval.vocab_size = VOCAB

    fake_corpus  = [{"_id": f"d{i}", "text": f"doc {i}"}   for i in range(10)]
    fake_queries = [{"_id": f"q{i}", "text": f"query {i}"} for i in range(3)]
    cfg = {
        "vocab_transplant": {"doc_max_length": 8, "query_max_length": 8},
        "eval": {"datasets": ["fake/ds"], "batch_size": 4},
    }

    with patch("eval.load_nanobeir") as mock_load:
        mock_load.return_value = (
            [r["_id"] for r in fake_corpus],
            [r["text"] for r in fake_corpus],
            [r["_id"] for r in fake_queries],
            [r["text"] for r in fake_queries],
            {"q0": {"d0": 1}, "q1": {"d1": 1}},
        )
        results = evaluate_asymmetric(
            vt_eval, fake_tok, doc_splade, cfg, DEVICE,
            run_doc_doc=True, override_k=0, section="vocab_transplant",
        )

    assert "ds" in results,              f"expected 'ds' key, got {list(results.keys())}"
    assert "query_doc" in results["ds"], "query_doc result missing"
    assert "doc_doc"   in results["ds"], "doc_doc result missing"
    print(
        f"  [PASS] evaluate_asymmetric — "
        f"query_doc={results['ds']['query_doc']:.4f}, "
        f"doc_doc={results['ds']['doc_doc']:.4f}"
    )


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    ("SAE forward/backward", test_sae_forward_backward),
    ("remove_parallel_gradient + post_step", test_sae_post_step),
    ("SAEPretrainModel", test_sae_pretrain_model),
    ("SAESPLADEModel + splade_loss", test_splade_model),
    ("Full SAE training step", test_full_sae_step),
    ("Full SPLADE training step", test_full_splade_step),
    ("evaluate_nanobeir", test_eval),
    ("Asymmetric loss + collate", test_asymmetric),
    ("ProjectedQuerySPLADE + alignment loss", test_projected),
    ("tokensurgeon transplant logic", test_tokensurgeon_transplant),
    ("VocabTransplantQuerySPLADE encode + grad", test_vocab_transplant_model),
    ("Full vocab_transplant loop + eval", test_full_vocab_transplant_loop),
]

if __name__ == "__main__":
    passed = failed = 0
    for name, fn in TESTS:
        print(f"\n{name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
