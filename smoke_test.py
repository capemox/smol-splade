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

from model import TopKSAE, SAEPretrainModel, sae_loss, SAESPLADEModel, splade_loss

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


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    ("SAE forward/backward", test_sae_forward_backward),
    ("remove_parallel_gradient + post_step", test_sae_post_step),
    ("SAEPretrainModel", test_sae_pretrain_model),
    ("SAESPLADEModel + splade_loss", test_splade_model),
    ("Full SAE training step", test_full_sae_step),
    ("Full SPLADE training step", test_full_splade_step),
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
