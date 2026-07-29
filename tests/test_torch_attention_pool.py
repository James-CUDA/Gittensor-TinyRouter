"""Tests for AttentivePool + AttentiveTriageRouter (encode→attention→head)."""
from __future__ import annotations

import numpy as np
import pytest


def _torch():
    return pytest.importorskip("torch", reason="torch required")


def _mod():
    from trinity.coordinator import attention_pool as ap

    return ap


def test_zero_query_is_uniform_mean_pool():
    torch = _torch()
    ap = _mod()
    pool = ap.AttentivePool(d_h=4)
    H = torch.tensor(
        [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]],
        dtype=torch.float32,
    )
    h, alpha = pool(H)
    assert torch.allclose(alpha, torch.full((3,), 1 / 3))
    assert torch.allclose(h, H.mean(dim=0))


def test_mask_drops_tokens():
    torch = _torch()
    ap = _mod()
    pool = ap.AttentivePool(d_h=2)
    H = torch.tensor([[10.0, 0.0], [0.0, 10.0], [0.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([True, False, False])
    h, alpha = pool(H, mask)
    assert torch.allclose(alpha, torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(h, H[0])


def test_attentive_triage_5_and_20():
    torch = _torch()
    ap = _mod()
    r5 = ap.make_attentive_triage("5-domain", d_h=8)
    r20 = ap.make_attentive_triage("20-domain", d_h=8)
    assert r5.head.n_domains == 5
    assert r20.head.n_domains == 20
    H = torch.randn(6, 8)
    dom, diff, alpha = r5(H)
    assert dom.shape == (5,)
    assert diff.shape == (5,)
    assert alpha.shape == (6,)
    domain, difficulty, dbg = r5.select(H, sample=False)
    assert domain in r5.head.domains
    assert 1 <= difficulty <= 5
    assert dbg["pool"] == "attentive"
    assert "alpha" in dbg


def test_batched_forward():
    torch = _torch()
    ap = _mod()
    pool = ap.AttentivePool(d_h=3)
    H = torch.randn(2, 5, 3)
    h, alpha = pool(H)
    assert h.shape == (2, 3)
    assert alpha.shape == (2, 5)


def test_pack_unpack():
    ap = _mod()
    pool = ap.AttentivePool(d_h=5, use_key_proj=True)
    rng = np.random.default_rng(1)
    theta = rng.normal(size=pool.n_params)
    pool.unpack_into(theta)
    assert np.allclose(pool.pack(), theta)
