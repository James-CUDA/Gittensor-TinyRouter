"""Offline coverage for Milestone-1 TriageHead (5-domain + 20-domain).

Filename keeps the ``test_torch_`` prefix so it sorts after
``test_shaped_fitness.py`` (torch-import invariant), same as
``test_torch_coordinator_head.py``.
"""
from __future__ import annotations

import numpy as np
import pytest


def _torch():
    return pytest.importorskip("torch", reason="torch required for TriageHead")


def _mod():
    from trinity.coordinator import triage_head as th

    return th


def test_5_domain_shape_and_zero_init():
    torch = _torch()
    th = _mod()
    head = th.TriageHead5Domain(d_h=8)
    assert head.domains == th.DOMAINS_5
    assert head.n_domains == 5
    assert head.n_diff == 5
    assert tuple(head.W_domain.shape) == (5, 8)
    assert tuple(head.W_diff.shape) == (5, 8)
    assert head.n_params == 5 * 8 + 5 * 8
    assert int(torch.count_nonzero(head.W_domain)) == 0
    assert int(torch.count_nonzero(head.W_diff)) == 0


def test_20_domain_shape_matches_gci():
    th = _mod()
    head = th.TriageHead20Domain(d_h=4)
    assert head.n_domains == 20
    assert head.domains == th.DOMAINS_20
    assert "cooking" in head.domains
    assert "math" not in head.domains  # GCI-only; not the 5-domain bucket
    assert head.n_params == 20 * 4 + 5 * 4


def test_factory():
    th = _mod()
    assert isinstance(th.make_triage_head("5-domain"), th.TriageHead5Domain)
    assert isinstance(th.make_triage_head("20-domain"), th.TriageHead20Domain)
    assert isinstance(th.make_triage_head("gci"), th.TriageHead20Domain)
    with pytest.raises(ValueError, match="unknown triage"):
        th.make_triage_head("coarse")


def test_forward_split_and_select_argmax():
    torch = _torch()
    th = _mod()
    head = th.TriageHead5Domain(d_h=4)
    # Make domain logit 2 ("knowledge") and difficulty index 3 → label 4 win.
    Wd = np.zeros((5, 4), dtype=np.float32)
    Wf = np.zeros((5, 4), dtype=np.float32)
    Wd[2, :] = 10.0
    Wf[3, :] = 10.0
    head.load_weights(Wd, Wf)
    h = torch.ones(4)
    dom_logits, diff_logits = head.forward(h)
    assert dom_logits.shape == (5,)
    assert diff_logits.shape == (5,)
    domain, difficulty, dbg = head.select(h, sample=False)
    assert domain == "knowledge"
    assert difficulty == 4
    assert dbg["sampled"] is False


def test_pack_unpack_roundtrip():
    th = _mod()
    head = th.TriageHead20Domain(d_h=3)
    rng = np.random.default_rng(0)
    Wd = rng.normal(size=(20, 3)).astype(np.float64)
    Wf = rng.normal(size=(5, 3)).astype(np.float64)
    head.load_weights(Wd, Wf)
    theta = head.pack()
    assert theta.shape == (head.n_params,)
    head2 = th.TriageHead20Domain(d_h=3)
    head2.unpack_into(theta)
    assert np.allclose(head2.W_domain.detach().numpy(), Wd)
    assert np.allclose(head2.W_diff.detach().numpy(), Wf)


def test_difficulty_index_mapping():
    th = _mod()
    head = th.TriageHead5Domain(d_h=2)
    assert head.difficulty_to_index(1) == 0
    assert head.difficulty_to_index(5) == 4
    assert head.index_to_difficulty(0) == 1
    with pytest.raises(ValueError):
        head.difficulty_to_index(0)
    with pytest.raises(ValueError):
        head.index_to_difficulty(5)


def test_batched_forward():
    torch = _torch()
    th = _mod()
    head = th.TriageHead5Domain(d_h=4)
    h = torch.randn(7, 4)
    z_d, z_f = head.forward(h)
    assert z_d.shape == (7, 5)
    assert z_f.shape == (7, 5)


def test_select_rejects_batch():
    torch = _torch()
    th = _mod()
    head = th.TriageHead5Domain(d_h=4)
    with pytest.raises(ValueError, match="single hidden state"):
        head.select(torch.randn(2, 4), sample=False)
