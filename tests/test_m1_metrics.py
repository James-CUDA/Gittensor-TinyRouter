"""Torch-free Milestone-1 metrics + pack roundtrip (numpy only for pack shapes)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trinity.m1.metrics import macro_f1, score_triage
from trinity.m1.pack import Milestone1Pack, load_milestone1_pack, save_milestone1_pack


def test_perfect_scores():
    y_d = ["math", "code", "math"]
    y_f = [1, 2, 5]
    m = score_triage(y_d, y_f, y_d, y_f)
    assert m.n == 3
    assert m.domain_accuracy == 1.0
    assert m.difficulty_exact == 1.0
    assert m.difficulty_within_1 == 1.0
    assert m.joint_accuracy == 1.0
    assert m.composite == 1.0


def test_partial_and_within1():
    y_d = ["math", "code"]
    y_f = [3, 3]
    p_d = ["math", "math"]
    p_f = [4, 1]  # within1 on first, not second
    m = score_triage(y_d, y_f, p_d, p_f)
    assert m.domain_accuracy == 0.5
    assert m.difficulty_exact == 0.0
    assert m.difficulty_within_1 == 0.5
    assert m.joint_accuracy == 0.0
    assert abs(m.composite - (0.7 * 0.5 + 0.3 * 0.0)) < 1e-9


def test_macro_f1_balanced():
    y = ["a", "b", "a", "b"]
    p = ["a", "b", "a", "a"]
    # a: tp=2 fp=1 fn=0 → p=2/3 r=1 f1=0.8; b: tp=1 fp=0 fn=1 → p=1 r=0.5 f1=2/3
    f1 = macro_f1(y, p)
    assert 0.7 < f1 < 0.8


def test_pack_roundtrip(tmp_path: Path):
    pytest.importorskip("torch")
    Wd = np.zeros((5, 8), dtype=np.float32)
    Wf = np.zeros((5, 8), dtype=np.float32)
    Wd[0, 0] = 1.0
    pack = Milestone1Pack(
        config="5-domain",
        pool="penultimate",
        d_h=8,
        W_domain=Wd,
        W_diff=Wf,
    )
    save_milestone1_pack(tmp_path / "m1", pack)
    loaded = load_milestone1_pack(tmp_path / "m1")
    assert loaded.config == "5-domain"
    assert np.allclose(loaded.W_domain, Wd)
    cfg = json.loads((tmp_path / "m1" / "config.json").read_text())
    assert cfg["schema"] == "tinyrouter-m1-v1"
    head = loaded.build_head()
    assert head.n_domains == 5


def test_attentive_pack_requires_query(tmp_path: Path):
    pack = Milestone1Pack(
        config="5-domain",
        pool="attentive",
        d_h=4,
        W_domain=np.zeros((5, 4), np.float32),
        W_diff=np.zeros((5, 4), np.float32),
        attention_query=None,
    )
    with pytest.raises(ValueError, match="attention_query"):
        pack.validate()
