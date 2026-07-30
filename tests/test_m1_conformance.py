"""MILESTONE1.md ↔ trinity.m1 conformance + the offline miner workflow.

Two things nothing else guards:

* the miner-facing rules in ``docs/MILESTONE1.md`` (param budget, rate, win
  margin, composite weights, pack layout) match ``trinity.m1``'s code;
* the offline ``pack → preflight`` path the document prescribes actually round-
  trips, at the documented boundaries.

``CODE_PACK_FILES`` in the conformance module is a hand-maintained list; the
``test_code_pack_files_matches_what_the_packer_writes`` test saves a real pack
and asserts the list is exactly what lands on disk, so it cannot drift.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from trinity.m1.conformance import (
    CODE_PACK_FILES,
    check,
    default_doc_path,
    parse_pack_layout,
    parse_rules,
    parse_scoring_weights,
    render,
)
from trinity.m1.constants import (
    DOMAIN_WEIGHT,
    MAX_HEAD_PARAMS,
    RATE_LIMIT_MAX_SUBMISSIONS,
    WIN_MARGIN,
)
from trinity.m1.gates import run_m1_gates
from trinity.m1.leaderboard import decide_m1_winner
from trinity.m1.pack import Milestone1Pack, load_milestone1_pack, save_milestone1_pack
from trinity.m1.domains import DOMAINS_5

_REPO = Path(__file__).resolve().parents[1]
_DOC = _REPO / "docs" / "MILESTONE1.md"
_SCRIPT = _REPO / "scripts" / "milestone1_conformance_report.py"
_SRC = str(_REPO / "src")

D_H = 1024


def _pack(config="5-domain", d_h=D_H, scale=0.01):
    rng = np.random.default_rng(0)
    n_dom = len(DOMAINS_5) if config == "5-domain" else 20
    return Milestone1Pack(
        config=config,
        pool="penultimate",
        d_h=d_h,
        W_domain=(rng.normal(size=(n_dom, d_h)) * scale).astype(np.float32),
        W_diff=(rng.normal(size=(5, d_h)) * scale).astype(np.float32),
    )


# --------------------------------------------------------------------------
# the document parses, and matches the code
# --------------------------------------------------------------------------


def test_conformance_holds_on_the_shipped_doc():
    report = check()
    assert report.ok, render(report)


def test_default_doc_path_points_at_the_repo_doc():
    assert default_doc_path() == _DOC
    assert _DOC.exists()


def test_parse_rules_reads_the_three_limits():
    rules = parse_rules(_DOC.read_text())
    assert rules["max_params"] == MAX_HEAD_PARAMS == 1_000_000
    assert rules["rate_per_day"] == RATE_LIMIT_MAX_SUBMISSIONS == 1
    assert rules["win_margin"] == WIN_MARGIN == 0.02


def test_parse_scoring_weights_matches_the_constant():
    dom, diff = parse_scoring_weights(_DOC.read_text())
    assert dom == DOMAIN_WEIGHT
    assert dom + diff == pytest.approx(1.0)


def test_parse_pack_layout_lists_the_core_files():
    files = parse_pack_layout(_DOC.read_text())
    assert "config.json" in files
    assert "W_domain.npy" in files
    assert "W_diff.npy" in files


# --------------------------------------------------------------------------
# drift is detected
# --------------------------------------------------------------------------


def _doc_with(**subs):
    md = _DOC.read_text()
    for old, new in subs.items():
        md = md.replace(old, new)
    return md


def test_a_changed_param_budget_in_the_doc_is_caught():
    report = check(md=_doc_with(**{"< 1,000,000": "< 2,000,000"}))
    assert report.ok is False
    assert any(f.check == "param_budget" for f in report.failures)


def test_a_changed_win_margin_in_the_doc_is_caught():
    report = check(md=_doc_with(**{"≥ 0.02": "≥ 0.05"}))
    assert report.ok is False
    assert any(f.check == "win_margin" for f in report.failures)


def test_a_changed_composite_weight_in_the_doc_is_caught():
    report = check(md=_doc_with(**{"0.7 × domain_accuracy + 0.3": "0.6 × domain_accuracy + 0.4"}))
    assert report.ok is False
    assert any(f.check == "composite_weights" for f in report.failures)


def test_a_phantom_pack_file_in_the_doc_is_caught():
    report = check(md=_doc_with(**{"W_diff.npy": "W_phantom.npy"}))
    assert report.ok is False
    assert any(f.check == "pack_layout_no_phantom_files" for f in report.failures)


def test_a_missing_section_is_a_parse_error():
    with pytest.raises(ValueError, match="Rules"):
        parse_rules("# doc with no rules section\n")


# --------------------------------------------------------------------------
# CODE_PACK_FILES cannot drift from the packer
# --------------------------------------------------------------------------


def test_code_pack_files_matches_what_the_packer_writes(tmp_path):
    """Save required + all-optional packs; the union of files must equal the list."""
    # penultimate pack -> the required files only
    save_milestone1_pack(tmp_path / "req", _pack())
    req = {p.name for p in (tmp_path / "req").iterdir()}
    assert req == set(f for f, k in CODE_PACK_FILES.items() if k == "required")

    # attentive pack with key projection -> every optional file too. Use a small
    # d_h so W_k (d_h × d_h) stays well under the 1M param budget.
    small = 100
    rng = np.random.default_rng(1)
    full = Milestone1Pack(
        config="5-domain", pool="attentive", d_h=small,
        W_domain=(rng.normal(size=(5, small)) * 0.01).astype(np.float32),
        W_diff=(rng.normal(size=(5, small)) * 0.01).astype(np.float32),
        attention_query=(rng.normal(size=(small,)) * 0.01).astype(np.float32),
        W_k=(rng.normal(size=(small, small)) * 0.001).astype(np.float32),
    )
    save_milestone1_pack(tmp_path / "full", full)
    allf = {p.name for p in (tmp_path / "full").iterdir()}
    assert allf == set(CODE_PACK_FILES)


def test_the_wk_omission_is_reported_as_a_note():
    """W_k.npy is writable but absent from the doc's layout block."""
    notes = check().notes
    assert any("W_k.npy" in n for n in notes)


# --------------------------------------------------------------------------
# the offline pack -> preflight workflow (MILESTONE1.md steps 2-3)
# --------------------------------------------------------------------------


def test_pack_round_trips_through_disk(tmp_path):
    save_milestone1_pack(tmp_path / "m1", _pack())
    reloaded = load_milestone1_pack(tmp_path / "m1")
    assert reloaded.config == "5-domain"
    assert reloaded.n_params == 10_240  # doc: "≈ 10,240 params"


def test_documented_default_pack_size():
    """The doc states the default 5-domain penultimate pack ≈ 10,240 params."""
    assert _pack().n_params == 5 * D_H + 5 * D_H == 10_240


def test_default_pack_passes_all_offline_gates(tmp_path):
    save_milestone1_pack(tmp_path / "m1", _pack())
    pack = load_milestone1_pack(tmp_path / "m1")
    results = run_m1_gates(pack, miner="alice", repo_root=tmp_path, skip_rate_limit=True)
    assert all(r.ok for r in results), [r.reason for r in results if not r.ok]


def test_param_budget_boundary_is_exactly_the_documented_limit():
    """Doc says '< 1,000,000': 999,999 ok, 1,000,000 rejected."""
    # d_h chosen so 2 rows (domain rows == 5, diff == 5) won't hit 1M easily;
    # instead assert the pack.validate boundary directly via a fabricated count.
    from trinity.m1.pack import count_pack_params

    # A 5-domain penultimate pack has (n_dom + 5) * d_h params. Pick d_h so the
    # count straddles the limit: (5+5)*d_h. At d_h=100_000 -> 1,000,000 exactly.
    just_over = _pack(d_h=100_000)
    assert count_pack_params(just_over) == 1_000_000
    with pytest.raises(ValueError, match="limit"):
        just_over.validate()

    just_under = _pack(d_h=99_999)
    assert count_pack_params(just_under) == 999_990  # < 1,000,000
    just_under.validate()  # must not raise


def test_win_margin_decision_matches_the_documented_rule():
    # No king -> first challenger wins outright.
    wins, _ = decide_m1_winner(king_composite=None, challenger_composite=0.10)
    assert wins is True
    # Exactly at king + margin -> wins (>= is documented).
    wins, _ = decide_m1_winner(king_composite=0.80, challenger_composite=0.80 + WIN_MARGIN)
    assert wins is True
    # A hair under the margin -> loses.
    wins, _ = decide_m1_winner(king_composite=0.80, challenger_composite=0.80 + WIN_MARGIN - 1e-6)
    assert wins is False


# --------------------------------------------------------------------------
# report + CLI
# --------------------------------------------------------------------------


def test_report_serializes():
    payload = check().to_dict()
    json.dumps(payload)
    assert payload["ok"] is True
    assert len(payload["findings"]) >= 6


def test_render_reads_cleanly():
    assert "doc and code agree" in render(check())


def _run(*args):
    env = {**os.environ, "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, env=env
    )


def test_cli_passes_on_the_real_doc():
    r = _run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "conformance" in r.stdout


def test_cli_json():
    r = _run("--json")
    assert r.returncode == 0
    assert json.loads(r.stdout)["ok"] is True


def test_cli_flags_a_mismatched_doc(tmp_path):
    doc = tmp_path / "MILESTONE1.md"
    doc.write_text(_doc_with(**{"≥ 0.02": "≥ 0.09"}))
    r = _run("--doc", str(doc))
    assert r.returncode == 1
    assert "MISMATCH" in r.stdout


def test_cli_missing_doc_is_graceful(tmp_path):
    r = _run("--doc", str(tmp_path / "nope.md"))
    assert r.returncode == 2
    assert "no such file" in r.stderr


def test_cli_unparseable_doc_is_graceful(tmp_path):
    doc = tmp_path / "MILESTONE1.md"
    doc.write_text("# nothing here\n")
    r = _run("--doc", str(doc))
    assert r.returncode == 2
    assert "could not parse" in r.stderr


# --------------------------------------------------------------------------
# import cost
# --------------------------------------------------------------------------


def test_module_imports_without_torch():
    env = {**os.environ, "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    code = ("import sys; import trinity.m1.conformance; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True, env=env)
    assert out.stdout.strip() == "False", out.stdout
