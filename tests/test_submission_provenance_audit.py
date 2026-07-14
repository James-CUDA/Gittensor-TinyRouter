"""Offline tests for fitness-history, ledger-volume, and head-diversity gates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trinity.llm import cost_ledger as CL
from trinity.submission.constants import EXPECTED_HEAD_SHAPE, EXPECTED_TOTAL_PARAMS, N_HEAD_MODELS
from trinity.submission.gates import (
    OFFLINE_ADVISORIES,
    OFFLINE_GATES,
    PreflightContext,
    run_offline_advisories,
    run_offline_gates,
)
from trinity.submission.pack import SubmissionPack
from trinity.submission.provenance_audit import (
    FitnessHistorySequenceAudit,
    HeadRoutingDiversityAudit,
    LedgerTrainingVolumeAudit,
    validate_fitness_history_sequence,
    validate_head_routing_diversity,
    validate_ledger_call_volume,
)

_HEAD_SHAPE = EXPECTED_HEAD_SHAPE
_N_SVF = 7168
_POOL = ["qwen3.5-35b-a3b", "minimax-m3", "deepseek-v4-flash"]


def _honest_history(n: int = 6) -> list[dict]:
    rows = [
        (0.30, 0.50, 0.50), (0.42, 0.58, 0.58), (0.39, 0.61, 0.61),
        (0.55, 0.69, 0.69), (0.58, 0.66, 0.69), (0.60, 0.72, 0.72),
    ]
    return [
        {"generation": i, "mean_fitness": m, "max_fitness": mx, "best_fitness": b}
        for i, (m, mx, b) in enumerate(rows[:n])
    ]


def _honest_receipt(**overrides) -> dict:
    history = overrides.pop("fitness_history", _honest_history())
    base = {
        "benchmark": "math500",
        "pool_models": _POOL,
        "n_total": EXPECTED_TOTAL_PARAMS,
        "popsize": 33,
        "m_cma": 16,
        "total_cost_usd": 21.5,
        "generations": len(history),
        "best_fitness": 0.72,
        "fitness_history": history,
    }
    base.update(overrides)
    return base


def _rand_head(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.05, _HEAD_SHAPE).astype(np.float32)


def _write_plausible_ledger(path: Path, *, entries: int = 12) -> None:
    models = _POOL
    for i in range(entries):
        CL.append_ledger_entry(path, models[i % len(models)], 50_000, 10_000)


# --------------------------------------------------------------------------- #
# Fitness history sequence (gate 10)
# --------------------------------------------------------------------------- #
def test_fitness_history_sequence_accepts_honest_receipt():
    assert validate_fitness_history_sequence(_honest_receipt()) is None


@pytest.mark.parametrize(
    "mutator,reason_prefix",
    [
        (lambda r: r.update({"fitness_history": [{"generation": 0}, {"generation": 0}]}),
         "receipt_fitness_history_duplicate_generations"),
        (lambda r: r.update({"fitness_history": [{"generation": 0}, {"generation": 2}]}),
         "receipt_fitness_history_nonconsecutive"),
        (
            lambda r: r.update({
                "fitness_history": [
                    {"generation": 0, "mean_fitness": 0.9, "max_fitness": 0.5, "best_fitness": 0.5},
                ],
            }),
            "receipt_fitness_history_mean_gt_max",
        ),
        (
            lambda r: r.update({
                "fitness_history": [
                    {"generation": 0, "mean_fitness": 0.5, "max_fitness": 0.9, "best_fitness": 0.9},
                    {"generation": 1, "mean_fitness": 0.3, "max_fitness": 0.4, "best_fitness": 0.4},
                ],
            }),
            "receipt_fitness_history_best_regressed",
        ),
        (lambda r: r.update({"best_fitness": 0.10}), "receipt_best_fitness_history_peak_mismatch"),
    ],
)
def test_fitness_history_sequence_rejects_structural_fraud(mutator, reason_prefix: str):
    receipt = _honest_receipt()
    mutator(receipt)
    err = validate_fitness_history_sequence(receipt)
    assert err is not None
    assert err.startswith(reason_prefix)


def test_fitness_history_sequence_accepts_one_based_generations():
    receipt = _honest_receipt()
    for i, row in enumerate(receipt["fitness_history"], start=1):
        row["generation"] = i
    receipt["generations"] = len(receipt["fitness_history"])
    assert FitnessHistorySequenceAudit().validate(receipt) is None


# --------------------------------------------------------------------------- #
# Ledger call volume (advisory)
# --------------------------------------------------------------------------- #
def test_ledger_call_volume_skipped_without_ledger():
    assert validate_ledger_call_volume(_honest_receipt(), None) is None


def test_ledger_call_volume_passes_plausible_ledger(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    _write_plausible_ledger(ledger, entries=15)
    assert validate_ledger_call_volume(_honest_receipt(), str(ledger)) is None


def test_ledger_call_volume_rejects_sparse_ledger(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    CL.append_ledger_entry(ledger, "qwen3.5-35b-a3b", 100, 50)
    err = validate_ledger_call_volume(_honest_receipt(), str(ledger))
    assert err is not None
    assert "ledger_call_count_too_low" in err or "ledger_volume_too_low" in err


def test_ledger_call_volume_rejects_single_model_coverage(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    for _ in range(12):
        CL.append_ledger_entry(ledger, "qwen3.5-35b-a3b", 50_000, 10_000)
    err = validate_ledger_call_volume(_honest_receipt(), str(ledger))
    assert err is not None
    assert err.startswith("ledger_pool_coverage_too_low")


def test_ledger_call_volume_rejects_tampered_chain(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    _write_plausible_ledger(ledger)
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(text.replace('"p":50000', '"p":5'), encoding="utf-8")
    err = validate_ledger_call_volume(_honest_receipt(), str(ledger))
    assert err is not None
    assert "ledger_volume_unverifiable" in err


# --------------------------------------------------------------------------- #
# Head routing diversity (gate 10)
# --------------------------------------------------------------------------- #
def test_head_routing_diversity_accepts_random_head():
    assert validate_head_routing_diversity(_rand_head(3)) is None


def test_head_routing_diversity_rejects_identical_agent_rows():
    head = _rand_head(9)
    head[1] = head[0] + 1e-7
    head[2] = head[0] + 2e-7
    err = validate_head_routing_diversity(head)
    assert err is not None
    assert err.startswith("head_routing_diversity_agent")


def test_head_routing_diversity_rejects_near_zero_agent_block():
    head = _rand_head(4)
    head[:N_HEAD_MODELS] = 0.0
    err = HeadRoutingDiversityAudit().validate(head)
    assert err is not None
    assert err.startswith("head_routing_diversity_single_active_agent_row") or err.startswith(
        "head_routing_diversity_agent_rows_near_zero"
    )


# --------------------------------------------------------------------------- #
# Gate / advisory wiring
# --------------------------------------------------------------------------- #
def test_offline_gates_include_fitness_history_sequence_as_gate_10():
    names = [gate.name for gate in OFFLINE_GATES]
    assert names[-1] == "fitness_history_sequence"
    assert names[-3:] == [
        "artifact_manifest",
        "receipt_cmaes",
        "fitness_history_sequence",
    ]
    assert len(names) == 10


def test_offline_advisories_include_svf_ledger_and_head_checks():
    names = [adv.name for adv in OFFLINE_ADVISORIES]
    assert names == [
        "svf_training_signal",
        "ledger_call_volume",
        "head_routing_diversity",
    ]


def test_run_offline_advisories_warn_without_blocking(tmp_path: Path):
    pack_dir = tmp_path / "alice" / "1"
    pack_dir.mkdir(parents=True)
    head = _rand_head(1)
    head[1] = head[0] + 1e-7
    head[2] = head[0] + 2e-7
    np.save(pack_dir / "head_weights.npy", head)
    np.save(pack_dir / "svf_scales.npy", np.ones(_N_SVF, dtype=np.float32))
    receipt = _honest_receipt()
    (pack_dir / "receipt.json").write_text(__import__("json").dumps(receipt), encoding="utf-8")

    pack = SubmissionPack(
        path=pack_dir,
        miner="alice",
        generation=1,
        head_weights=head,
        svf_scales=np.ones(_N_SVF, dtype=np.float32),
        receipt=receipt,
    )
    ctx = PreflightContext(
        benchmark="math500",
        leaderboard={"benchmarks": {}},
        submissions_root=tmp_path,
    )
    advisories = run_offline_advisories(pack, ctx)
    triggered = [a for a in advisories if a.triggered]
    assert len(triggered) >= 1
    assert any(a.advisory == "head_routing_diversity" for a in triggered)


def test_run_offline_gates_stops_on_fitness_history_failure(tmp_path: Path):
    pack_dir = tmp_path / "alice" / "1"
    pack_dir.mkdir(parents=True)
    np.save(pack_dir / "head_weights.npy", _rand_head(1))
    np.save(pack_dir / "svf_scales.npy", np.ones(_N_SVF, dtype=np.float32))
    receipt = _honest_receipt()
    receipt["fitness_history"][1]["generation"] = 5
    (pack_dir / "receipt.json").write_text(__import__("json").dumps(receipt), encoding="utf-8")

    pack = SubmissionPack(
        path=pack_dir,
        miner="alice",
        generation=1,
        head_weights=np.load(pack_dir / "head_weights.npy"),
        svf_scales=np.load(pack_dir / "svf_scales.npy"),
        receipt=receipt,
    )
    ctx = PreflightContext(
        benchmark="math500",
        leaderboard={"benchmarks": {}},
        submissions_root=tmp_path,
    )
    results = run_offline_gates(
        pack,
        ctx,
        gates=tuple(g for g in OFFLINE_GATES if g.name == "fitness_history_sequence"),
    )
    failed = [r for r in results if r.failed]
    assert len(failed) == 1
    assert failed[0].gate == "fitness_history_sequence"


def test_ledger_volume_audit_min_tokens_formula():
    audit = LedgerTrainingVolumeAudit(min_tokens_per_candidate=150)
    assert audit.min_tokens_per_candidate == 150
    assert 6 * 33 * 150 == 29_700
