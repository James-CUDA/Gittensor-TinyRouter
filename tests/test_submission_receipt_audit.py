"""Offline tests for CMA-ES receipt audit and SVF training-signal gates (9–10)."""
from __future__ import annotations

import numpy as np
import pytest

from trinity.optim.sep_cmaes import default_popsize
from trinity.submission.constants import EXPECTED_TOTAL_PARAMS
from trinity.submission.receipt_audit import (
    ReceiptCmaesAudit,
    SvfTrainingSignalAudit,
    validate_receipt_cmaes,
    validate_svf_training_signal,
)

_N_SVF = 7168


def _base_receipt(**overrides) -> dict:
    receipt = {
        "benchmark": "math500",
        "pool_models": ["qwen3.5-35b-a3b", "minimax-m3", "deepseek-v4-flash"],
        "n_total": EXPECTED_TOTAL_PARAMS,
        "popsize": default_popsize(EXPECTED_TOTAL_PARAMS),
        "m_cma": default_popsize(EXPECTED_TOTAL_PARAMS) // 2,
        "generations": 60,
        "best_fitness": 0.72,
        "total_cost_usd": 21.5,
        "fitness_history": [
            {
                "generation": i,
                "mean_fitness": (0.30 + i * 0.005) + (0.02 if i % 7 == 0 else -0.01 if i % 5 == 0 else 0.0),
                "max_fitness": 0.50 + i * 0.004,
                "best_fitness": min(0.72, 0.50 + i * 0.004),
            }
            for i in range(60)
        ],
    }
    receipt.update(overrides)
    return receipt


def _trained_svf(seed: int = 1, std: float = 0.05) -> np.ndarray:
    return (1.0 + np.random.default_rng(seed).normal(0, std, _N_SVF)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Receipt CMA-ES audit (gate 9)
# --------------------------------------------------------------------------- #
def test_validate_receipt_cmaes_accepts_honest_receipt():
    assert validate_receipt_cmaes(_base_receipt()) is None


@pytest.mark.parametrize(
    "field,value,prefix",
    [
        ("popsize", 99, "receipt_popsize_mismatch"),
        ("m_cma", 99, "receipt_m_cma_mismatch"),
        ("n_total", 999, "receipt_n_total_mismatch"),
        ("generations", 1, "receipt_generations_too_low"),
    ],
)
def test_validate_receipt_cmaes_rejects_bad_metadata(field, value, prefix):
    receipt = _base_receipt(**{field: value})
    err = validate_receipt_cmaes(receipt)
    assert err is not None
    assert err.startswith(prefix)


def test_receipt_cmaes_audit_expected_defaults_match_spec():
    audit = ReceiptCmaesAudit()
    assert audit.expected_popsize() == 33
    assert audit.expected_m_cma(33) == 16


def test_validate_receipt_cmaes_rejects_generations_history_drift():
    receipt = _base_receipt(generations=60, fitness_history=[{"generation": 0}] * 10)
    err = validate_receipt_cmaes(receipt)
    assert err is not None
    assert "receipt_generations_history_mismatch" in err


# --------------------------------------------------------------------------- #
# SVF training signal (gate 10)
# --------------------------------------------------------------------------- #
def test_validate_svf_training_signal_accepts_adapted_svf():
    svf = _trained_svf()
    receipt = _base_receipt(best_fitness=0.72)
    assert validate_svf_training_signal(svf, receipt) is None


def test_validate_svf_training_signal_ignores_low_fitness():
    svf = np.ones(_N_SVF, dtype=np.float32)
    receipt = _base_receipt(best_fitness=0.10)
    assert validate_svf_training_signal(svf, receipt) is None


def test_validate_svf_training_signal_rejects_identity_svf_with_high_fitness():
    svf = np.ones(_N_SVF, dtype=np.float32)
    receipt = _base_receipt(best_fitness=0.72)
    err = validate_svf_training_signal(svf, receipt)
    assert err is not None
    assert err.startswith("svf_untrained_identity")


def test_validate_svf_training_signal_rejects_non_positive_svf():
    svf = _trained_svf()
    svf[0] = -0.01
    receipt = _base_receipt(best_fitness=0.72)
    err = validate_svf_training_signal(svf, receipt)
    assert err == "svf_non_positive"


def test_svf_training_signal_audit_rejects_near_identity_fraction():
    svf = np.ones(_N_SVF, dtype=np.float32)
    svf[:5] = 1.01  # only 5 / 7168 differ — below 1% threshold
    receipt = _base_receipt(best_fitness=0.72)
    err = SvfTrainingSignalAudit().validate(svf, receipt)
    assert err is not None
    assert "svf_insufficient_adaptation" in err


def test_preflight_receipt_audit_integration_with_manifest(tmp_path):
    """End-to-end: pack with manifest + adapted SVF passes receipt gates."""
    from trinity.submission.manifest import build_submission_manifest, MANIFEST_FILENAME
    from trinity.submission.gates import run_offline_gates, PreflightContext
    from trinity.submission.pack import load_submission_pack
    import json
    from pathlib import Path

    subs = tmp_path / "submissions"
    pack_dir = subs / "grace" / "1"
    pack_dir.mkdir(parents=True)
    np.save(pack_dir / "head_weights.npy", np.random.default_rng(1).normal(0, 0.05, (6, 1024)).astype(np.float32))
    np.save(pack_dir / "svf_scales.npy", _trained_svf())
    (pack_dir / "receipt.json").write_text(json.dumps(_base_receipt()), encoding="utf-8")
    manifest = build_submission_manifest(pack_dir, miner="grace", generation=1, benchmark="math500")
    (pack_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    pack = load_submission_pack(pack_dir, submissions_root=subs)
    assert pack is not None
    ctx = PreflightContext(
        benchmark="math500",
        leaderboard={"benchmarks": {"math500": {"attempts": []}}},
        submissions_root=subs,
    )
    results = run_offline_gates(pack, ctx)
    assert all(r.ok for r in results)
    assert [r.gate for r in results][-3:] == ["artifact_manifest", "receipt_cmaes", "svf_training_signal"]
