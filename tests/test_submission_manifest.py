"""Offline tests for submission artifact manifest (gate 8)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trinity.submission.constants import EXPECTED_HEAD_SHAPE
from trinity.submission.gates import OFFLINE_GATES
from trinity.submission.manifest import (
    MANIFEST_FILENAME,
    ManifestBuilder,
    build_submission_manifest,
    load_manifest,
    manifest_content_hash,
    sha256_file,
    validate_artifact_manifest,
)

_HEAD_SHAPE = EXPECTED_HEAD_SHAPE
_N_SVF = 7168


def _rand_head(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.05, _HEAD_SHAPE).astype(np.float32)


def _near_identity_svf(seed: int, std: float = 0.02) -> np.ndarray:
    return (1.0 + np.random.default_rng(seed).normal(0, std, _N_SVF)).astype(np.float32)


def _valid_receipt(*, benchmark: str = "math500") -> dict:
    return {
        "benchmark": benchmark,
        "pool_models": ["qwen3.5-35b-a3b", "minimax-m3", "deepseek-v4-flash"],
        "n_total": 13312,
        "popsize": 33,
        "m_cma": 16,
        "total_cost_usd": 21.5,
        "generations": 6,
        "best_fitness": 0.72,
        "seed": 7,
        "fitness_history": [
            {"generation": i, "mean_fitness": 0.3 + i * 0.05, "max_fitness": 0.5 + i * 0.04}
            for i in range(6)
        ],
    }


def _write_pack(
    root: Path,
    miner: str,
    gen: int,
    *,
    benchmark: str = "math500",
    with_manifest: bool = True,
) -> Path:
    pack_dir = root / miner / str(gen)
    pack_dir.mkdir(parents=True, exist_ok=True)
    np.save(pack_dir / "head_weights.npy", _rand_head(gen))
    np.save(pack_dir / "svf_scales.npy", _near_identity_svf(gen))
    (pack_dir / "receipt.json").write_text(json.dumps(_valid_receipt(benchmark=benchmark)), encoding="utf-8")
    if with_manifest:
        manifest = build_submission_manifest(pack_dir, miner=miner, generation=gen, benchmark=benchmark)
        (pack_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return pack_dir


def test_sha256_file_is_deterministic(tmp_path: Path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"trinity-submission")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_manifest_content_hash_is_order_invariant():
    from trinity.submission.manifest import ArtifactRecord

    a = (
        ArtifactRecord("head_weights.npy", "aa" * 32, 100),
        ArtifactRecord("receipt.json", "bb" * 32, 200),
        ArtifactRecord("svf_scales.npy", "cc" * 32, 300),
    )
    b = (a[2], a[0], a[1])
    assert manifest_content_hash(a) == manifest_content_hash(b)


def test_build_submission_manifest_pins_all_artifacts(tmp_path: Path):
    pack_dir = _write_pack(tmp_path, "alice", 1, with_manifest=False)
    manifest = build_submission_manifest(pack_dir, miner="alice", generation=1, benchmark="math500")
    assert manifest["miner"] == "alice"
    assert manifest["generation"] == 1
    assert manifest["benchmark"] == "math500"
    assert len(manifest["artifacts"]) == 3
    assert validate_artifact_manifest(pack_dir, manifest, miner="alice", generation=1, benchmark="math500") is None


def test_validate_artifact_manifest_rejects_missing_manifest():
    err = validate_artifact_manifest(Path("/nope"), None, miner="a", generation=1, benchmark="math500")
    assert err == "manifest_missing"


def test_validate_artifact_manifest_rejects_tampered_receipt(tmp_path: Path):
    pack_dir = _write_pack(tmp_path, "bob", 2)
    receipt_path = pack_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["total_cost_usd"] = 999.0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest = load_manifest(pack_dir)
    err = validate_artifact_manifest(pack_dir, manifest, miner="bob", generation=2, benchmark="math500")
    assert err is not None
    assert "manifest_" in err and "receipt.json" in err


def test_validate_artifact_manifest_rejects_miner_mismatch(tmp_path: Path):
    pack_dir = _write_pack(tmp_path, "carol", 3)
    manifest = load_manifest(pack_dir)
    err = validate_artifact_manifest(pack_dir, manifest, miner="other", generation=3, benchmark="math500")
    assert err == "manifest_miner_mismatch: manifest 'carol' != pack 'other'"


def test_validate_artifact_manifest_rejects_benchmark_mismatch(tmp_path: Path):
    pack_dir = _write_pack(tmp_path, "dave", 4, benchmark="math500")
    manifest = load_manifest(pack_dir)
    err = validate_artifact_manifest(pack_dir, manifest, miner="dave", generation=4, benchmark="mmlu")
    assert err is not None
    assert "manifest_benchmark_mismatch" in err


def test_validate_artifact_manifest_rejects_content_hash_tamper(tmp_path: Path):
    pack_dir = _write_pack(tmp_path, "erin", 5)
    manifest = load_manifest(pack_dir)
    assert manifest is not None
    manifest["content_hash"] = "0" * 64
    err = validate_artifact_manifest(pack_dir, manifest, miner="erin", generation=5, benchmark="math500")
    assert err == "manifest_content_hash_mismatch"


def test_manifest_builder_raises_when_artifact_missing(tmp_path: Path):
    pack_dir = tmp_path / "incomplete"
    pack_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="head_weights.npy"):
        ManifestBuilder(miner="x", generation=1, benchmark="math500").build(pack_dir)


def test_offline_gates_include_manifest_receipt_cmaes_and_svf():
    names = [gate.name for gate in OFFLINE_GATES]
    assert names[-3:] == ["artifact_manifest", "receipt_cmaes", "svf_training_signal"]


def test_load_manifest_returns_none_on_bad_json(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_manifest(tmp_path) is None
