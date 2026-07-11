"""SHA-256 artifact manifest for routing-head submission packs.

Gate 8 binds ``head_weights.npy``, ``svf_scales.npy``, and ``receipt.json`` so
a miner cannot edit the receipt without invalidating the manifest, or swap weight
files after packing. Pure offline — no GPU, no network.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "MANIFEST_VERSION",
    "MANIFEST_FILENAME",
    "HASHED_ARTIFACTS",
    "ArtifactRecord",
    "SubmissionManifest",
    "ManifestBuilder",
    "sha256_file",
    "build_submission_manifest",
    "load_manifest",
    "validate_artifact_manifest",
]

MANIFEST_VERSION: int = 1
MANIFEST_FILENAME: str = "manifest.json"

# Files whose bytes are pinned by the manifest hash.
HASHED_ARTIFACTS: tuple[str, ...] = (
    "head_weights.npy",
    "svf_scales.npy",
    "receipt.json",
)


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactRecord:
    """One pinned artifact on disk."""

    name: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class SubmissionManifest:
    """Public manifest written beside a submission pack."""

    version: int
    miner: str
    generation: int
    benchmark: str
    artifacts: tuple[ArtifactRecord, ...]
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "miner": self.miner,
            "generation": self.generation,
            "benchmark": self.benchmark,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "content_hash": self.content_hash,
        }


def _canonical_artifacts_payload(artifacts: tuple[ArtifactRecord, ...]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in sorted(artifacts, key=lambda a: a.name)]


def manifest_content_hash(artifacts: tuple[ArtifactRecord, ...]) -> str:
    """Hash the sorted artifact table — independent of miner metadata."""
    payload = json.dumps(_canonical_artifacts_payload(artifacts), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ManifestBuilder:
    """Build a manifest from files already written to a submission directory."""

    def __init__(self, *, miner: str, generation: int, benchmark: str) -> None:
        self.miner = miner
        self.generation = generation
        self.benchmark = benchmark

    def build(self, pack_dir: Path) -> SubmissionManifest:
        records: list[ArtifactRecord] = []
        for name in HASHED_ARTIFACTS:
            path = pack_dir / name
            if not path.exists():
                raise FileNotFoundError(f"missing artifact for manifest: {name}")
            records.append(
                ArtifactRecord(name=name, sha256=sha256_file(path), size_bytes=path.stat().st_size),
            )
        artifact_tuple = tuple(records)
        return SubmissionManifest(
            version=MANIFEST_VERSION,
            miner=self.miner,
            generation=self.generation,
            benchmark=self.benchmark,
            artifacts=artifact_tuple,
            content_hash=manifest_content_hash(artifact_tuple),
        )


def build_submission_manifest(
    pack_dir: Path,
    *,
    miner: str,
    generation: int,
    benchmark: str,
) -> dict[str, Any]:
    """Build and return a JSON-serialisable manifest dict for ``pack_dir``."""
    manifest = ManifestBuilder(miner=miner, generation=generation, benchmark=benchmark).build(pack_dir)
    return manifest.to_dict()


def load_manifest(pack_dir: Path) -> dict[str, Any] | None:
    """Load ``manifest.json`` when present; return ``None`` on absence or parse failure."""
    path = pack_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _parse_artifact_records(raw: Mapping[str, Any]) -> tuple[ArtifactRecord, ...] | None:
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return None
    records: list[ArtifactRecord] = []
    for entry in artifacts:
        if not isinstance(entry, dict):
            return None
        name = entry.get("name")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if not isinstance(name, str) or not isinstance(digest, str) or not isinstance(size, int):
            return None
        records.append(ArtifactRecord(name=name, sha256=digest.lower(), size_bytes=size))
    return tuple(records)


def validate_artifact_manifest(
    pack_dir: Path,
    manifest: Mapping[str, Any] | None,
    *,
    miner: str,
    generation: int,
    benchmark: str,
) -> str | None:
    """Gate 8: verify ``manifest.json`` matches on-disk artifacts and context."""
    if manifest is None:
        return "manifest_missing"

    if manifest.get("version") != MANIFEST_VERSION:
        return f"manifest_version_mismatch: got {manifest.get('version')!r}, expected {MANIFEST_VERSION}"

    if manifest.get("miner") != miner:
        return f"manifest_miner_mismatch: manifest {manifest.get('miner')!r} != pack {miner!r}"

    try:
        manifest_gen = int(manifest.get("generation", -1))
    except (TypeError, ValueError):
        return "manifest_generation_invalid"
    if manifest_gen != generation:
        return f"manifest_generation_mismatch: manifest {manifest_gen} != pack {generation}"

    if manifest.get("benchmark") != benchmark:
        return (
            f"manifest_benchmark_mismatch: manifest {manifest.get('benchmark')!r} "
            f"!= expected {benchmark!r}"
        )

    records = _parse_artifact_records(manifest)
    if records is None:
        return "manifest_artifacts_invalid"

    expected_names = set(HASHED_ARTIFACTS)
    got_names = {r.name for r in records}
    if got_names != expected_names:
        missing = sorted(expected_names - got_names)
        extra = sorted(got_names - expected_names)
        return f"manifest_artifacts_incomplete: missing={missing} extra={extra}"

    claimed_hash = manifest.get("content_hash")
    if not isinstance(claimed_hash, str):
        return "manifest_content_hash_missing"
    recomputed = manifest_content_hash(records)
    if claimed_hash.lower() != recomputed:
        return "manifest_content_hash_mismatch"

    for record in records:
        path = pack_dir / record.name
        if not path.exists():
            return f"manifest_artifact_missing: {record.name}"
        actual_size = path.stat().st_size
        if actual_size != record.size_bytes:
            return (
                f"manifest_size_mismatch: {record.name} "
                f"manifest {record.size_bytes} vs disk {actual_size}"
            )
        actual_digest = sha256_file(path)
        if actual_digest != record.sha256:
            return f"manifest_hash_mismatch: {record.name}"

    return None
