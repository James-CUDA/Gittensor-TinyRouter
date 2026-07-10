"""Load a routing-head submission directory into typed artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["SubmissionPack", "load_submission_pack", "parse_submission_identity"]


@dataclass(frozen=True)
class SubmissionPack:
    """One miner generation's submission artifacts."""

    path: Path
    miner: str
    generation: int
    head_weights: np.ndarray
    svf_scales: np.ndarray
    receipt: dict[str, Any]

    @property
    def has_receipt(self) -> bool:
        return bool(self.receipt)


def parse_submission_identity(submission_dir: Path, submissions_root: Path) -> tuple[str, int]:
    """Infer miner name and generation from ``submissions/<miner>/<gen>/`` layout."""
    rel = submission_dir.resolve().relative_to(submissions_root.resolve())
    parts = rel.parts
    if len(parts) < 2:
        return submission_dir.name, 0
    miner = parts[0]
    try:
        generation = int(parts[1])
    except ValueError:
        generation = 0
    return miner, generation


def load_submission_pack(
    submission_dir: Path,
    *,
    submissions_root: Path | None = None,
) -> SubmissionPack | None:
    """Load ``head_weights.npy``, ``svf_scales.npy``, and optional ``receipt.json``.

    Returns ``None`` when required weight files are missing or unreadable.
    """
    root = submissions_root or submission_dir.parent.parent
    miner, generation = parse_submission_identity(submission_dir, root)

    hw_path = submission_dir / "head_weights.npy"
    sv_path = submission_dir / "svf_scales.npy"
    rc_path = submission_dir / "receipt.json"

    if not hw_path.exists() or not sv_path.exists():
        return None

    try:
        head_weights = np.load(str(hw_path)).astype(np.float32)
        svf_scales = np.load(str(sv_path)).astype(np.float32)
    except (ValueError, OSError):
        return None

    receipt: dict[str, Any] = {}
    if rc_path.exists():
        try:
            receipt = json.loads(rc_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            receipt = {}

    return SubmissionPack(
        path=submission_dir,
        miner=miner,
        generation=generation,
        head_weights=head_weights,
        svf_scales=svf_scales,
        receipt=receipt,
    )
