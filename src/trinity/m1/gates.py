"""Offline gates for Milestone-1 head submissions.

Miners submit **only** head (+ optional attention) weights. The host owns the
frozen encoder. Default architecture is TriageHead / AttentiveTriageRouter;
param budget and rate limit apply to every pack.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trinity.m1.constants import (
    MAX_HEAD_PARAMS,
    RATE_LIMIT_MAX_SUBMISSIONS,
    RATE_LIMIT_WINDOW_DAYS,
)
from trinity.m1.pack import Milestone1Pack, count_pack_params


@dataclass(frozen=True)
class M1GateResult:
    gate: str
    ok: bool
    reason: str | None = None

    @property
    def failed(self) -> bool:
        return not self.ok


def check_param_budget(pack: Milestone1Pack) -> M1GateResult:
    """Reject if head (+ attention) params ≥ ``MAX_HEAD_PARAMS`` (1M)."""
    n = count_pack_params(pack)
    if n >= MAX_HEAD_PARAMS:
        return M1GateResult(
            gate="param_budget",
            ok=False,
            reason=f"head params {n:,} >= limit {MAX_HEAD_PARAMS:,}",
        )
    return M1GateResult(gate="param_budget", ok=True, reason=f"n_params={n}")


def check_weights_finite(pack: Milestone1Pack) -> M1GateResult:
    try:
        pack.validate()
    except ValueError as e:
        return M1GateResult(gate="weight_sanity", ok=False, reason=str(e))
    return M1GateResult(gate="weight_sanity", ok=True)


def _attempts_path(repo_root: Path) -> Path:
    return Path(repo_root) / "submissions" / "m1_attempts.jsonl"


def load_m1_attempts(repo_root: Path) -> list[dict[str, Any]]:
    path = _attempts_path(repo_root)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def append_m1_attempt(
    repo_root: Path,
    *,
    miner: str,
    submission: str,
    ok: bool,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Append one attempt row (for rate-limit accounting)."""
    path = _attempts_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "miner": miner,
        "submission": submission,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ok": bool(ok),
        "metrics": metrics or {},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def check_rate_limit(
    miner: str,
    repo_root: Path,
    *,
    now: float | None = None,
) -> M1GateResult:
    """At most ``RATE_LIMIT_MAX_SUBMISSIONS`` M1 attempts per miner per window."""
    now = time.time() if now is None else now
    cutoff = now - RATE_LIMIT_WINDOW_DAYS * 86400
    recent = 0
    for row in load_m1_attempts(repo_root):
        if str(row.get("miner", "")).lower() != miner.lower():
            continue
        ts = row.get("ts") or ""
        # YYYY-MM-DDTHH:MM:SSZ
        try:
            import calendar

            parts = ts.rstrip("Z").split("T")
            y, m, d = (int(x) for x in parts[0].split("-"))
            hh, mm, ss = (int(float(x)) for x in parts[1].split(":"))
            epoch = calendar.timegm((y, m, d, hh, mm, ss))
        except Exception:
            continue
        if epoch >= cutoff:
            recent += 1
    if recent >= RATE_LIMIT_MAX_SUBMISSIONS:
        return M1GateResult(
            gate="rate_limit",
            ok=False,
            reason=(
                f"miner {miner!r} already has {recent} M1 submission(s) in the last "
                f"{RATE_LIMIT_WINDOW_DAYS} day(s) (max {RATE_LIMIT_MAX_SUBMISSIONS})"
            ),
        )
    return M1GateResult(
        gate="rate_limit",
        ok=True,
        reason=f"recent={recent}/{RATE_LIMIT_MAX_SUBMISSIONS}",
    )


def run_m1_gates(
    pack: Milestone1Pack,
    *,
    miner: str,
    repo_root: Path,
    skip_rate_limit: bool = False,
) -> list[M1GateResult]:
    """Run blocking M1 gates in order."""
    results = [check_weights_finite(pack), check_param_budget(pack)]
    if not skip_rate_limit:
        results.append(check_rate_limit(miner, repo_root))
    return results
