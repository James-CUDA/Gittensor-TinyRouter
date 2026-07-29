"""Milestone-1 king / history record (host-side)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trinity.m1.constants import M1_LEADERBOARD_NAME, WIN_MARGIN


@dataclass
class M1King:
    miner: str
    submission: str
    config: str
    composite: float
    metrics: dict[str, Any] = field(default_factory=dict)
    ts: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class M1Leaderboard:
    config: str
    king: M1King | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "win_margin": WIN_MARGIN,
            "king": None if self.king is None else self.king.as_dict(),
            "history": list(self.history),
        }


def leaderboard_path(repo_root: Path, config: str) -> Path:
    """Per-taxonomy file: ``m1_leaderboard_5-domain.json`` etc."""
    safe = config.replace("/", "-")
    stem = M1_LEADERBOARD_NAME.replace(".json", f"_{safe}.json")
    return Path(repo_root) / "submissions" / stem


def load_m1_leaderboard(repo_root: Path, config: str) -> M1Leaderboard:
    path = leaderboard_path(repo_root, config)
    if not path.exists():
        return M1Leaderboard(config=config)
    raw = json.loads(path.read_text(encoding="utf-8"))
    king_raw = raw.get("king")
    king = M1King(**king_raw) if king_raw else None
    return M1Leaderboard(
        config=str(raw.get("config", config)),
        king=king,
        history=list(raw.get("history") or []),
    )


def save_m1_leaderboard(repo_root: Path, lb: M1Leaderboard) -> Path:
    path = leaderboard_path(repo_root, lb.config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lb.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def decide_m1_winner(
    *,
    king_composite: float | None,
    challenger_composite: float,
    win_margin: float = WIN_MARGIN,
) -> tuple[bool, str]:
    """Return ``(challenger_wins, reason)``.

    No king → first valid challenger becomes king (no margin).
    With king → need ``challenger >= king + win_margin``.
    """
    if king_composite is None:
        return True, "no_king_challenger_becomes_first_king"
    need = float(king_composite) + float(win_margin)
    if challenger_composite >= need:
        return True, (
            f"challenger {challenger_composite:.4f} >= king {king_composite:.4f} "
            f"+ margin {win_margin:.4f} (need {need:.4f})"
        )
    return False, (
        f"challenger {challenger_composite:.4f} < king {king_composite:.4f} "
        f"+ margin {win_margin:.4f} (need {need:.4f})"
    )


def promote_king(
    lb: M1Leaderboard,
    *,
    miner: str,
    submission: str,
    composite: float,
    metrics: dict[str, Any],
) -> M1Leaderboard:
    """Install challenger as king and append history row."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prev = None if lb.king is None else lb.king.as_dict()
    lb.history.append(
        {
            "ts": ts,
            "event": "promote",
            "miner": miner,
            "submission": submission,
            "composite": composite,
            "previous_king": prev,
        }
    )
    lb.king = M1King(
        miner=miner,
        submission=submission,
        config=lb.config,
        composite=float(composite),
        metrics=dict(metrics),
        ts=ts,
    )
    return lb
