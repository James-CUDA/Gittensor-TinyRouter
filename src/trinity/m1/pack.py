"""Milestone-1 submission pack: triage head (+ optional attention) weights."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trinity.m1.constants import MAX_HEAD_PARAMS
from trinity.m1.domains import DOMAINS_20, DOMAINS_5, N_DIFFICULTY


def count_pack_params(pack: "Milestone1Pack") -> int:
    """Trainable head (+ optional attention) parameter count."""
    n = int(pack.W_domain.size + pack.W_diff.size)
    if pack.attention_query is not None:
        n += int(pack.attention_query.size)
    if pack.W_k is not None:
        n += int(pack.W_k.size)
    return n


@dataclass
class Milestone1Pack:
    """On-disk M1 miner pack under ``submissions/<miner>/m1/``."""

    config: str  # "5-domain" | "20-domain"
    pool: str  # "penultimate" | "attentive"
    d_h: int
    W_domain: np.ndarray  # (C, d_h)
    W_diff: np.ndarray  # (5, d_h)
    attention_query: np.ndarray | None = None  # (d_h,)
    W_k: np.ndarray | None = None  # (d_h, d_h) optional key proj
    meta: dict[str, Any] | None = None

    @property
    def n_domains(self) -> int:
        return int(self.W_domain.shape[0])

    def expected_domains(self) -> tuple[str, ...]:
        key = self.config.strip().lower().replace("_", "-")
        if key in ("5-domain", "5"):
            return DOMAINS_5
        if key in ("20-domain", "20", "gci"):
            return DOMAINS_20
        raise ValueError(f"unknown config {self.config!r}")

    def validate(self) -> None:
        domains = self.expected_domains()
        if self.W_domain.shape != (len(domains), self.d_h):
            raise ValueError(
                f"W_domain shape {self.W_domain.shape} != {(len(domains), self.d_h)}"
            )
        if self.W_diff.shape != (N_DIFFICULTY, self.d_h):
            raise ValueError(
                f"W_diff shape {self.W_diff.shape} != {(N_DIFFICULTY, self.d_h)}"
            )
        if not np.isfinite(self.W_domain).all() or not np.isfinite(self.W_diff).all():
            raise ValueError("W_domain / W_diff contain NaN/Inf")
        if self.pool not in ("penultimate", "attentive"):
            raise ValueError(f"pool must be penultimate|attentive, got {self.pool!r}")
        if self.pool == "attentive":
            if self.attention_query is None:
                raise ValueError("attentive pack requires attention_query.npy")
            if self.attention_query.shape != (self.d_h,):
                raise ValueError(
                    f"attention_query shape {self.attention_query.shape} != {(self.d_h,)}"
                )
        if self.W_k is not None and self.W_k.shape != (self.d_h, self.d_h):
            raise ValueError(f"W_k shape {self.W_k.shape} != {(self.d_h, self.d_h)}")
        n = count_pack_params(self)
        if n >= MAX_HEAD_PARAMS:
            raise ValueError(
                f"head params {n:,} >= Milestone-1 limit {MAX_HEAD_PARAMS:,}"
            )

    @property
    def n_params(self) -> int:
        return count_pack_params(self)

    def build_head(self):
        """Build TriageHead (imports torch)."""
        from trinity.coordinator.triage_head import make_triage_head

        self.validate()
        head = make_triage_head(self.config, d_h=self.d_h)
        head.load_weights(self.W_domain.astype(np.float32), self.W_diff.astype(np.float32))
        return head

    def build_router(self):
        """Return TriageHead or AttentiveTriageRouter (imports torch)."""
        head = self.build_head()
        if self.pool == "penultimate":
            return head
        from trinity.coordinator.attention_pool import AttentiveTriageRouter

        use_key = self.W_k is not None
        router = AttentiveTriageRouter(
            head, d_h=self.d_h, use_key_proj=use_key, l2_normalize=True
        )
        q = self.attention_query.astype(np.float64)
        if use_key:
            theta = np.concatenate([q.reshape(-1), self.W_k.astype(np.float64).reshape(-1)])
        else:
            theta = q.reshape(-1)
        router.pool.unpack_into(theta)
        return router


def save_milestone1_pack(path: str | Path, pack: Milestone1Pack) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    pack.validate()
    cfg = {
        "config": pack.config,
        "pool": pack.pool,
        "d_h": int(pack.d_h),
        "use_key_proj": pack.W_k is not None,
        "n_domains": pack.n_domains,
        "n_diff": int(N_DIFFICULTY),
        "schema": "tinyrouter-m1-v1",
    }
    if pack.meta:
        cfg["meta"] = pack.meta
    (path / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    np.save(path / "W_domain.npy", pack.W_domain.astype(np.float32))
    np.save(path / "W_diff.npy", pack.W_diff.astype(np.float32))
    if pack.attention_query is not None:
        np.save(path / "attention_query.npy", pack.attention_query.astype(np.float32))
    if pack.W_k is not None:
        np.save(path / "W_k.npy", pack.W_k.astype(np.float32))
    return path


def load_milestone1_pack(path: str | Path) -> Milestone1Pack:
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    W_domain = np.load(path / "W_domain.npy")
    W_diff = np.load(path / "W_diff.npy")
    aq = path / "attention_query.npy"
    wk = path / "W_k.npy"
    pack = Milestone1Pack(
        config=str(cfg["config"]),
        pool=str(cfg.get("pool", "penultimate")),
        d_h=int(cfg.get("d_h", W_domain.shape[1])),
        W_domain=W_domain,
        W_diff=W_diff,
        attention_query=np.load(aq) if aq.exists() else None,
        W_k=np.load(wk) if wk.exists() else None,
        meta=cfg.get("meta"),
    )
    pack.validate()
    return pack
