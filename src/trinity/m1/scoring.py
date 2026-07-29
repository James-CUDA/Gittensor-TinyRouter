"""Score an M1 pack (or router) on a fixed split — shared by eval + validator."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from trinity.m1.metrics import TriageMetrics, score_triage
from trinity.m1.pack import Milestone1Pack, load_milestone1_pack


def load_eval_split(
    config: str,
    split: str,
    *,
    hub: str = "James-Cuda/tinyrouter-m1",
    local: Path | None = None,
    limit: int = 0,
) -> tuple[list[str], list[str], list[str], list[int]]:
    """Return ``(ids, prompts, domains, difficulties)``."""
    if local is not None and Path(local).exists():
        import pandas as pd

        df = pd.read_parquet(local)
        if limit > 0:
            df = df.iloc[:limit]
        return (
            df["id"].astype(str).tolist(),
            df["prompt"].astype(str).tolist(),
            df["domain"].astype(str).tolist(),
            [int(x) for x in df["difficulty"].tolist()],
        )
    from datasets import load_dataset

    ds = load_dataset(hub, config, split=split)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return (
        list(ds["id"]),
        list(ds["prompt"]),
        list(ds["domain"]),
        [int(x) for x in ds["difficulty"]],
    )


def predict_penultimate(head, features: np.ndarray) -> tuple[list[str], list[int]]:
    import torch

    pred_d, pred_f = [], []
    for i in range(features.shape[0]):
        h = torch.from_numpy(np.asarray(features[i], dtype=np.float32))
        d, f, _ = head.select(h, sample=False)
        pred_d.append(d)
        pred_f.append(f)
    return pred_d, pred_f


def predict_live(
    router,
    prompts: Sequence[str],
    *,
    pool: str,
    device: str,
    model_name: str,
) -> tuple[list[str], list[int]]:
    import torch
    from trinity.coordinator.slm import CoordinatorEncoder
    from trinity.orchestration.session import _transcript_text

    enc = CoordinatorEncoder(model_name=model_name, device=device)
    pred_d, pred_f = [], []
    for i, p in enumerate(prompts):
        text = _transcript_text(p, [])
        if pool == "penultimate":
            h = torch.from_numpy(enc.encode(text))
            if hasattr(router, "head") and hasattr(router, "pool"):
                d, f, _ = router.head.select(h, sample=False)
            else:
                d, f, _ = router.select(h, sample=False)
        else:
            H, mask = enc.encode_sequence(text)
            d, f, _ = router.select(
                torch.from_numpy(H), torch.from_numpy(mask), sample=False
            )
        pred_d.append(d)
        pred_f.append(f)
    return pred_d, pred_f


def score_pack(
    pack: Milestone1Pack | str | Path,
    *,
    prompts: Sequence[str],
    y_domain: Sequence[str],
    y_diff: Sequence[int],
    features: np.ndarray | None = None,
    device: str = "cuda:0",
    model_name: str = "Qwen/Qwen3-0.6B",
    domain_weight: float = 0.7,
) -> TriageMetrics:
    """Score one pack on an already-loaded split."""
    if not isinstance(pack, Milestone1Pack):
        pack = load_milestone1_pack(pack)
    pack.validate()
    if features is not None:
        if pack.pool != "penultimate":
            raise ValueError("features scoring requires pool=penultimate")
        if features.shape[0] != len(prompts):
            raise ValueError(
                f"features N={features.shape[0]} != prompts N={len(prompts)}"
            )
        pred_d, pred_f = predict_penultimate(pack.build_head(), features)
    else:
        pred_d, pred_f = predict_live(
            pack.build_router(),
            list(prompts),
            pool=pack.pool,
            device=device,
            model_name=model_name,
        )
    return score_triage(
        list(y_domain), list(y_diff), pred_d, pred_f, domain_weight=domain_weight
    )


def compare_report(
    *,
    king_metrics: TriageMetrics | None,
    challenger_metrics: TriageMetrics,
    win_margin: float,
    king_meta: dict[str, Any] | None = None,
    challenger_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from trinity.m1.leaderboard import decide_m1_winner

    kc = None if king_metrics is None else king_metrics.composite
    wins, reason = decide_m1_winner(
        king_composite=kc,
        challenger_composite=challenger_metrics.composite,
        win_margin=win_margin,
    )
    return {
        "merge": wins,
        "reason": reason,
        "win_margin": win_margin,
        "king": None
        if king_metrics is None
        else {
            **(king_meta or {}),
            "composite": king_metrics.composite,
            "metrics": king_metrics.as_dict(),
        },
        "challenger": {
            **(challenger_meta or {}),
            "composite": challenger_metrics.composite,
            "metrics": challenger_metrics.as_dict(),
        },
    }
