#!/usr/bin/env python3
"""Evaluate a Milestone-1 triage submission (domain + difficulty).

Offline only — no OpenRouter. Scores a miner pack (or a zero/random baseline)
on ``James-Cuda/tinyrouter-m1`` test (or a local parquet).

Metrics:
  domain accuracy / macro-F1, difficulty exact / within-1, joint, composite
  (composite = 0.7 * domain_acc + 0.3 * difficulty_exact).

Usage::

    # Score a packed submission (needs GPU + encoder unless --features is set)
    python scripts/eval_milestone1.py \\
        --submission submissions/alice/m1 \\
        --config 5-domain

    # Precomputed penultimate features (N, d_h) matching test order
    python scripts/eval_milestone1.py \\
        --submission submissions/alice/m1 \\
        --features experiments/m1/test_features.npy

    # Majority-domain + mid difficulty baseline (no pack)
    python scripts/eval_milestone1.py --config 5-domain --baseline majority
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.m1.metrics import score_triage  # noqa: E402



def _load_split(config: str, split: str, *, hub: str, local: Path | None, limit: int):
    if local is not None and local.exists():
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


def _predict_baseline(domains: list[str], diffs: list[int], kind: str):
    if kind == "majority":
        maj_d = Counter(domains).most_common(1)[0][0]
        # median difficulty on labels
        mid = int(np.median(np.asarray(diffs, dtype=int)))
        return [maj_d] * len(domains), [mid] * len(diffs)
    if kind == "random":
        rng = np.random.default_rng(0)
        uniq = sorted(set(domains))
        return (
            [uniq[i] for i in rng.integers(0, len(uniq), size=len(domains))],
            [int(x) for x in rng.integers(1, 6, size=len(diffs))],
        )
    raise ValueError(kind)


def _predict_penultimate(head, features: np.ndarray) -> tuple[list[str], list[int]]:
    import torch

    pred_d, pred_f = [], []
    for i in range(features.shape[0]):
        h = torch.from_numpy(np.asarray(features[i], dtype=np.float32))
        d, f, _ = head.select(h, sample=False)
        pred_d.append(d)
        pred_f.append(f)
    return pred_d, pred_f


def _predict_live(router, prompts: list[str], *, pool: str, device: str, model_name: str):
    import torch
    from trinity.coordinator.slm import CoordinatorEncoder
    from trinity.orchestration.session import _transcript_text

    enc = CoordinatorEncoder(model_name=model_name, device=device)
    pred_d, pred_f = [], []
    for i, p in enumerate(prompts):
        text = _transcript_text(p, [])
        if pool == "penultimate":
            h = torch.from_numpy(enc.encode(text))
            if hasattr(router, "select") and not hasattr(router, "pool"):
                d, f, _ = router.select(h, sample=False)
            else:
                # Attentive router mis-used: fall back to head on penultimate
                d, f, _ = router.head.select(h, sample=False) if hasattr(router, "head") else router.select(h, sample=False)
        else:
            H, mask = enc.encode_sequence(text)
            Ht = torch.from_numpy(H)
            mt = torch.from_numpy(mask)
            d, f, _ = router.select(Ht, mt, sample=False)
        pred_d.append(d)
        pred_f.append(f)
        if (i + 1) % 50 == 0:
            print(f"[eval] {i + 1}/{len(prompts)}", flush=True)
    return pred_d, pred_f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="5-domain", choices=("5-domain", "20-domain"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--hub", default="James-Cuda/tinyrouter-m1")
    ap.add_argument(
        "--local",
        type=Path,
        default=None,
        help="Optional local parquet (columns id,domain,difficulty,prompt)",
    )
    ap.add_argument("--submission", type=Path, default=None, help="M1 pack directory")
    ap.add_argument(
        "--baseline",
        choices=("majority", "random"),
        default=None,
        help="Score a trivial baseline instead of a pack",
    )
    ap.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Precomputed penultimate features .npy shape (N, d_h), row-aligned to split",
    )
    ap.add_argument("--limit", type=int, default=0, help="Eval first N rows (0 = all)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--domain-weight", type=float, default=0.7)
    ap.add_argument("--out", type=Path, default=None, help="Write metrics JSON here")
    ap.add_argument(
        "--miner-name",
        default=None,
        help="Miner id for rate-limit / attempt log (required with --record-attempt)",
    )
    ap.add_argument(
        "--record-attempt",
        action="store_true",
        help="Run M1 gates and append to submissions/m1_attempts.jsonl (1/day)",
    )
    ap.add_argument(
        "--skip-rate-limit",
        action="store_true",
        help="Skip 1/day gate (local dry-run)",
    )
    args = ap.parse_args()

    ids, prompts, y_dom, y_diff = _load_split(
        args.config,
        args.split,
        hub=args.hub,
        local=args.local
        or (_REPO / f"datasets/tinyrouter-m1/{args.config}/{args.split}.parquet"),
        limit=args.limit,
    )
    print(f"[data] config={args.config} split={args.split} n={len(prompts)}")

    if args.baseline:
        pred_d, pred_f = _predict_baseline(y_dom, y_diff, args.baseline)
        pool = "baseline"
    else:
        if args.submission is None:
            raise SystemExit("need --submission PACK or --baseline majority|random")
        from trinity.m1.gates import append_m1_attempt, run_m1_gates
        from trinity.m1.pack import load_milestone1_pack

        pack = load_milestone1_pack(args.submission)
        if pack.config != args.config:
            print(
                f"[warn] pack config={pack.config!r} vs --config {args.config!r}",
                file=sys.stderr,
            )
        miner = args.miner_name or (
            pack.meta.get("miner") if pack.meta else None
        ) or args.submission.parent.name
        if args.record_attempt:
            gate_results = run_m1_gates(
                pack,
                miner=str(miner),
                repo_root=_REPO,
                skip_rate_limit=args.skip_rate_limit,
            )
            for g in gate_results:
                tag = "PASS" if g.ok else "FAIL"
                print(f"[gate {tag}] {g.gate}: {g.reason}")
            if any(g.failed for g in gate_results):
                append_m1_attempt(
                    _REPO,
                    miner=str(miner),
                    submission=str(args.submission),
                    ok=False,
                )
                raise SystemExit("M1 gates failed")

        router = pack.build_router()
        pool = pack.pool
        if args.features is not None:
            if pool != "penultimate":
                raise SystemExit("--features only supported for penultimate packs")
            feats = np.load(args.features)
            if feats.shape[0] != len(prompts):
                raise SystemExit(
                    f"features N={feats.shape[0]} != prompts N={len(prompts)}"
                )
            head = pack.build_head()
            pred_d, pred_f = _predict_penultimate(head, feats)
        else:
            pred_d, pred_f = _predict_live(
                router,
                prompts,
                pool=pool,
                device=args.device,
                model_name=args.model_name,
            )

    metrics = score_triage(
        y_dom, y_diff, pred_d, pred_f, domain_weight=args.domain_weight
    )
    report = {
        "config": args.config,
        "split": args.split,
        "pool": pool if not args.baseline else f"baseline:{args.baseline}",
        "submission": str(args.submission) if args.submission else None,
        "metrics": metrics.as_dict(),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[write] {args.out}")

    if args.record_attempt and not args.baseline:
        from trinity.m1.gates import append_m1_attempt

        miner = args.miner_name or args.submission.parent.name
        append_m1_attempt(
            _REPO,
            miner=str(miner),
            submission=str(args.submission),
            ok=True,
            metrics=metrics.as_dict(),
        )
        print(f"[attempt] recorded for miner={miner} (counts toward 1/day)")


if __name__ == "__main__":
    main()
