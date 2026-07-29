#!/usr/bin/env python3
"""Pack Milestone-1 triage weights for offline eval / sharing.

Writes::

    submissions/<miner>/m1/
      config.json
      W_domain.npy
      W_diff.npy
      attention_query.npy   # only if --pool attentive

Usage::

    # From a trained .npz (keys W_domain, W_diff[, attention_query])
    python scripts/pack_milestone1.py \\
        --weights experiments/m1/head.npz \\
        --miner-name alice --config 5-domain --pool penultimate

    # Pack a zero (uniform) head for smoke tests
    python scripts/pack_milestone1.py \\
        --zero --miner-name alice --config 5-domain
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.m1.domains import DOMAINS_20, DOMAINS_5  # noqa: E402
from trinity.m1.pack import Milestone1Pack, save_milestone1_pack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--miner-name", required=True)
    ap.add_argument("--config", default="5-domain", choices=("5-domain", "20-domain"))
    ap.add_argument("--pool", default="penultimate", choices=("penultimate", "attentive"))
    ap.add_argument("--d-h", type=int, default=1024)
    ap.add_argument("--weights", type=Path, default=None, help=".npz with W_domain / W_diff")
    ap.add_argument("--zero", action="store_true", help="Pack zero-init uniform head")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default submissions/<miner>/m1)",
    )
    args = ap.parse_args()

    out = args.out or (_REPO / "submissions" / args.miner_name / "m1")
    d_h = args.d_h
    n_dom = len(DOMAINS_5 if args.config == "5-domain" else DOMAINS_20)

    if args.zero:
        W_domain = np.zeros((n_dom, d_h), dtype=np.float32)
        W_diff = np.zeros((5, d_h), dtype=np.float32)
        aq = np.zeros(d_h, dtype=np.float32) if args.pool == "attentive" else None
    elif args.weights is not None:
        blob = np.load(args.weights)
        W_domain = np.asarray(blob["W_domain"], dtype=np.float32)
        W_diff = np.asarray(blob["W_diff"], dtype=np.float32)
        aq = None
        if args.pool == "attentive":
            if "attention_query" not in blob.files:
                raise SystemExit("attentive pack needs attention_query in .npz")
            aq = np.asarray(blob["attention_query"], dtype=np.float32)
        d_h = int(W_domain.shape[1])
    else:
        raise SystemExit("need --weights FILE.npz or --zero")

    pack = Milestone1Pack(
        config=args.config,
        pool=args.pool,
        d_h=d_h,
        W_domain=W_domain,
        W_diff=W_diff,
        attention_query=aq,
        meta={"miner": args.miner_name},
    )
    save_milestone1_pack(out, pack)
    print(f"[pack] → {out}")
    print(f"  config={pack.config} pool={pack.pool} W_domain={pack.W_domain.shape}")


if __name__ == "__main__":
    main()
