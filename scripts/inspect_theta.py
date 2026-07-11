#!/usr/bin/env python3
"""Offline diagnostics for a trained parameter vector (theta).

Reports whether a submission's theta actually trained: per-block norms, how far
the head and SVF scales moved from their init, and any non-finite entry. It
surfaces the two silent failure modes a shape check misses — a head still at its
W=0 init (routing is the uniform policy) and SVF scales still at 1.0 (the SLM was
not adapted).

    python scripts/inspect_theta.py --theta experiments/math500/run/best_theta.npy
    python scripts/inspect_theta.py --head head_weights.npy --svf svf_scales.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.coordinator import params as P  # noqa: E402
from trinity.theta_inspect import inspect_theta  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print the theta diagnostics; exit non-zero if it never trained or is non-finite."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--theta", type=Path, help="packed theta .npy")
    ap.add_argument("--head", type=Path, help="head_weights.npy (with --svf)")
    ap.add_argument("--svf", type=Path, help="svf_scales.npy (with --head)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.theta is not None:
        theta = np.load(args.theta)
    elif args.head is not None and args.svf is not None:
        theta = P.pack(np.load(args.head), np.load(args.svf))
    else:
        ap.error("provide --theta, or both --head and --svf")

    report = inspect_theta(theta)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"[theta] {report.n_total} params | finite={report.finite} | trained={report.trained}")
        print(f"  head: moved {report.head.n_moved}/{report.head.size}, "
              f"dist_from_init {report.head.dist_from_init:.4g}, norm {report.head.l2_norm:.4g}")
        print(f"  svf:  moved {report.svf.n_moved}/{report.svf.size}, "
              f"dist_from_init {report.svf.dist_from_init:.4g}")
        for w in report.warnings:
            print(f"  WARNING: {w}")
    return 0 if (report.trained and report.finite) else 1


if __name__ == "__main__":
    raise SystemExit(main())
