#!/usr/bin/env python3
"""Read the oracle ceiling across reachability levels L0/L1/L2. Zero API cost.

``trinity.analysis.reachability`` (merged in #417) implements ORACLE §2.2's
multi-level oracle, the L0 ≤ L1 ≤ L2 monotonicity guard, and §6's rule that the
deciding headroom is read at *"the widest reachability level (L2 if run, else
L1)"*. Until now nothing could invoke it: ``scripts/oracle_ceiling.py`` reports a
single matrix, and its INCONCLUSIVE branch advises widening the level — advice
with no tool behind it. This is that tool.

Supply one collected matrix per level::

    python scripts/reachability_report.py \\
        --matrix L0=experiments/final/oracle_matrix_math500.json \\
        --matrix L1=experiments/final/oracle_matrix_math500_l1.json

Confidence intervals are optional and come from a JSON mapping level to
``[lo, hi]``; without them a level cannot *rule out* routing, only report its
point headroom (verdict ``NEEDS_CI``)::

    python scripts/reachability_report.py --matrix L0=m0.json --cis cis.json --json

Matrices use the ``oracle_matrix`` schema that ``oracle_ceiling.py --collect``
writes. **The level is whatever you label it**: collection is single-turn Worker
(L0) today, so labelling an L0-collected matrix as ``L1`` will produce a
confident-looking reading of something that was never measured.

Exit codes: 0 a usable reading, 1 the monotonicity guard fired
(``INCONSISTENT``), 2 a usage error (nothing supplied / unreadable input).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.reachability import (  # noqa: E402
    LEVEL_ORDER,
    analyze,
    render,
)


def _parse_assignment(raw: str) -> tuple[str, Path]:
    """Split ``LEVEL=PATH``, validating the level against ORACLE §2.2."""
    if "=" not in raw:
        raise ValueError(f"expected LEVEL=PATH, got {raw!r}")
    level, _, path = raw.partition("=")
    level = level.strip().upper()
    if level not in LEVEL_ORDER:
        raise ValueError(f"unknown level {level!r}; expected one of {list(LEVEL_ORDER)}")
    if not path.strip():
        raise ValueError(f"no path given for level {level}")
    return level, Path(path.strip())


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


def _load_cis(path: Path) -> dict[str, tuple[float, float]]:
    """Load ``{level: [lo, hi]}``, validating shape and ordering."""
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object of level -> [lo, hi]")
    out: dict[str, tuple[float, float]] = {}
    for level, bounds in raw.items():
        key = str(level).strip().upper()
        if key not in LEVEL_ORDER:
            raise ValueError(f"{path}: unknown level {level!r}")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"{path}: {level} interval must be [lo, hi]")
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo > hi:
            raise ValueError(f"{path}: {level} interval is inverted ({lo} > {hi})")
        out[key] = (lo, hi)
    return out


def main(argv: list[str] | None = None) -> int:
    """Print the multi-level reachability report; exit per the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--matrix", action="append", default=[], metavar="LEVEL=PATH",
        help="oracle matrix for one level (repeatable), e.g. L1=matrix_l1.json",
    )
    ap.add_argument("--cis", type=Path, help="JSON: level -> [lo, hi] headroom CI")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="solve threshold passed through to the oracle (default 0.5)")
    ap.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    args = ap.parse_args(argv)

    if not args.matrix:
        print("no matrices supplied; pass at least one --matrix LEVEL=PATH",
              file=sys.stderr)
        return 2

    matrices: dict[str, dict] = {}
    try:
        for raw in args.matrix:
            level, path = _parse_assignment(raw)
            if level in matrices:
                raise ValueError(f"level {level} supplied more than once")
            payload = _load_json(path)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}: expected an oracle_matrix object")
            matrices[level] = payload
        cis = _load_cis(args.cis) if args.cis else None
    except FileNotFoundError as exc:
        print(f"no such file: {exc}", file=sys.stderr)
        return 2
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"bad input: {exc}", file=sys.stderr)
        return 2

    summary = analyze(matrices, threshold=args.threshold, cis=cis)

    if args.as_json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(render(summary))
        mislabel_risk = [lv for lv in matrices if lv != "L0"]
        if mislabel_risk:
            print()
            print(f"note: levels {mislabel_risk} were read from the matrices you "
                  "supplied; oracle_ceiling.py --collect does not itself branch on "
                  "level, so confirm these were collected with the wider reachability.")

    if summary.verdict == "INCONSISTENT":
        return 1
    if summary.verdict == "NO_DATA":
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
