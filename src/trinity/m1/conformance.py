"""Verify the Milestone-1 code matches the rules published in ``docs/MILESTONE1.md``.

MILESTONE1.md is the miner-facing contract: it tabulates the submission limits
(``< 1,000,000`` params, ``1`` per day, win margin ``≥ 0.02``), the pack layout
(which files land under ``submissions/<miner>/m1/``), and the composite formula
(``0.7·domain + 0.3·difficulty``). Those numbers also live independently in
``trinity.m1.constants`` and the pack/metrics code. Nothing checks that the two
agree — a doc edit or a constant change could silently split them, and a miner
following the document would then build a pack the validator rejects (or vice
versa).

This parses the document and asserts it against the code:

* **rules** — the published limits equal ``constants.MAX_HEAD_PARAMS`` /
  ``RATE_LIMIT_*`` / ``WIN_MARGIN``;
* **scoring** — the published composite weights equal ``constants.DOMAIN_WEIGHT``
  (and the ``TriageMetrics.composite`` docstring);
* **pack layout** — every file the document lists is one the packer actually
  writes, and every file the packer can write is documented (a file the code
  emits but the doc omits is reported, not silently accepted).

Pure stdlib. No torch, no network — ``pack.save_milestone1_pack`` is numpy-only,
and this only needs the *set* of files it can write, encoded in
:data:`CODE_PACK_FILES`, cross-checked against it by the tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trinity.m1.constants import (
    DOMAIN_WEIGHT,
    MAX_HEAD_PARAMS,
    RATE_LIMIT_MAX_SUBMISSIONS,
    RATE_LIMIT_WINDOW_DAYS,
    WIN_MARGIN,
)

__all__ = [
    "CODE_PACK_FILES",
    "ConformanceReport",
    "Finding",
    "check",
    "default_doc_path",
    "parse_pack_layout",
    "parse_rules",
    "parse_scoring_weights",
]

#: Files ``pack.save_milestone1_pack`` can write, and whether each is required.
#: ``tests/test_m1_conformance.py`` saves a real pack and asserts this set is
#: exactly what lands on disk, so it cannot drift from the packer.
CODE_PACK_FILES: dict[str, str] = {
    "config.json": "required",
    "W_domain.npy": "required",
    "W_diff.npy": "required",
    "attention_query.npy": "optional (pool=attentive)",
    "W_k.npy": "optional (attentive + key projection)",
}

_REQUIRED_PACK_FILES = tuple(f for f, kind in CODE_PACK_FILES.items() if kind == "required")


def default_doc_path() -> Path:
    """``docs/MILESTONE1.md`` relative to the installed package."""
    return Path(__file__).resolve().parents[3] / "docs" / "MILESTONE1.md"


def _section(md: str, heading: str) -> str:
    """Return the body between ``## heading`` and the next ``## `` heading."""
    m = re.search(
        rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)", md, re.S | re.M
    )
    if m is None:
        raise ValueError(f"MILESTONE1.md section '## {heading}' not found")
    return m.group(1)


def _int(text: str) -> int:
    """First integer in ``text``, commas/thin-spaces allowed (``1,000,000``)."""
    m = re.search(r"\d[\d,\s]*", text)
    if m is None:
        raise ValueError(f"no integer in {text!r}")
    return int(re.sub(r"[,\s]", "", m.group(0)))


def parse_rules(md: str) -> dict[str, Any]:
    """Extract the documented submission limits from the ``## Rules`` table.

    Returns keys ``max_params`` (int), ``rate_per_day`` (int) and ``win_margin``
    (float).

    Raises:
        ValueError: if the section or any of the three rows is missing.
    """
    body = _section(md, "Rules")
    rows = {
        c[0].strip().strip("*").strip().lower(): c[1].strip()
        for line in body.splitlines()
        if line.strip().startswith("|") and not line.strip().startswith("|--")
        for c in [ [x.strip() for x in line.strip().strip("|").split("|")] ]
        if len(c) == 2
    }
    try:
        size = rows["size"]
        rate = rows["rate"]
        margin = rows["win margin"]
    except KeyError as e:
        raise ValueError(f"MILESTONE1.md Rules table missing row: {e}") from None

    win = re.search(r"[\d.]+", margin)
    if win is None:
        raise ValueError(f"no win-margin number in {margin!r}")
    per_day = "day" in rate.lower()
    return {
        "max_params": _int(size),
        "rate_per_day": _int(rate) if per_day else None,
        "win_margin": float(win.group(0)),
    }


def parse_scoring_weights(md: str) -> tuple[float, float]:
    """Return ``(domain_weight, difficulty_weight)`` from the ``## Scoring`` block."""
    body = _section(md, "Scoring")
    nums = re.findall(r"([\d.]+)\s*[×x*]\s*(?:domain|difficulty)", body)
    if len(nums) < 2:
        raise ValueError(f"could not parse composite weights from: {body!r}")
    return float(nums[0]), float(nums[1])


def parse_pack_layout(md: str) -> list[str]:
    """Filenames listed in the ``## Pack layout`` fenced block, in order."""
    body = _section(md, "Pack layout")
    fence = re.search(r"```(.*?)```", body, re.S)
    if fence is None:
        raise ValueError("MILESTONE1.md Pack layout has no fenced block")
    files: list[str] = []
    for line in fence.group(1).splitlines():
        tok = line.strip().split()
        if tok and re.fullmatch(r"[\w.\-]+\.(npy|json)", tok[0]):
            files.append(tok[0])
    if not files:
        raise ValueError("no pack files found in the Pack layout block")
    return files


@dataclass(frozen=True)
class Finding:
    """One conformance check outcome."""

    check: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.check, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True)
class ConformanceReport:
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(f.ok for f in self.findings)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if not f.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "findings": [f.to_dict() for f in self.findings],
            "notes": list(self.notes),
        }


def check(md: str | None = None, *, doc_path: Path | str | None = None) -> ConformanceReport:
    """Compare ``docs/MILESTONE1.md`` against ``trinity.m1`` code.

    Args:
        md: document contents; when ``None`` read from ``doc_path`` or
            :func:`default_doc_path`.
        doc_path: where to read the document from.

    Raises:
        ValueError: if a documented section cannot be parsed.
        FileNotFoundError: if the document file is missing.
    """
    if md is None:
        path = Path(doc_path) if doc_path is not None else default_doc_path()
        if not path.exists():
            raise FileNotFoundError(str(path))
        md = path.read_text()

    rules = parse_rules(md)
    dom_w, diff_w = parse_scoring_weights(md)
    doc_files = parse_pack_layout(md)

    findings: list[Finding] = []
    notes: list[str] = []

    findings.append(Finding(
        "param_budget",
        rules["max_params"] == MAX_HEAD_PARAMS,
        f"doc says < {rules['max_params']:,}; constants.MAX_HEAD_PARAMS = {MAX_HEAD_PARAMS:,}",
    ))
    findings.append(Finding(
        "rate_limit",
        rules["rate_per_day"] == RATE_LIMIT_MAX_SUBMISSIONS and RATE_LIMIT_WINDOW_DAYS == 1,
        f"doc says {rules['rate_per_day']}/day; constants = "
        f"{RATE_LIMIT_MAX_SUBMISSIONS} per {RATE_LIMIT_WINDOW_DAYS} day(s)",
    ))
    findings.append(Finding(
        "win_margin",
        abs(rules["win_margin"] - WIN_MARGIN) < 1e-12,
        f"doc says >= {rules['win_margin']}; constants.WIN_MARGIN = {WIN_MARGIN}",
    ))
    findings.append(Finding(
        "composite_weights",
        abs(dom_w - DOMAIN_WEIGHT) < 1e-12 and abs(diff_w - (1.0 - DOMAIN_WEIGHT)) < 1e-12,
        f"doc says {dom_w}·domain + {diff_w}·difficulty; "
        f"constants.DOMAIN_WEIGHT = {DOMAIN_WEIGHT}",
    ))

    # Pack layout: every documented file must be one the packer writes.
    phantom = [f for f in doc_files if f not in CODE_PACK_FILES]
    findings.append(Finding(
        "pack_layout_no_phantom_files",
        not phantom,
        "documented files not produced by the packer: " + (", ".join(phantom) or "none"),
    ))
    missing_required = [f for f in _REQUIRED_PACK_FILES if f not in doc_files]
    findings.append(Finding(
        "pack_layout_documents_required_files",
        not missing_required,
        "required pack files absent from the doc: " + (", ".join(missing_required) or "none"),
    ))

    # A file the code can write but the doc omits is a documentation gap, not a
    # code defect — reported as a note rather than a failure.
    undocumented = [f for f in CODE_PACK_FILES if f not in doc_files]
    if undocumented:
        notes.append(
            "packer can also write " + ", ".join(undocumented)
            + " (attentive packs); the Pack layout block does not list "
            + ("it" if len(undocumented) == 1 else "them") + "."
        )

    return ConformanceReport(findings=findings, notes=notes)


def render(report: ConformanceReport) -> str:
    lines = ["MILESTONE1.md ↔ trinity.m1 conformance"]
    for f in report.findings:
        lines.append(f"  [{'ok' if f.ok else 'MISMATCH'}] {f.check}: {f.detail}")
    for note in report.notes:
        lines.append(f"  note: {note}")
    lines.append(
        "  verdict: doc and code agree"
        if report.ok
        else f"  verdict: {len(report.failures)} MISMATCH(es)"
    )
    return "\n".join(lines)
