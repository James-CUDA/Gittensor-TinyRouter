"""Sandboxed EvalPlus solution evaluator (issue #254).

The EvalPlus adapter (:mod:`trinity.adapters.evalplus`) turns a task into a prompt and
lets a model emit a full Python solution; this module is the *dedicated runner* that
grades that solution by assembling it with the task's ``check(candidate)`` harness and
running it in a **subprocess with a wall-clock timeout**. It grades the *base* set
first (when present) and then the rigorous *plus* set, and a solution is *resolved*
only if the plus set passes — EvalPlus's whole point is catching solutions that pass
the sparse base tests but fail the augmented ones.

Isolation & safety, mirroring the repo's ``reward.run_pass_at_1`` sandbox: the
candidate code is **never** ``exec``'d in-process — it is written to a temp file and
run in a fresh subprocess, and each harness is ``compile``-checked first so an
unparseable module is a clean failure rather than a crash. No network and no GPU.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evalplus import extract_solution_code

__all__ = [
    "EvalPlusResult",
    "build_harness",
    "run_module",
    "evaluate_solution",
    "score_evalplus",
]

#: Default wall-clock limit (seconds) for one harness run.
_TEST_TIMEOUT = 120


@dataclass
class EvalPlusResult:
    """Outcome of grading one candidate solution.

    ``passed`` is the binary reward (``True`` iff the plus set passes). ``reason`` is a
    stable tag (``no_code_found`` / ``no_tests`` / ``harness_error`` / ``base_failed``
    / ``plus_failed`` / ``passed``); ``base_pass`` / ``plus_pass`` record each set.
    """

    passed: bool
    reason: str
    base_pass: bool = False
    plus_pass: bool = False
    detail: str = ""

    @property
    def reward(self) -> float:
        return 1.0 if self.passed else 0.0


def build_harness(solution: str, check_src: str, entry_point: str) -> str:
    """Assemble a runnable module: solution, the ``check`` harness, and its call.

    The solution comes first so ``check`` can reference the entry point it defines;
    then the ``check(candidate)`` harness; then ``check(<entry_point>)`` so the module
    raises (and the process exits non-zero) iff an assertion fails.
    """
    parts = [
        solution.rstrip("\n"),
        "",
        check_src.rstrip("\n"),
        "",
        f"check({entry_point})",
        "",
    ]
    return "\n".join(parts)


def run_module(source: str, *, timeout: int = _TEST_TIMEOUT) -> tuple[bool, str]:
    """Write ``source`` to a temp module and run it in a subprocess.

    Returns ``(passed, detail)``. ``passed`` is ``True`` only if the process exits
    ``0`` (no assertion or error). The module runs with the cache disabled and a
    wall-clock timeout, so a hanging or crashing solution is contained.
    """
    with tempfile.TemporaryDirectory(prefix="evalplus-") as tmp:
        script = Path(tmp) / "harness.py"
        script.write_text(source, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-B", str(script)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "harness timed out"
        except (OSError, ValueError) as exc:
            return False, f"harness could not run: {exc}"
    tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-15:])
    return proc.returncode == 0, tail


def _run_check(solution: str, check_src: str, entry_point: str, timeout: int) -> tuple[bool, str]:
    """Compile-check then run one ``check`` harness; returns ``(passed, detail)``."""
    harness = build_harness(solution, check_src, entry_point)
    try:
        compile(harness, "<evalplus>", "exec")
    except SyntaxError as exc:
        return False, f"syntax: {exc}"
    return run_module(harness, timeout=timeout)


def evaluate_solution(
    candidate: str,
    reference: Any,
    *,
    timeout: int = _TEST_TIMEOUT,
) -> EvalPlusResult:
    """Grade ``candidate`` against ``reference``'s base and plus harnesses.

    Runs the base set first (when the reference carries one) and then the plus set;
    the solution is resolved only if the plus set passes. Never raises for a bad
    solution — an unextractable solution, an unparseable harness, or a failing test is
    a clean ``passed=False`` result with a reason.
    """
    ref = reference if isinstance(reference, dict) else {}
    entry_point = str(ref.get("entry_point", "") or "")
    plus_test = str(ref.get("plus_test", "") or "")
    base_test = str(ref.get("base_test", "") or "")

    solution = extract_solution_code(candidate)
    if not solution.strip():
        return EvalPlusResult(False, "no_code_found")
    if not plus_test.strip() or not entry_point:
        return EvalPlusResult(False, "no_tests")

    base_pass = True
    if base_test.strip():
        base_pass, base_detail = _run_check(solution, base_test, entry_point, timeout)
        if not base_pass:
            reason = "harness_error" if base_detail.startswith("syntax:") else "base_failed"
            return EvalPlusResult(False, reason, base_pass=False, detail=base_detail)

    plus_pass, plus_detail = _run_check(solution, plus_test, entry_point, timeout)
    if plus_detail.startswith("syntax:"):
        return EvalPlusResult(False, "harness_error", base_pass=base_pass, detail=plus_detail)
    if not plus_pass:
        return EvalPlusResult(False, "plus_failed", base_pass=base_pass, plus_pass=False, detail=plus_detail)
    return EvalPlusResult(True, "passed", base_pass=base_pass, plus_pass=True, detail=plus_detail)


def score_evalplus(
    candidate: str,
    reference: Any,
    *,
    timeout: int = _TEST_TIMEOUT,
) -> float | None:
    """Binary reward for a candidate solution, or ``None`` if it cannot be executed.

    Returns ``1.0``/``0.0`` from :func:`evaluate_solution` when the reference carries a
    plus harness; with none there is nothing to run against, so it returns ``None`` —
    the caller (the adapter) then falls back to its cheap exact-match placeholder.
    """
    ref = reference if isinstance(reference, dict) else {}
    if not str(ref.get("plus_test", "") or "").strip():
        return None
    return evaluate_solution(candidate, reference, timeout=timeout).reward
