"""Sandboxed BigCodeBench solution evaluator (issue #212).

The BigCodeBench adapter (:mod:`trinity.adapters.bigcodebench`) turns a task into a
prompt and lets a model emit a full Python solution; this module is the *dedicated
runner* that grades that solution by assembling it with the task's
``unittest.TestCase`` harness and running the module in a **subprocess with a
wall-clock timeout**. It is the execution counterpart to the adapter's cheap
exact-match placeholder, and the BigCodeBench analogue of ``swebench_runner``.

Isolation & safety, mirroring the repo's existing ``reward.run_pass_at_1`` sandbox:
the candidate code is **never** ``exec``'d in-process — it is written to a temp
file and run with the current interpreter in a fresh subprocess, and the harness is
``compile``-checked first so an unparseable solution is a clean failure rather than a
crash. No network and no GPU: the whole grading mechanism runs against a throwaway
local module, so it is fully offline-testable.

This module is imported only when a solution is actually executed, and it touches no
other benchmark's scoring path.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bigcodebench import extract_solution_code

__all__ = [
    "BigCodeBenchResult",
    "build_harness",
    "run_module",
    "evaluate_solution",
    "score_bigcodebench",
]

#: Default wall-clock limit (seconds) for one harness run. Bounded so an
#: adversarial or hanging solution cannot stall the evaluator.
_TEST_TIMEOUT = 120


@dataclass
class BigCodeBenchResult:
    """Outcome of grading one candidate solution.

    ``passed`` is the binary reward. ``reason`` is a stable machine-readable tag
    (``no_code_found`` / ``no_tests`` / ``harness_error`` / ``tests_failed`` /
    ``passed``), and ``detail`` carries human-readable context (a tail of the
    unittest output) without leaking into the reward.
    """

    passed: bool
    reason: str
    detail: str = ""

    @property
    def reward(self) -> float:
        return 1.0 if self.passed else 0.0


def build_harness(solution: str, test: str) -> str:
    """Assemble a runnable module from a solution and its unittest harness.

    The solution is placed first so the test's ``TestCase`` can reference the
    entry point it defines, then the test source, then a ``unittest.main()`` guard
    so the module exits non-zero iff a test fails. The guard is only appended when
    the harness does not already call ``unittest.main`` itself.
    """
    parts = [solution.rstrip("\n"), "", test.rstrip("\n"), ""]
    if "unittest.main(" not in test:
        parts += [
            "if __name__ == '__main__':",
            "    import unittest",
            "    unittest.main()",
            "",
        ]
    return "\n".join(parts)


def run_module(source: str, *, timeout: int = _TEST_TIMEOUT) -> tuple[bool, str]:
    """Write ``source`` to a temp module and run it in a subprocess.

    Returns ``(all_passed, detail)``. The module runs in a fresh
    ``python <file>`` process with the cache disabled and a wall-clock timeout, so
    a hanging or crashing solution is contained. ``all_passed`` is ``True`` only if
    the process exits ``0`` (every unittest assertion passed).
    """
    with tempfile.TemporaryDirectory(prefix="bigcodebench-") as tmp:
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


def evaluate_solution(
    candidate: str,
    reference: Any,
    *,
    timeout: int = _TEST_TIMEOUT,
) -> BigCodeBenchResult:
    """Grade ``candidate`` against ``reference``'s unittest harness.

    Steps: extract the solution code, assemble it with the harness, ``compile``-check
    the module, then run it. Never raises for a bad solution — an unextractable
    solution, an unparseable module, or a failing test is a clean ``passed=False``
    result with a reason.
    """
    ref = reference if isinstance(reference, dict) else {}
    test = str(ref.get("test", "") or "")

    solution = extract_solution_code(candidate)
    if not solution.strip():
        return BigCodeBenchResult(False, "no_code_found")
    if not test.strip():
        return BigCodeBenchResult(False, "no_tests")

    harness = build_harness(solution, test)
    try:
        compile(harness, "<bigcodebench>", "exec")
    except SyntaxError as exc:
        return BigCodeBenchResult(False, "harness_error", detail=str(exc))

    ok, detail = run_module(harness, timeout=timeout)
    if ok:
        return BigCodeBenchResult(True, "passed", detail=detail)
    return BigCodeBenchResult(False, "tests_failed", detail=detail)


def score_bigcodebench(
    candidate: str,
    reference: Any,
    *,
    timeout: int = _TEST_TIMEOUT,
) -> float | None:
    """Binary reward for a candidate solution, or ``None`` if it cannot be executed.

    Returns ``1.0``/``0.0`` from :func:`evaluate_solution` when the reference carries
    a unittest harness; with no harness there is nothing to run against, so it
    returns ``None`` — the caller (the adapter) then falls back to its cheap
    exact-match placeholder rather than guessing.
    """
    ref = reference if isinstance(reference, dict) else {}
    if not str(ref.get("test", "") or "").strip():
        return None
    return evaluate_solution(candidate, reference, timeout=timeout).reward
