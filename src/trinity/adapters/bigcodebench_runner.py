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

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

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

#: Default CPU-seconds and output-file-size caps for the child (POSIX only).
_CPU_SECONDS = 60
_MAX_FILE_BYTES = 64 * 1024 * 1024

#: Environment variables that are safe to forward to the untrusted child. Anything
#: NOT listed here — crucially every secret, e.g. ``OPENROUTER_API_KEY``,
#: ``FIREWORKS_API_KEY``, ``BENCHMARK_PASSWORD`` — is dropped, so the graded code
#: cannot read (or exfiltrate) a credential from the host environment.
_ENV_ALLOWLIST = (
    "PATH", "HOME", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP", "TMPDIR",
    "LANG", "LC_ALL", "LC_CTYPE",
)


def _sandbox_env() -> dict[str, str]:
    """A minimal, secret-free environment for the untrusted child interpreter."""
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    for key in _ENV_ALLOWLIST:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _resource_limiter() -> Optional[Callable[[], None]]:
    """A POSIX ``preexec_fn`` that caps CPU time, output size, and core dumps.

    Returns ``None`` on platforms without :mod:`resource` (e.g. Windows), where the
    wall-clock timeout and scrubbed environment are the isolation. Belt-and-braces
    with the timeout: ``RLIMIT_CPU`` bounds CPU burn even if the process forks.
    """
    try:
        import resource
    except ImportError:
        return None

    def _apply() -> None:  # pragma: no cover - runs only in the child process
        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS))
        resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_FILE_BYTES, _MAX_FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return _apply


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
    """Write ``source`` to a temp module and run it in an isolated subprocess.

    Returns ``(all_passed, detail)``. ``all_passed`` is ``True`` only if the process
    exits ``0`` (every unittest assertion passed). The untrusted code is **never**
    ``exec``'d in-process; it runs in a fresh interpreter that is isolated as far as
    a pure-Python, cross-platform executor can be — matching the repo's existing
    ``reward.run_pass_at_1`` sandbox:

    * ``python -I`` (isolated mode): ignores ``PYTHON*`` env vars, ``PYTHONPATH``,
      and the user site-packages dir, so the child can't be steered via the
      environment or a planted module.
    * a **scrubbed environment** (:func:`_sandbox_env`): no host secrets
      (``OPENROUTER_API_KEY`` etc.) are visible to the graded code.
    * a throwaway working directory and a wall-clock ``timeout``.
    * POSIX resource caps (:func:`_resource_limiter`): CPU-seconds, output file
      size, and no core dumps.

    This is defense-in-depth, not a kernel jail: network egress and the wider
    read-only filesystem are not blocked here. A deployment that needs a hard jail
    injects its own executor via the adapter's ``runner`` seam (e.g. a
    container/nsjail-backed ``score`` callable); this is the safe in-repo default.
    """
    with tempfile.TemporaryDirectory(prefix="bigcodebench-") as tmp:
        script = Path(tmp) / "harness.py"
        script.write_text(source, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(script)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_sandbox_env(),
                preexec_fn=_resource_limiter(),   # None on Windows -> no-op
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
