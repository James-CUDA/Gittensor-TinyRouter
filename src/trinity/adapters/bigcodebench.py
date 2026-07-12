"""BigCodeBench benchmark adapter with a structured, unittest-harness schema.

BigCodeBench is a library-heavy, function-call style code benchmark: the model is
shown a task with imports and a function signature/docstring and must return a
complete Python solution, which is graded by running the task's
``unittest.TestCase`` harness against it. That harness is richer than the
stdin/stdout or single-``fn_name`` cases :func:`trinity.orchestration.reward.run_pass_at_1`
grades for LiveCodeBench, so BigCodeBench gets the same adapter-plus-dedicated-runner
split the repo uses for SWE-bench (issue #212):

* :class:`BigCodeBenchReference` — the structured ``reference`` for a BigCodeBench
  task (entry point, unittest source, canonical solution, code prompt, libs), with
  dict (de)serialization and validation.
* :func:`build_bigcodebench_prompt` — the prompt format for a BigCodeBench task.
* :func:`load_bigcodebench_tasks` — a lazy/guarded HuggingFace loader
  (``bigcode/bigcodebench``) with an offline toy fallback, returning normalized
  :class:`~trinity.types.Task` objects carrying entry-point/libs metadata.
* :class:`BigCodeBenchAdapter` — task type :data:`TaskType.CODE`.

**Execution.** Like LiveCodeBench (which runs code via ``reward.run_pass_at_1``), the
**registered** adapter grades by *executing* the solution against the task's harness in
an isolated subprocess (:mod:`trinity.adapters.bigcodebench_runner`): ``python -I``, a
scrubbed environment that hides host secrets such as ``OPENROUTER_API_KEY``, a throwaway
work-tree, a wall-clock timeout, and POSIX resource caps. The executor is a pluggable
seam — the adapter's ``runner`` argument — so a deployment needing a hard jail
(container/nsjail) injects its own. ``BigCodeBenchAdapter(runner=None)`` keeps a
conservative **exact normalized-solution match** (no execution) for unit tests and any
caller that must not run model code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from trinity.types import Task

from .base import BenchmarkAdapter, ScoringMode, TaskType
from .registry import register_adapter

__all__ = [
    "BENCHMARK",
    "ALIASES",
    "BigCodeBenchReference",
    "build_bigcodebench_prompt",
    "load_bigcodebench_tasks",
    "extract_solution_code",
    "normalize_code",
    "score_solution_exact",
    "BigCodeBenchAdapter",
    "register_bigcodebench_adapter",
]

#: Canonical benchmark name this adapter registers under.
BENCHMARK = "bigcodebench"

#: Additional names that resolve to the same adapter (``reward.CODE_BENCHMARKS``
#: already recognises ``bigcode`` for scoring).
ALIASES = ("bigcode",)

#: HuggingFace dataset id for BigCodeBench.
_HF_DATASET = "bigcode/bigcodebench"

#: Default config/split to request from HuggingFace. The loader is guarded, so a
#: wrong/renamed split just falls back to the offline toy set rather than raising.
_DEFAULT_HF_SPLIT = "v0.1.0_hf"


# --------------------------------------------------------------------------- #
# Structured reference schema
# --------------------------------------------------------------------------- #
@dataclass
class BigCodeBenchReference:
    """The structured ``reference`` for a BigCodeBench task (stored as ``Task.answer``).

    Carries everything the sandboxed runner needs to grade a candidate solution
    (the ``unittest`` harness and the entry point it exercises) plus the canonical
    solution the offline placeholder compares against.
    """

    entry_point: str
    test: str
    canonical_solution: str = ""
    code_prompt: str = ""
    libs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (the on-disk / hidden-benchmark form)."""
        return {
            "entry_point": self.entry_point,
            "test": self.test,
            "canonical_solution": self.canonical_solution,
            "code_prompt": self.code_prompt,
            "libs": list(self.libs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BigCodeBenchReference":
        """Rebuild a :class:`BigCodeBenchReference` from :meth:`to_dict` output."""
        return cls(
            entry_point=str(data.get("entry_point", "")),
            test=str(data.get("test", "")),
            canonical_solution=str(data.get("canonical_solution", "")),
            code_prompt=str(data.get("code_prompt", "")),
            libs=list(data.get("libs", []) or []),
        )

    def is_valid(self) -> bool:
        """A reference is gradable iff it carries a unittest harness."""
        return bool(self.test.strip())


def _as_str_list(value: Any) -> list[str]:
    """Coerce a BigCodeBench ``libs`` field (JSON string or list) to ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, str):
        import json

        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


# --------------------------------------------------------------------------- #
# Prompt format
# --------------------------------------------------------------------------- #
def build_bigcodebench_prompt(task_prompt: str, entry_point: str) -> str:
    """Render the BigCodeBench prompt shown to a pool model.

    ``task_prompt`` is the dataset's complete/instruct prompt (imports + signature
    + docstring). We append an explicit response instruction so the (fenced) output
    is a full, importable module the runner can execute.
    """
    parts = [
        task_prompt.strip(),
        "",
        "## Response format",
        "Return ONLY a complete, self-contained Python solution in a ```python "
        f"code block, including all imports and the full definition of `{entry_point}`. "
        "Do not include explanations, tests, or example calls.",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Solution extraction + normalization + placeholder scoring (real run is the runner)
# --------------------------------------------------------------------------- #
_FENCE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_LEADING_FENCE = re.compile(r"^```[a-zA-Z]*\n|\n```$", re.MULTILINE)


#: Cheap "looks like Python source" markers, matching ``reward.has_answer``'s code
#: heuristic, used only when the output has no fenced block.
_CODE_MARKERS = ("def ", "class ", "import ", "return ", "print(")


def extract_solution_code(text: str) -> str:
    """Pull the Python solution out of a (possibly chatty) model output.

    Prefers the last fenced ```` ```python ```` block. With no fence, returns the
    stray-fence-stripped text only when it looks like Python source (so chatty prose
    yields ``""``, which the runner reports as ``no_code_found`` rather than a
    compile error). Returns an empty string for empty input.
    """
    if not text:
        return ""
    blocks = _FENCE_BLOCK.findall(text)
    if blocks:
        return blocks[-1].strip("\n") + "\n"
    stripped = _LEADING_FENCE.sub("", text).strip("\n")
    if stripped and any(marker in stripped for marker in _CODE_MARKERS):
        return stripped + "\n"
    return ""


def normalize_code(code: str) -> str:
    """Normalise Python source for tolerant exact comparison.

    Strips markdown fences, drops blank lines, and right-strips each line. This is
    deliberately conservative — it only removes formatting noise, never anything
    that changes what the code does.
    """
    if not code:
        return ""
    text = _LEADING_FENCE.sub("", code)
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip())


def score_solution_exact(candidate: str, reference: Any) -> float:
    """Placeholder scorer: exact normalized match against the canonical solution.

    Until (or unless) a sandboxed runner is injected, correctness is exact
    normalized-code equality with the reference solution. This can only
    *under*-credit (a correct-but-different solution scores 0); it never credits a
    wrong solution, so it is safe to report offline.
    """
    ref = reference if isinstance(reference, dict) else {}
    gold = str(ref.get("canonical_solution", "")) if ref else ""
    if not gold.strip():
        return 0.0
    return 1.0 if normalize_code(extract_solution_code(candidate)) == normalize_code(gold) else 0.0


# --------------------------------------------------------------------------- #
# Loader (HuggingFace + offline toy fallback)
# --------------------------------------------------------------------------- #
def _row_get(row: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        try:
            if k in row and row[k] is not None:
                return row[k]
        except TypeError:
            break
    return default


def _hf_bigcodebench(split: str) -> list[Task] | None:
    """Load BigCodeBench from HuggingFace, or ``None`` on any failure."""
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        ds = load_dataset(_HF_DATASET, split=split or _DEFAULT_HF_SPLIT)
    except Exception:
        return None

    tasks: list[Task] = []
    for i, row in enumerate(ds):
        task_id = str(_row_get(row, "task_id", default=f"bigcodebench-{i}"))
        prompt = _row_get(row, "complete_prompt", "instruct_prompt", "prompt", default="")
        test = _row_get(row, "test", default="")
        entry_point = _row_get(row, "entry_point", default="")
        if not prompt or not test:
            continue
        ref = BigCodeBenchReference(
            entry_point=str(entry_point),
            test=str(test),
            canonical_solution=str(_row_get(row, "canonical_solution", default="")),
            code_prompt=str(_row_get(row, "code_prompt", default="")),
            libs=_as_str_list(_row_get(row, "libs")),
        )
        tasks.append(_make_task(task_id, str(prompt), ref))
    return tasks or None


def _make_task(task_id: str, task_prompt: str, ref: BigCodeBenchReference) -> Task:
    """Normalise one BigCodeBench instance into a :class:`Task`."""
    return Task(
        task_id=task_id,
        benchmark=BENCHMARK,
        prompt=build_bigcodebench_prompt(task_prompt, ref.entry_point),
        answer=ref.to_dict(),
        meta={
            "source": _HF_DATASET,
            "entry_point": ref.entry_point,
            "libs": list(ref.libs),
            "task_type": TaskType.CODE.value,
        },
    )


def _toy_bigcodebench() -> list[Task]:
    """Tiny, self-contained BigCodeBench-style tasks so smoke tests need no network."""
    ref = BigCodeBenchReference(
        entry_point="add_positive",
        test=(
            "import unittest\n\n"
            "class TestCases(unittest.TestCase):\n"
            "    def test_sum(self):\n"
            "        self.assertEqual(add_positive([1, -2, 3]), 4)\n"
            "    def test_empty(self):\n"
            "        self.assertEqual(add_positive([]), 0)\n"
        ),
        canonical_solution=(
            "def add_positive(nums):\n"
            "    return sum(n for n in nums if n > 0)\n"
        ),
        code_prompt="def add_positive(nums):\n",
        libs=[],
    )
    return [
        _make_task(
            "bigcodebench-toy-0",
            (
                "def add_positive(nums):\n"
                '    """Return the sum of the strictly positive numbers in ``nums``."""\n'
            ),
            ref,
        )
    ]


def load_bigcodebench_tasks(split: str, max_items: int | None, seed: int = 0) -> list[Task]:
    """Load BigCodeBench as a deterministic list of :class:`Task`.

    Tries HuggingFace (lazy/guarded); on any failure falls back to the built-in toy
    set. Applies a ``seed``-seeded shuffle and truncates to ``max_items``, so
    repeated calls with identical arguments return identical lists.
    """
    import random

    tasks = _hf_bigcodebench(split) or _toy_bigcodebench()
    tasks = list(tasks)
    random.Random(seed).shuffle(tasks)
    if max_items is not None:
        tasks = tasks[: max(0, int(max_items))]
    return tasks


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class BigCodeBenchAdapter(BenchmarkAdapter):
    """BigCodeBench: library-heavy prompt in, complete Python solution out.

    ``runner`` is the executor seam: a callable ``(output, reference) -> float | None``
    that grades the solution against the task's ``unittest`` harness. The **registered**
    adapter (``register_bigcodebench_adapter``) wires in
    :func:`trinity.adapters.bigcodebench_runner.score_bigcodebench` — the isolated
    subprocess executor (scrubbed env, ``python -I``, resource caps) — so a normal
    ``get_adapter("bigcodebench").score_output(...)`` **actually runs the tests**, the
    same way LiveCodeBench executes via ``reward.run_pass_at_1``. A deployment that
    needs a harder jail (container/nsjail) injects its own ``runner`` here.

    Constructing ``BigCodeBenchAdapter()`` with ``runner=None`` keeps the cheap
    exact-match placeholder (no execution) — used by unit tests and any caller that
    must not run model code.
    """

    name = BENCHMARK

    def __init__(self, *, runner: Optional[Callable[[str, Any], Optional[float]]] = None):
        self._runner = runner

    def load_tasks(self, split: str, max_items: int | None, seed: int = 0) -> list[Task]:
        return load_bigcodebench_tasks(split, max_items, seed=seed)

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def score_output(self, output: str, reference: Any) -> float:
        if self._runner is None:
            return score_solution_exact(output, reference)
        result = self._runner(output, reference)
        return 0.0 if result is None else float(result)

    def scoring_modes(self) -> frozenset[ScoringMode]:
        # Cheap cached exact-match, plus an optional sandboxed unittest run.
        return frozenset({ScoringMode.CACHED, ScoringMode.EXECUTION})

    def score_execution(self, output: str, reference: Any, *, context: Any = None) -> float | None:
        """Grade a solution by live execution when an executor is supplied.

        ``context`` is an executor callable ``(output, reference) -> float | None``
        (e.g. one built on the sandboxed unittest runner). Without one, execution is
        unavailable here and this returns ``None`` so the dispatcher falls back to
        the cached exact-match path.
        """
        executor = context if callable(context) else self._runner
        if executor is None:
            return None
        result = executor(output, reference)
        return None if result is None else float(result)

    def task_type(self) -> TaskType:
        return TaskType.CODE

    def serialize_task(self, task: Task) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "prompt": task.prompt,
            "reference": task.answer,
            "task_type": TaskType.CODE.value,
            "meta": dict(task.meta),
        }


def register_bigcodebench_adapter() -> None:
    """Register the BigCodeBench adapter under its name and aliases (idempotent-friendly).

    The registered adapter is wired to the sandboxed executor
    (:func:`trinity.adapters.bigcodebench_runner.score_bigcodebench`), so evaluation
    through ``get_adapter("bigcodebench")`` runs the tests rather than falling back to
    exact source matching.
    """
    from .bigcodebench_runner import score_bigcodebench
    from .registry import is_registered

    adapter = BigCodeBenchAdapter(runner=score_bigcodebench)
    for name in (BENCHMARK, *ALIASES):
        if not is_registered(name):
            register_adapter(name, adapter)
