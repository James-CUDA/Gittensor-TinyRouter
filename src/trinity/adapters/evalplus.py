"""EvalPlus benchmark adapters (HumanEval+ / MBPP+) with a base+plus test schema.

EvalPlus augments the classic HumanEval and MBPP code-generation benchmarks with a
much larger *plus* test set that catches solutions which pass the sparse original
(*base*) tests but are actually wrong. The model is shown a function signature and
docstring and must return a complete solution, graded by running the base and then
the plus assertions against it. That harness is richer than the stdin/stdout or
single-``fn_name`` cases :func:`trinity.orchestration.reward.run_pass_at_1` handles,
so EvalPlus gets the same adapter-plus-dedicated-runner split the repo uses for
SWE-bench and BigCodeBench (issue #254):

* :class:`EvalPlusReference` — the structured ``reference`` for an EvalPlus task
  (entry point, prompt, canonical solution, and the base/plus check harnesses), with
  dict (de)serialization and validation.
* :func:`build_evalplus_prompt` — the prompt format for a completion task.
* :func:`load_evalplus_tasks` — a lazy/guarded HuggingFace loader
  (``evalplus/humanevalplus`` / ``evalplus/mbppplus``) with an offline toy fallback.
* :class:`EvalPlusAdapter` — task type :data:`TaskType.CODE`, one instance per
  dataset (``humaneval_plus`` / ``mbpp_plus``, with ``humaneval`` / ``mbpp`` aliases).

**Scope note.** Executing untrusted model code is opt-in, exactly like SWE-bench's
``repo_provider`` and BigCodeBench's ``runner``: :class:`EvalPlusAdapter` takes an
optional ``runner`` callable ``(output, reference) -> float | None`` (e.g.
:func:`trinity.adapters.evalplus_runner.score_evalplus`). When set,
:meth:`score_output` grades through the sandboxed runner; when ``None`` (the default)
it uses a conservative **exact normalized-solution match** against the canonical
solution, which never reports a wrong solution as correct — so the adapter stays
offline and code-execution-free out of the box.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from trinity.types import Task

from .base import BenchmarkAdapter, ScoringMode, TaskType
from .registry import register_adapter

__all__ = [
    "HUMANEVAL_PLUS",
    "MBPP_PLUS",
    "DATASETS",
    "EvalPlusReference",
    "build_evalplus_prompt",
    "load_evalplus_tasks",
    "extract_solution_code",
    "normalize_code",
    "score_solution_exact",
    "EvalPlusAdapter",
    "register_evalplus_adapters",
]

#: Canonical benchmark names + their aliases and HuggingFace dataset ids.
HUMANEVAL_PLUS = "humaneval_plus"
MBPP_PLUS = "mbpp_plus"

#: ``name -> (hf_dataset, aliases)``. The aliases resolve to the same adapter.
DATASETS: dict[str, tuple[str, tuple[str, ...]]] = {
    HUMANEVAL_PLUS: ("evalplus/humanevalplus", ("humaneval", "humaneval+")),
    MBPP_PLUS: ("evalplus/mbppplus", ("mbpp", "mbpp+")),
}


# --------------------------------------------------------------------------- #
# Structured reference schema
# --------------------------------------------------------------------------- #
@dataclass
class EvalPlusReference:
    """The structured ``reference`` for an EvalPlus task (stored as ``Task.answer``).

    ``plus_test`` / ``base_test`` are each a ``check(candidate)`` harness: a snippet
    defining ``def check(candidate): ...`` whose assertions fail (raise) on a wrong
    solution. The runner calls ``check(<entry_point>)``. A task is *resolved* only if
    the **plus** harness passes (EvalPlus's rigorous rule); ``base_test`` is optional
    and, when present, lets the runner report base vs plus separately.
    """

    entry_point: str
    prompt: str = ""
    canonical_solution: str = ""
    plus_test: str = ""
    base_test: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (the on-disk / hidden-benchmark form)."""
        return {
            "entry_point": self.entry_point,
            "prompt": self.prompt,
            "canonical_solution": self.canonical_solution,
            "plus_test": self.plus_test,
            "base_test": self.base_test,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalPlusReference":
        """Rebuild an :class:`EvalPlusReference` from :meth:`to_dict` output."""
        return cls(
            entry_point=str(data.get("entry_point", "")),
            prompt=str(data.get("prompt", "")),
            canonical_solution=str(data.get("canonical_solution", "")),
            plus_test=str(data.get("plus_test", "")),
            base_test=str(data.get("base_test", "")),
        )

    def is_valid(self) -> bool:
        """A reference is gradable iff it names an entry point and a plus harness."""
        return bool(self.entry_point and self.plus_test.strip())


# --------------------------------------------------------------------------- #
# Prompt format
# --------------------------------------------------------------------------- #
def build_evalplus_prompt(task_prompt: str, entry_point: str) -> str:
    """Render the EvalPlus prompt shown to a pool model.

    ``task_prompt`` is the dataset's completion prompt (imports + signature +
    docstring). We append an explicit response instruction so the (fenced) output is
    a full, importable module the runner can execute.
    """
    parts = [
        task_prompt.strip(),
        "",
        "## Response format",
        "Return ONLY a complete, self-contained Python solution in a ```python code "
        f"block, including all imports and the full definition of `{entry_point}`. "
        "Do not include explanations, tests, or example calls.",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Solution extraction + normalization + placeholder scoring
# --------------------------------------------------------------------------- #
_FENCE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_LEADING_FENCE = re.compile(r"^```[a-zA-Z]*\n|\n```$", re.MULTILINE)
_CODE_MARKERS = ("def ", "class ", "import ", "return ", "print(")


def extract_solution_code(text: str) -> str:
    """Pull the Python solution out of a (possibly chatty) model output.

    Prefers the last fenced ```` ```python ```` block. With no fence, returns the
    stray-fence-stripped text only when it looks like Python source (so chatty prose
    yields ``""``, which the runner reports as ``no_code_found``). Empty for empty
    input.
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

    Strips markdown fences, drops blank lines, and right-strips each line — only
    formatting noise, never anything that changes what the code does.
    """
    if not code:
        return ""
    text = _LEADING_FENCE.sub("", code)
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip())


def score_solution_exact(candidate: str, reference: Any) -> float:
    """Placeholder scorer: exact normalized match against the canonical solution.

    Correctness is exact normalized-code equality with the reference solution. This
    can only *under*-credit (a correct-but-different solution scores 0); it never
    credits a wrong solution, so it is safe to report offline.
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


def _hf_evalplus(hf_dataset: str, benchmark: str, split: str) -> list[Task] | None:
    """Load an EvalPlus dataset from HuggingFace, or ``None`` on any failure."""
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        ds = load_dataset(hf_dataset, split=split or "test")
    except Exception:
        return None

    tasks: list[Task] = []
    for i, row in enumerate(ds):
        task_id = str(_row_get(row, "task_id", default=f"{benchmark}-{i}"))
        prompt = _row_get(row, "prompt", "text", default="")
        entry_point = _row_get(row, "entry_point", default="")
        # EvalPlus rows carry the plus harness under `test`; some mirrors also expose
        # a smaller `base_test`/`test_list`. Keep both when present.
        plus_test = _row_get(row, "test", "plus_test", default="")
        base_test = _row_get(row, "base_test", default="")
        if not prompt or not entry_point or not str(plus_test).strip():
            continue
        ref = EvalPlusReference(
            entry_point=str(entry_point),
            prompt=str(prompt),
            canonical_solution=str(_row_get(row, "canonical_solution", default="")),
            plus_test=str(plus_test),
            base_test=str(base_test),
        )
        tasks.append(_make_task(benchmark, task_id, ref))
    return tasks or None


def _make_task(benchmark: str, task_id: str, ref: EvalPlusReference) -> Task:
    """Normalise one EvalPlus instance into a :class:`Task`."""
    return Task(
        task_id=task_id,
        benchmark=benchmark,
        prompt=build_evalplus_prompt(ref.prompt, ref.entry_point),
        answer=ref.to_dict(),
        meta={
            "source": DATASETS[benchmark][0],
            "entry_point": ref.entry_point,
            "task_type": TaskType.CODE.value,
        },
    )


def _toy_evalplus(benchmark: str) -> list[Task]:
    """A tiny, self-contained EvalPlus-style task so smoke tests need no network.

    The base check only exercises a non-negative input; the plus check adds the
    negative-input edge the base misses — so a solution that special-cases negatives
    passes base but fails plus, exactly the case EvalPlus exists to catch.
    """
    ref = EvalPlusReference(
        entry_point="add_one",
        prompt='def add_one(x):\n    """Return x + 1."""\n',
        canonical_solution="def add_one(x):\n    return x + 1\n",
        base_test="def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(0) == 1\n",
        plus_test=(
            "def check(candidate):\n"
            "    assert candidate(1) == 2\n"
            "    assert candidate(0) == 1\n"
            "    assert candidate(-5) == -4\n"
        ),
    )
    return [_make_task(benchmark, f"{benchmark}-toy-0", ref)]


def load_evalplus_tasks(benchmark: str, split: str, max_items: int | None, seed: int = 0) -> list[Task]:
    """Load an EvalPlus benchmark as a deterministic list of :class:`Task`.

    Tries HuggingFace (lazy/guarded); on any failure falls back to the built-in toy
    set. Applies a ``seed``-seeded shuffle and truncates to ``max_items``, so repeated
    calls with identical arguments return identical lists.
    """
    import random

    if benchmark not in DATASETS:
        raise ValueError(f"Unknown EvalPlus benchmark {benchmark!r}; known: {sorted(DATASETS)}")
    hf_dataset = DATASETS[benchmark][0]
    tasks = _hf_evalplus(hf_dataset, benchmark, split) or _toy_evalplus(benchmark)
    tasks = list(tasks)
    random.Random(seed).shuffle(tasks)
    if max_items is not None:
        tasks = tasks[: max(0, int(max_items))]
    return tasks


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class EvalPlusAdapter(BenchmarkAdapter):
    """An EvalPlus benchmark (HumanEval+ or MBPP+): prompt in, full solution out.

    One instance per dataset (``name`` is ``humaneval_plus`` or ``mbpp_plus``).
    ``runner`` opts into real execution: a callable ``(output, reference) -> float | None``
    that grades against the base+plus harness in a sandbox (e.g.
    :func:`trinity.adapters.evalplus_runner.score_evalplus`). When set,
    :meth:`score_output` grades through it; when ``None`` (the default) it uses the
    cheap exact-match placeholder, so the adapter never executes model code out of
    the box.
    """

    def __init__(
        self,
        name: str,
        *,
        runner: Optional[Callable[[str, Any], Optional[float]]] = None,
    ):
        if name not in DATASETS:
            raise ValueError(f"Unknown EvalPlus benchmark {name!r}; known: {sorted(DATASETS)}")
        self.name = name
        self._runner = runner

    def load_tasks(self, split: str, max_items: int | None, seed: int = 0) -> list[Task]:
        return load_evalplus_tasks(self.name, split, max_items, seed=seed)

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def score_output(self, output: str, reference: Any) -> float:
        if self._runner is None:
            return score_solution_exact(output, reference)
        result = self._runner(output, reference)
        return 0.0 if result is None else float(result)

    def scoring_modes(self) -> frozenset[ScoringMode]:
        return frozenset({ScoringMode.CACHED, ScoringMode.EXECUTION})

    def score_execution(self, output: str, reference: Any, *, context: Any = None) -> float | None:
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


def register_evalplus_adapters() -> None:
    """Register the EvalPlus adapters under their names + aliases (idempotent-friendly)."""
    from .registry import is_registered

    for name, (_hf, aliases) in DATASETS.items():
        adapter = EvalPlusAdapter(name)
        for key in (name, *aliases):
            if not is_registered(key):
                register_adapter(key, adapter)
