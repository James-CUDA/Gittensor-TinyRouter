"""Logical ``eval``/``validation`` splits must resolve like ``test`` for the
single-upstream-split holdout benchmarks (GPQA, SWE-bench Verified, AIME).

``_TEST_SPLITS`` = {``test``, ``eval``, ``validation``}, so ``select_holdout``
serves *any* of the three the held-out subset. But ``_SPLIT_ALIASES`` previously
aliased only ``test`` (plus ``train``/``training``) onto the single upstream split
for these benchmarks, so logical ``eval``/``validation`` were forwarded verbatim,
the upstream load raised on the non-existent split name, and the loader silently
substituted the 2-item offline toy set -- an "eval" over two hand-written
questions. This is the same toy-fallback failure mode fixed for the ``test`` split
in #35 / #50 / #95 / #196.

Offline: installs a fake ``datasets`` module publishing only the real upstream
split; no network, no GPU. Mirrors ``test_gpqa_holdout_split.py`` and
``test_swebench_holdout_split.py``.
"""
from __future__ import annotations

import sys
import types

import pytest

import trinity.adapters.swebench as sb
from trinity.adapters.loaders import load_split
from trinity.adapters.split_policy import resolve_split

N_ROWS = 40


# --------------------------------------------------------------------------- #
# resolve_split: eval/validation alias onto the same upstream split as test
# --------------------------------------------------------------------------- #
def test_resolve_split_aliases_eval_and_validation_like_test():
    for bench in ("gpqa", "aime"):
        assert resolve_split(bench, "test") == "train"
        assert resolve_split(bench, "eval") == "train"
        assert resolve_split(bench, "validation") == "train"
    # SWE-bench Verified publishes only ``test``; test passes through, so eval/
    # validation must resolve there too.
    assert resolve_split("swebench_verified", "test") == "test"
    assert resolve_split("swebench_verified", "eval") == "test"
    assert resolve_split("swebench_verified", "validation") == "test"


# --------------------------------------------------------------------------- #
# GPQA / AIME (loaded through ``load_split``)
# --------------------------------------------------------------------------- #
def _install_fake_gpqa() -> None:
    def load_dataset(path, name=None, split=None, **kwargs):
        if path != "Idavidrein/gpqa" or name != "gpqa_diamond" or split != "train":
            raise ValueError(f"Unknown split {split!r}.")
        return [
            {
                "Question": f"q-{i}",
                "Correct Answer": f"right-{i}",
                "Incorrect Answer 1": f"wrong-{i}-a",
                "Incorrect Answer 2": f"wrong-{i}-b",
                "Incorrect Answer 3": f"wrong-{i}-c",
            }
            for i in range(N_ROWS)
        ]

    mod = types.ModuleType("datasets")
    mod.load_dataset = load_dataset
    sys.modules["datasets"] = mod


def _install_fake_aime() -> None:
    def load_dataset(path, name=None, split=None, **kwargs):
        if path != "AI-MO/aimo-validation-aime" or split != "train":
            raise ValueError(f"Unknown split {split!r}.")
        return [{"problem": f"p-{i}", "answer": str(i)} for i in range(N_ROWS)]

    mod = types.ModuleType("datasets")
    mod.load_dataset = load_dataset
    sys.modules["datasets"] = mod


@pytest.mark.parametrize("logical", ["eval", "validation"])
@pytest.mark.parametrize(
    "benchmark, install", [("gpqa", _install_fake_gpqa), ("aime", _install_fake_aime)]
)
def test_eval_validation_load_real_holdout_not_toy(benchmark, install, logical):
    install()
    try:
        tasks = load_split(benchmark, logical, max_items=None, seed=0)
        # The held-out rows are exactly the logical ``test`` subset.
        test_tasks = load_split(benchmark, "test", max_items=None, seed=0)
        train_tasks = load_split(benchmark, "train", max_items=None, seed=0)
    finally:
        sys.modules.pop("datasets", None)
    assert tasks, "logical split served no rows"
    # Real upstream rows, not the 2-item offline toy set.
    assert len(tasks) > 2
    assert all(f"{benchmark}-toy-" not in t.task_id for t in tasks)
    ids = {t.task_id for t in tasks}
    # eval/validation get the SAME held-out subset as logical test.
    assert ids == {t.task_id for t in test_tasks}
    # ... and that subset is disjoint from the training remainder, and together
    # they cover the single upstream split.
    train_ids = {t.task_id for t in train_tasks}
    assert ids.isdisjoint(train_ids)
    assert len(ids | train_ids) == N_ROWS


# --------------------------------------------------------------------------- #
# SWE-bench Verified (loaded through ``load_swebench_tasks``)
# --------------------------------------------------------------------------- #
def _install_fake_swebench() -> None:
    def load_dataset(path, name=None, split=None, **kwargs):
        if path != sb._HF_DATASET or split != "test":
            raise ValueError(f"Unknown split {split!r}.")
        return [
            {
                "instance_id": f"inst-{i}",
                "problem_statement": f"issue-{i}",
                "repo": f"org/repo-{i}",
                "base_commit": "0" * 40,
                "patch": "diff --git a/x b/x\n",
            }
            for i in range(N_ROWS)
        ]

    mod = types.ModuleType("datasets")
    mod.load_dataset = load_dataset
    sys.modules["datasets"] = mod


@pytest.mark.parametrize("logical", ["eval", "validation"])
def test_swebench_eval_validation_load_real_holdout_not_toy(logical):
    _install_fake_swebench()
    try:
        tasks = sb.load_swebench_tasks(logical, max_items=None, seed=0)
        test_tasks = sb.load_swebench_tasks("test", max_items=None, seed=0)
    finally:
        sys.modules.pop("datasets", None)
    assert tasks, "logical split served no rows"
    # Real upstream rows, not the one-item SWE-bench toy set.
    assert len(tasks) > 1
    assert all(t.task_id != "octo__calc-1" for t in tasks)
    assert all(t.meta["source"] == sb._HF_DATASET for t in tasks)
    # eval/validation get the SAME held-out subset as logical test.
    assert {t.task_id for t in tasks} == {t.task_id for t in test_tasks}
