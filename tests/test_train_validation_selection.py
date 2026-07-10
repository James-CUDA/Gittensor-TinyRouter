"""Offline tests for validation-based model selection (issue #172).

Covers the two pure pieces the training loop composes: the train/val split
(disjoint, deterministic, legacy-preserving) and the ValidationSelector
(argmax-val selection + early stopping). No GPU/API: the CMA-ES loop's
per-generation fitness is stubbed as a plain sequence.
"""
from __future__ import annotations

import random

import numpy as np

from trinity.optim.selection import ValidationSelector
from trinity.orchestration.dataset import split_train_val
from trinity.types import Task


def _tasks(n: int) -> list[Task]:
    return [Task(task_id=str(i), benchmark="math500", prompt=f"q{i}", answer="1") for i in range(n)]


# ---- split_train_val ----


def test_val_fraction_zero_is_legacy_no_holdout():
    tasks = _tasks(10)
    train, val = split_train_val(tasks, 0.0, random.Random(0))
    assert val == []
    assert train == tasks               # same tasks, same order (byte-identical path)


def test_split_is_disjoint_and_covers_all_tasks():
    tasks = _tasks(20)
    train, val = split_train_val(tasks, 0.25, random.Random(1))
    assert len(val) == 5 and len(train) == 15
    train_ids = {t.task_id for t in train}
    val_ids = {t.task_id for t in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {t.task_id for t in tasks}


def test_split_is_reproducible_under_a_fixed_seed():
    tasks = _tasks(20)
    a_tr, a_val = split_train_val(tasks, 0.3, random.Random(42))
    b_tr, b_val = split_train_val(tasks, 0.3, random.Random(42))
    assert [t.task_id for t in a_val] == [t.task_id for t in b_val]
    assert [t.task_id for t in a_tr] == [t.task_id for t in b_tr]
    # A different seed generally yields a different holdout.
    _, c_val = split_train_val(tasks, 0.3, random.Random(43))
    assert [t.task_id for t in a_val] != [t.task_id for t in c_val]


def test_split_keeps_at_least_one_train_and_one_val():
    # A tiny positive fraction still holds out exactly one task.
    train, val = split_train_val(_tasks(5), 0.01, random.Random(0))
    assert len(val) == 1 and len(train) == 4
    # Fewer than 2 tasks: nothing to hold out.
    assert split_train_val(_tasks(1), 0.5, random.Random(0)) == (_tasks(1), [])


def test_val_fraction_out_of_range_raises():
    import pytest

    with pytest.raises(ValueError):
        split_train_val(_tasks(4), 1.0, random.Random(0))


# ---- ValidationSelector ----


def test_selector_picks_the_argmax_val_generation():
    sel = ValidationSelector()
    val_curve = [0.40, 0.55, 0.52, 0.61, 0.58]   # best is gen 3
    for gen, vf in enumerate(val_curve):
        sel.update(gen, np.full(3, float(gen)), vf)
    assert sel.best_gen == 3
    assert sel.best_val_fitness == 0.61
    assert np.array_equal(sel.best_theta, np.full(3, 3.0))


def test_selector_keeps_first_on_ties_and_copies_theta():
    sel = ValidationSelector()
    theta0 = np.zeros(3)
    sel.update(0, theta0, 0.5)
    sel.update(1, np.ones(3), 0.5)   # equal, not strictly better -> keep gen 0
    assert sel.best_gen == 0
    # Stored theta is a copy, not a live reference to the caller's array.
    theta0[:] = 9.0
    assert np.array_equal(sel.best_theta, np.zeros(3))


def test_early_stop_after_patience_stale_generations():
    sel = ValidationSelector()
    sel.update(0, np.zeros(3), 0.60)   # best
    assert not sel.should_stop(2)
    sel.update(1, np.zeros(3), 0.59)   # stale 1
    assert not sel.should_stop(2)
    sel.update(2, np.zeros(3), 0.58)   # stale 2 -> stop
    assert sel.should_stop(2)


def test_patience_zero_never_stops():
    sel = ValidationSelector()
    sel.update(0, np.zeros(3), 0.6)
    for gen in range(1, 6):
        sel.update(gen, np.zeros(3), 0.1)
    assert not sel.should_stop(0)


def test_improvement_resets_the_stale_counter():
    sel = ValidationSelector()
    sel.update(0, np.zeros(3), 0.5)
    sel.update(1, np.zeros(3), 0.4)    # stale 1
    sel.update(2, np.ones(3), 0.7)     # new best -> reset
    assert sel.stale_generations == 0
    assert not sel.should_stop(2)
    assert sel.best_gen == 2


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
