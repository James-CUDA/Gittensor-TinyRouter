"""The hidden benchmark must never be built from the offline toy set.

`benchmark_protocol.sample_pool` draws its pool from the ``"train"`` split. The
loaders substitute a 2-3 item toy set whenever HuggingFace is unreachable, a
dataset is gated (``Idavidrein/gpqa``), or a split does not resolve -- emitting a
``ToyFallbackWarning``. A warning is not enough here: the hidden benchmark's
questions are sealed and integrity-hashed, so toy data must abort the build.

Before this guard, ``_sample_pool`` returned the toy tasks and the build died much
later inside ``select_splits`` with ``pool has 2 tasks but the protocol needs 220``
-- an error naming neither the toy fallback nor the split that failed to load.

Offline: no network. ``datasets`` is faked (or removed) per test.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


protocol = _load("benchmark_protocol", "scripts/benchmark_protocol.py")
build_benchmark = _load("build_benchmark", "scripts/build_benchmark.py")


def _install_missing_datasets(monkeypatch):
    """Make ``import datasets`` fail — the offline dev box, or a gated dataset."""
    monkeypatch.setitem(sys.modules, "datasets", None)


def _install_fake_datasets(monkeypatch, n_rows: int = 2000):
    """A working ``datasets`` serving `cais/mmlu`'s real split set."""
    real = {("cais/mmlu", "all"): {"test", "validation", "dev", "auxiliary_train"}}

    def load_dataset(path, name=None, split=None, **kw):
        available = real.get((path, name))
        if available is None:
            raise ValueError(f"Dataset {path!r} is not available.")
        if split not in available:
            raise ValueError(f"Unknown split {split!r}.")
        return [
            {
                "question": f"real question {i}",
                "choices": ["alpha", "beta", "gamma", "delta"],
                "answer": i % 4,
                "subject": "astronomy",
            }
            for i in range(n_rows)
        ]

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #
def test_toy_fallback_aborts_the_build(monkeypatch):
    """The regression: previously this returned 2 toy tasks and limped on."""
    _install_missing_datasets(monkeypatch)
    counts = protocol.split_counts("mmlu")

    with pytest.raises(RuntimeError, match="Refusing to build the hidden benchmark"):
        build_benchmark._sample_pool("mmlu", counts)


def test_abort_message_names_the_benchmark(monkeypatch):
    _install_missing_datasets(monkeypatch)
    counts = protocol.split_counts("gpqa")

    with pytest.raises(RuntimeError, match="'gpqa'"):
        build_benchmark._sample_pool("gpqa", counts)


def test_abort_message_explains_the_remedy(monkeypatch):
    _install_missing_datasets(monkeypatch)
    counts = protocol.split_counts("mmlu")

    with pytest.raises(RuntimeError) as excinfo:
        build_benchmark._sample_pool("mmlu", counts)

    message = str(excinfo.value)
    assert "toy set" in message
    assert "datasets" in message


def test_abort_chains_the_original_warning(monkeypatch):
    """The `ToyFallbackWarning` is preserved as `__cause__`, not swallowed."""
    from trinity.adapters.split_policy import ToyFallbackWarning

    _install_missing_datasets(monkeypatch)
    counts = protocol.split_counts("mmlu")

    with pytest.raises(RuntimeError) as excinfo:
        build_benchmark._sample_pool("mmlu", counts)

    assert isinstance(excinfo.value.__cause__, ToyFallbackWarning)


def test_no_silent_short_pool(monkeypatch):
    """It must raise, not return a pool too small for the protocol."""
    _install_missing_datasets(monkeypatch)
    counts = protocol.split_counts("mmlu")

    with pytest.raises(RuntimeError):
        pool = build_benchmark._sample_pool("mmlu", counts)
        assert len(pool) >= protocol.pool_size(counts), "unreachable: must have raised"


# --------------------------------------------------------------------------- #
# The happy path must be untouched
# --------------------------------------------------------------------------- #
def test_real_data_builds_a_full_pool(monkeypatch):
    _install_fake_datasets(monkeypatch)
    counts = protocol.split_counts("mmlu")

    pool = build_benchmark._sample_pool("mmlu", counts)

    assert len(pool) == protocol.pool_size(counts)
    assert not any(str(t.task_id).startswith("mmlu-toy") for t in pool)


def test_real_data_carves_the_protocol_splits(monkeypatch):
    _install_fake_datasets(monkeypatch)
    counts = protocol.split_counts("mmlu")

    pool = build_benchmark._sample_pool("mmlu", counts)
    splits = protocol.select_splits(pool, counts)

    assert {k: len(v) for k, v in splits.items()} == dict(counts)


def test_the_warning_filter_does_not_leak(monkeypatch):
    """`catch_warnings` must restore the global filters for other callers."""
    import warnings

    from trinity.adapters.split_policy import ToyFallbackWarning

    _install_fake_datasets(monkeypatch)
    build_benchmark._sample_pool("mmlu", protocol.split_counts("mmlu"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("still just a warning", ToyFallbackWarning)

    assert len(caught) == 1
