"""Regression: ``pr_eval._compute_novelty`` must hand the policy back on the SUBMITTER's θ.

``_compute_novelty`` loads the reigning king's head + SVF into the policy to collect the
king's routing decisions. That policy is the SHARED object the benchmark loop scores with,
and novelty is computed only for ``benchmarks[0]`` — so leaving the king's weights loaded
silently scored every LATER benchmark as the king. With
``COMPETITION_BENCHMARKS = (math500, mmlu, livecodebench)`` that is two thirds of the
composite: ``hidden_acc`` / ``audit_acc`` / ``live_acc`` for mmlu and livecodebench became
the king's numbers, the overfit gate compared king-vs-king, and no miner could ever displace
the king on those two boards.

These tests pin the invariant with a fake policy that records every ``configure`` call — no
torch, no encoder, no network.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pr_eval", _REPO / "scripts" / "pr_eval.py")
pr_eval = importlib.util.module_from_spec(_spec)
sys.modules["pr_eval"] = pr_eval
_spec.loader.exec_module(pr_eval)

from trinity.novelty import NEUTRAL_NOVELTY  # noqa: E402  (after the script import)

_HEAD_SHAPE = (6, 1024)
_SVF_SIZE = 7 * 1024
_SUBMITTER_THETA = np.zeros(_HEAD_SHAPE[0] * _HEAD_SHAPE[1] + _SVF_SIZE)
_KING_VALUE = 7.0


class _FakePolicy:
    """Records every θ it is configured with; starts on the submitter's."""

    def __init__(self):
        self.configured = [_SUBMITTER_THETA.copy()]

    def configure(self, theta, spec):
        self.configured.append(np.asarray(theta, dtype=np.float64).copy())

    @property
    def current(self):
        return self.configured[-1]

    def on_submitter(self) -> bool:
        return bool(np.allclose(self.current, 0.0))

    def saw_king(self) -> bool:
        return any(np.allclose(t, _KING_VALUE) for t in self.configured)


def _king_dir(tmp: Path) -> Path:
    np.save(tmp / "head_weights.npy", np.full(_HEAD_SHAPE, _KING_VALUE, dtype=np.float32))
    np.save(tmp / "svf_scales.npy", np.full(_SVF_SIZE, _KING_VALUE, dtype=np.float32))
    return tmp


@pytest.fixture
def king(tmp_path, monkeypatch):
    """A reigning king on disk, with the leaderboard/routing plumbing stubbed out."""
    monkeypatch.setattr(pr_eval, "_load_leaderboard", lambda: {})
    monkeypatch.setattr(pr_eval, "_king_submission_dir", lambda b, lb, root: _king_dir(tmp_path))
    monkeypatch.setattr(pr_eval, "_routing_decisions",
                        lambda policy, items, ref_count: [0] * ref_count)
    return tmp_path


def _items(n=5):
    return [{"question_text": "q", "correct_answer": "a"} for _ in range(n)]


def test_policy_is_returned_on_the_submitters_theta(king):
    """THE regression: after novelty the loop must still be scoring the submitter."""
    p = _FakePolicy()
    pr_eval._compute_novelty("math500", p, object(), _items(), _SUBMITTER_THETA)
    assert p.on_submitter(), "later benchmarks would be scored with the king's head"


def test_the_king_is_still_loaded_mid_flight(king):
    """The fix must not defeat the comparison it exists to make."""
    p = _FakePolicy()
    pr_eval._compute_novelty("math500", p, object(), _items(), _SUBMITTER_THETA)
    assert p.saw_king(), "novelty must still measure against the king's routing"
    assert p.on_submitter()


def test_theta_is_restored_even_when_the_comparison_raises(king, monkeypatch):
    """A mid-comparison failure must not leak the king's weights into the caller."""
    calls = {"n": 0}

    def flaky(policy, items, ref_count):
        calls["n"] += 1
        if calls["n"] == 2:          # the king's decisions
            raise RuntimeError("routing failed")
        return [0] * ref_count

    monkeypatch.setattr(pr_eval, "_routing_decisions", flaky)
    p = _FakePolicy()
    with pytest.raises(RuntimeError):
        pr_eval._compute_novelty("math500", p, object(), _items(), _SUBMITTER_THETA)
    assert p.on_submitter()


def test_no_king_returns_neutral_and_never_touches_theta(tmp_path, monkeypatch):
    monkeypatch.setattr(pr_eval, "_load_leaderboard", lambda: {})
    monkeypatch.setattr(pr_eval, "_king_submission_dir", lambda b, lb, root: None)
    p = _FakePolicy()
    got = pr_eval._compute_novelty("math500", p, object(), _items(), _SUBMITTER_THETA)
    assert got == NEUTRAL_NOVELTY
    assert len(p.configured) == 1 and p.on_submitter()


def test_no_eval_items_returns_neutral_and_never_touches_theta(king):
    p = _FakePolicy()
    got = pr_eval._compute_novelty("math500", p, object(), [], _SUBMITTER_THETA)
    assert got == NEUTRAL_NOVELTY
    assert len(p.configured) == 1 and p.on_submitter()


def test_unreadable_king_weights_return_neutral_and_never_touch_theta(tmp_path, monkeypatch):
    (tmp_path / "head_weights.npy").write_text("not a numpy file")
    (tmp_path / "svf_scales.npy").write_text("not a numpy file")
    monkeypatch.setattr(pr_eval, "_load_leaderboard", lambda: {})
    monkeypatch.setattr(pr_eval, "_king_submission_dir", lambda b, lb, root: tmp_path)
    p = _FakePolicy()
    got = pr_eval._compute_novelty("math500", p, object(), _items(), _SUBMITTER_THETA)
    assert got == NEUTRAL_NOVELTY
    assert len(p.configured) == 1 and p.on_submitter()


def test_signature_requires_the_submitter_theta():
    """The θ to restore is explicit, so a future caller cannot silently reintroduce this."""
    import inspect

    params = list(inspect.signature(pr_eval._compute_novelty).parameters)
    assert params[-1] == "submitter_theta"


def test_three_benchmark_loop_scores_every_bench_with_the_submitter(king, monkeypatch):
    """End-to-end shape of the bug: novelty runs on bench 0, benches 1-2 must be unaffected."""
    seen: list[float] = []

    def score(policy):
        seen.append(float(policy.current[0]))

    p = _FakePolicy()
    for i, bench in enumerate(("math500", "mmlu", "livecodebench")):
        score(p)                                    # what this benchmark is scored with
        if i == 0:
            pr_eval._compute_novelty(bench, p, object(), _items(), _SUBMITTER_THETA)
    assert seen == [0.0, 0.0, 0.0], f"a benchmark was scored with the king ({seen})"
