"""Torch-path check for the seeded train-time sampling hook (issue #130).

This file imports torch (inside a fixture), which pollutes ``sys.modules`` for the
rest of the process, and ``test_shaped_fitness.py::test_no_torch_imported`` asserts
torch stays out of ``sys.modules``. So — like ``test_torch_coordinator_head.py`` —
this must be a ``test_torch_*`` module, which sorts AFTER the torch-free canaries;
never import torch from a test file that sorts before them.

The seed-plumbing itself (rng threading, order-invariance, sha256 derivation) is
covered torch-free in ``test_fitness_sampling_seed.py``. This adds the one check
that needs real torch: the ``rng`` hook the fix now feeds is reproducible.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def torch():
    return pytest.importorskip("torch", reason="torch required for LinearHead sampling")


def test_head_select_generator_hook_is_reproducible(torch):
    """Same-seeded generators -> identical (agent, role) draws; different seed differs."""
    from trinity.coordinator.head import LinearHead

    head = LinearHead(n_a=6, d_h=4, n_models=3)   # zero-init -> uniform categoricals
    h = torch.ones(4)

    def draw(gen):
        return [head.select(h, sample=True, rng=gen)[:2] for _ in range(20)]

    assert draw(torch.Generator().manual_seed(123)) == draw(torch.Generator().manual_seed(123))
    assert draw(torch.Generator().manual_seed(123)) != draw(torch.Generator().manual_seed(124))


def test_coordinator_policy_make_rng_is_reproducible(torch):
    """CoordinatorPolicy.make_rng builds a working, reproducible generator.

    Only the head's device is read, so no encoder/SVF is needed.
    """
    from trinity.coordinator.head import LinearHead
    from trinity.coordinator.policy import CoordinatorPolicy

    head = LinearHead(n_a=6, d_h=4, n_models=3)
    h = torch.ones(4)
    policy = CoordinatorPolicy(encoder=None, svf=None, head=head, n_models=3)

    def draw(gen):
        return [head.select(h, sample=True, rng=gen)[:2] for _ in range(20)]

    assert draw(policy.make_rng(999)) == draw(policy.make_rng(999))
    assert draw(policy.make_rng(1)) != draw(policy.make_rng(2))
