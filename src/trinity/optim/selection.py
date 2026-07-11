"""Validation-based model selection for CMA-ES training (issue #172).

The training loop scores each generation's incumbent on a held-out validation set
and keeps the one that generalizes best, rather than the candidate with the best
*training* fitness (which is the most prone to fitting the minibatch noise). This
module holds the pure, torch-free selection state so it is unit-testable with a
stubbed fitness sequence (no GPU/API).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ValidationSelector:
    """Tracks the best-on-validation candidate across CMA-ES generations.

    ``update`` is called once per generation with that generation's incumbent
    theta and its validation fitness; the selector keeps the theta with the
    highest validation fitness seen so far (first one wins on ties). ``should_stop``
    reports whether validation has plateaued for ``patience`` generations, so the
    caller can early-stop.

    Attributes:
        best_val_fitness: Highest validation fitness seen (``-inf`` until the
            first :meth:`update`).
        best_theta: A copy of the incumbent theta at ``best_val_fitness``, or
            ``None`` before the first update.
        best_gen: Generation index that produced ``best_theta`` (``-1`` if none).
        stale_generations: Generations since the last strict improvement.
    """

    best_val_fitness: float = -math.inf
    best_theta: np.ndarray | None = None
    best_gen: int = -1
    stale_generations: int = 0

    def update(self, generation: int, theta: np.ndarray, val_fitness: float) -> bool:
        """Record one generation's incumbent and validation fitness.

        Args:
            generation: The generation index (0-based).
            theta: The incumbent parameter vector for this generation.
            val_fitness: Its fitness on the held-out validation set.

        Returns:
            ``True`` if this is a new strict best (and thus the newly selected
            theta), else ``False``.
        """
        if self.best_theta is None or val_fitness > self.best_val_fitness:
            self.best_val_fitness = float(val_fitness)
            self.best_theta = np.asarray(theta, dtype=float).copy()
            self.best_gen = int(generation)
            self.stale_generations = 0
            return True
        self.stale_generations += 1
        return False

    def should_stop(self, patience: int) -> bool:
        """Whether to early-stop: ``patience > 0`` and that many stale generations.

        Args:
            patience: Generations without validation improvement before stopping.
                ``patience <= 0`` disables early stopping.

        Returns:
            ``True`` if training should stop now.
        """
        return patience > 0 and self.stale_generations >= patience
