"""Base trainer interface for pluggable training methods.

The coordinator head + SVF scales are trained by sep-CMA-ES by default.
This module defines the abstract interface that any alternative trainer
must implement (e.g., GRPO, PPO, supervised fine-tuning, random search).

To add a new trainer:
1. Subclass ``BaseTrainer``.
2. Implement ``train(policy, pool, tasks, **kwargs) -> dict``.
3. Add a CLI flag to ``train.py`` to select your trainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTrainer(ABC):
    """Abstract interface for coordinator training methods.

    A trainer takes a coordinator policy, a model pool, and a set of
    training tasks, and produces a trained head + SVF configuration
    that maximizes routing accuracy.

    The train method must be async (LLM calls are async).
    """

    @abstractmethod
    async def train(
        self,
        policy,           # CoordinatorPolicy
        pool,             # FireworksPool (or any object with async chat)
        tasks: List[Any],  # List[Task]
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the coordinator and return a summary dict.

        Parameters
        ----------
        policy:
            A ``CoordinatorPolicy`` with encoder already loaded on GPU.
            The trainer calls ``policy.configure(theta)`` to install
            candidate parameters and ``policy.decide(text)`` to route.
        pool:
            Async LLM pool client exposing ``async chat(model, messages, ...)``.
        tasks:
            Training tasks. The trainer is responsible for minibatch
            sampling and evaluation scheduling.
        **kwargs:
            Training hyperparameters (population size, generations, etc.).

        Returns
        -------
        dict
            Summary with at minimum:
            - ``"benchmark"`` (str): task benchmark name
            - ``"best_fitness"`` (float): best achieved mean reward
            - ``"best_theta_path"`` (str): path to saved best_theta.npy
            - ``"run_dir"`` (str): directory with all run artifacts
            - ``"total_cost_usd"`` (float): estimated API cost

            Additional keys are encouraged: ``"history"`` (list of per-iteration
            stats), ``"pool"`` (list of model names), ``"generations"``,
            ``"popsize"`` (if applicable).
        """
        ...
