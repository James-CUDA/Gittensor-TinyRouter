"""Evolutionary training (separable CMA-ES) + baseline optimizers."""
from __future__ import annotations

from trinity.optim.lra import LRAConfig
from trinity.optim.sep_cmaes import SepCMAES, default_popsize, run

__all__ = ["LRAConfig", "SepCMAES", "default_popsize", "run"]
