"""CMA-ES receipt plausibility and SVF training-signal checks (gates 9–10).

These gates catch fabricated training metadata and identity SVF packs that claim
high fitness — offline, with no GPU or OpenRouter calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from trinity.optim.sep_cmaes import default_popsize
from trinity.submission.constants import EXPECTED_TOTAL_PARAMS

__all__ = [
    "ReceiptCmaesAudit",
    "SvfTrainingSignalAudit",
    "validate_receipt_cmaes",
    "validate_svf_training_signal",
]

# A pack claiming above this best_fitness must show non-identity SVF adaptation.
_SVF_IDENTITY_FITNESS_THRESHOLD: float = 0.25
# Fraction of SVF scales that must differ from 1.0 when fitness is high.
_SVF_MIN_NONIDENTITY_FRACTION: float = 0.01
_SVF_IDENTITY_TOLERANCE: float = 1e-5


@dataclass(frozen=True)
class ReceiptCmaesAudit:
    """Validate receipt fields against SepCMAES defaults for ``n_total``."""

    n_total: int = EXPECTED_TOTAL_PARAMS

    def expected_popsize(self) -> int:
        return default_popsize(self.n_total)

    def expected_m_cma(self, popsize: int) -> int:
        return popsize // 2

    def validate(self, receipt: Mapping[str, Any]) -> str | None:
        if not receipt:
            return "receipt_missing"

        n_total = receipt.get("n_total")
        if not isinstance(n_total, int) or n_total <= 0:
            return "receipt_n_total_invalid"
        if n_total != self.n_total:
            return f"receipt_n_total_mismatch: got {n_total}, expected {self.n_total}"

        popsize = receipt.get("popsize")
        expected_pop = self.expected_popsize()
        if not isinstance(popsize, int) or popsize <= 0:
            return "receipt_popsize_invalid"
        if popsize != expected_pop:
            return f"receipt_popsize_mismatch: got {popsize}, expected {expected_pop} for n={n_total}"

        m_cma = receipt.get("m_cma")
        expected_mu = self.expected_m_cma(popsize)
        if not isinstance(m_cma, int) or m_cma <= 0:
            return "receipt_m_cma_invalid"
        if m_cma != expected_mu:
            return f"receipt_m_cma_mismatch: got {m_cma}, expected {expected_mu} (floor(popsize/2))"

        generations = receipt.get("generations")
        history = receipt.get("fitness_history")
        if isinstance(generations, int) and generations > 0:
            if generations < 3:
                return f"receipt_generations_too_low: {generations}"
            if isinstance(history, list) and len(history) > 0:
                if abs(len(history) - generations) > 5:
                    return (
                        f"receipt_generations_history_mismatch: "
                        f"generations={generations} history_len={len(history)}"
                    )

        popsize_field = receipt.get("popsize")
        if isinstance(popsize_field, int) and popsize_field > 200:
            return f"receipt_popsize_implausible: {popsize_field}"

        return None


@dataclass(frozen=True)
class SvfTrainingSignalAudit:
    """Reject high-fitness packs whose SVF block never left the CMA-ES initial mean."""

    fitness_threshold: float = _SVF_IDENTITY_FITNESS_THRESHOLD
    min_nonidentity_fraction: float = _SVF_MIN_NONIDENTITY_FRACTION
    identity_tolerance: float = _SVF_IDENTITY_TOLERANCE

    def validate(
        self,
        svf_scales: np.ndarray,
        receipt: Mapping[str, Any],
    ) -> str | None:
        best = float(receipt.get("best_fitness", 0.0) or 0.0)
        if best <= self.fitness_threshold:
            return None

        svf = np.asarray(svf_scales, dtype=np.float64).ravel()
        if svf.size == 0:
            return "svf_empty"

        if np.any(svf <= 0.0):
            return "svf_non_positive"

        if np.allclose(svf, 1.0, atol=self.identity_tolerance, rtol=0.0):
            return (
                f"svf_untrained_identity: best_fitness={best:.4f} > "
                f"{self.fitness_threshold} but all SVF scales are 1.0"
            )

        non_identity = np.mean(np.abs(svf - 1.0) > self.identity_tolerance)
        if non_identity < self.min_nonidentity_fraction:
            return (
                f"svf_insufficient_adaptation: best_fitness={best:.4f} but only "
                f"{non_identity:.2%} of SVF scales differ from 1.0"
            )

        return None


def validate_receipt_cmaes(receipt: Mapping[str, Any]) -> str | None:
    """Gate 9: receipt CMA-ES metadata must match SepCMAES defaults."""
    return ReceiptCmaesAudit().validate(receipt)


def validate_svf_training_signal(
    svf_scales: np.ndarray,
    receipt: Mapping[str, Any],
) -> str | None:
    """Gate 10: high-fitness packs must show plausible SVF adaptation."""
    return SvfTrainingSignalAudit().validate(svf_scales, receipt)
