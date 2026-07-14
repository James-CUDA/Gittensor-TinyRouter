"""Fitness-history, ledger-volume, and head-diversity provenance checks.

Gate 10 (``fitness_history_sequence``) is a hard offline gate (after #199's
gates 8–9: ``artifact_manifest``, ``receipt_cmaes``).

``ledger_call_volume`` and ``head_routing_diversity`` are **advisories** only:
they surface warnings in preflight/pr_eval but never reject a submission until
run-scoped ledger provenance exists (see #210 maintainer review).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from trinity.llm.cost_ledger import read_ledger_entries, verify_ledger_chain
from trinity.llm.openrouter_pricing import normalize_model_slug
from trinity.submission.constants import (
    HEAD_AGENT_COLLAPSE_COSINE,
    LEDGER_MIN_DISTINCT_MODELS,
    LEDGER_MIN_TOKENS_PER_CANDIDATE,
    N_HEAD_MODELS,
)

__all__ = [
    "FitnessHistorySequenceAudit",
    "HeadRoutingDiversityAudit",
    "LedgerTrainingVolumeAudit",
    "validate_fitness_history_sequence",
    "validate_head_routing_diversity",
    "validate_ledger_call_volume",
]


@dataclass(frozen=True)
class FitnessHistorySequenceAudit:
    """Validate per-generation fitness_history structure and ordering."""

    fitness_epsilon: float = 1e-6

    def validate(self, receipt: Mapping[str, Any]) -> str | None:
        if not receipt:
            return "receipt_missing"

        history = receipt.get("fitness_history")
        if not isinstance(history, list) or not history:
            return "receipt_fitness_history_invalid"

        generations: list[int] = []
        running_best = float("-inf")

        for idx, entry in enumerate(history):
            if not isinstance(entry, dict):
                return f"receipt_fitness_history_entry_not_object: index {idx}"

            gen_raw = entry.get("generation")
            if not isinstance(gen_raw, int) or gen_raw < 0:
                return f"receipt_fitness_history_generation_invalid: index {idx}"
            generations.append(gen_raw)

            mean = _optional_float(entry, "mean_fitness", "gen_mean_fitness")
            max_f = _optional_float(entry, "max_fitness", "gen_max_fitness")
            best = _optional_float(entry, "best_fitness")

            if mean is not None and max_f is not None and mean > max_f + self.fitness_epsilon:
                return (
                    f"receipt_fitness_history_mean_gt_max: generation {gen_raw} "
                    f"mean={mean:.6f} max={max_f:.6f}"
                )
            if max_f is not None and best is not None and max_f > best + self.fitness_epsilon:
                return (
                    f"receipt_fitness_history_max_gt_best: generation {gen_raw} "
                    f"max={max_f:.6f} best={best:.6f}"
                )

            peak = best if best is not None else max_f if max_f is not None else mean
            if peak is not None:
                if peak + self.fitness_epsilon < running_best:
                    return (
                        f"receipt_fitness_history_best_regressed: generation {gen_raw} "
                        f"peak={peak:.6f} < running_best={running_best:.6f}"
                    )
                running_best = max(running_best, peak)

        if len(set(generations)) != len(generations):
            return "receipt_fitness_history_duplicate_generations"

        sorted_gens = sorted(generations)
        base = sorted_gens[0]
        expected = list(range(base, base + len(sorted_gens)))
        if sorted_gens != expected:
            return (
                f"receipt_fitness_history_nonconsecutive: "
                f"got {sorted_gens[:5]}{'...' if len(sorted_gens) > 5 else ''}"
            )

        claimed = receipt.get("generations")
        if isinstance(claimed, int) and claimed > 0 and abs(claimed - len(history)) > 5:
            return (
                f"receipt_generations_history_mismatch: "
                f"generations={claimed} history_len={len(history)}"
            )

        receipt_best = _optional_float(receipt, "best_fitness")
        if receipt_best is not None and running_best > float("-inf"):
            if abs(receipt_best - running_best) > 0.05:
                return (
                    f"receipt_best_fitness_history_peak_mismatch: "
                    f"receipt={receipt_best:.4f} history_peak={running_best:.4f}"
                )

        return None


@dataclass(frozen=True)
class LedgerTrainingVolumeAudit:
    """When a ledger is available, verify API volume matches claimed CMA-ES scale."""

    min_tokens_per_candidate: int = LEDGER_MIN_TOKENS_PER_CANDIDATE
    min_distinct_models: int = LEDGER_MIN_DISTINCT_MODELS

    def validate(
        self,
        receipt: Mapping[str, Any],
        ledger_path: str | None,
    ) -> str | None:
        if not ledger_path:
            return None
        if not receipt:
            return None

        receipt_cost = float(receipt.get("total_cost_usd", 0.0) or 0.0)
        if receipt_cost <= 0.0:
            return None

        try:
            valid, num_entries, err = verify_ledger_chain(ledger_path)
        except OSError:
            return "ledger_volume_unreadable"
        if not valid:
            return f"ledger_volume_unverifiable: {err}"

        generations = receipt.get("generations")
        popsize = receipt.get("popsize")
        if not isinstance(generations, int) or generations < 1:
            return None
        if not isinstance(popsize, int) or popsize < 1:
            return None

        entries = read_ledger_entries(ledger_path)
        total_tokens = sum(entry.total_tokens for entry in entries)
        min_tokens = generations * popsize * self.min_tokens_per_candidate
        if total_tokens < min_tokens:
            return (
                f"ledger_volume_too_low: {total_tokens} tokens "
                f"< {min_tokens} ({generations}×{popsize}×{self.min_tokens_per_candidate})"
            )

        if num_entries < max(10, generations):
            return (
                f"ledger_call_count_too_low: {num_entries} entries "
                f"< min({generations} generations, 10)"
            )

        pool = receipt.get("pool_models")
        if isinstance(pool, list) and pool:
            pool_slugs = {normalize_model_slug(str(m)) for m in pool if isinstance(m, str)}
            seen = {normalize_model_slug(entry.model) for entry in entries}
            overlap = pool_slugs & seen
            if len(overlap) < min(self.min_distinct_models, len(pool_slugs)):
                return (
                    f"ledger_pool_coverage_too_low: {len(overlap)} pool models in ledger, "
                    f"need >= {min(self.min_distinct_models, len(pool_slugs))}"
                )

        return None


@dataclass(frozen=True)
class HeadRoutingDiversityAudit:
    """Reject heads whose agent logit rows collapse to identical routing."""

    n_agent_rows: int = N_HEAD_MODELS
    collapse_threshold: float = HEAD_AGENT_COLLAPSE_COSINE
    min_agent_row_norm: float = 1e-4

    def validate(self, head_weights: np.ndarray) -> str | None:
        head = np.asarray(head_weights, dtype=np.float64)
        if head.ndim != 2 or head.shape[0] < self.n_agent_rows:
            return "head_routing_diversity_shape_invalid"

        agent = head[: self.n_agent_rows].copy()
        agent -= agent.mean(axis=0, keepdims=True)

        norms = np.linalg.norm(agent, axis=1)
        if np.all(norms < self.min_agent_row_norm):
            return "head_routing_diversity_agent_rows_near_zero"

        active = int(np.sum(norms >= self.min_agent_row_norm))
        if active < 2:
            return "head_routing_diversity_single_active_agent_row"

        for i in range(self.n_agent_rows):
            for j in range(i + 1, self.n_agent_rows):
                if norms[i] < self.min_agent_row_norm or norms[j] < self.min_agent_row_norm:
                    continue
                sim = _cosine_similarity(agent[i], agent[j])
                if sim > self.collapse_threshold:
                    return (
                        f"head_routing_diversity_agent_collapse: "
                        f"rows {i} and {j} cosine={sim:.6f} > {self.collapse_threshold}"
                    )

        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a_arr, b_arr) / (na * nb))


def _optional_float(record: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        val = record.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def validate_fitness_history_sequence(receipt: Mapping[str, Any]) -> str | None:
    """Gate 8: fitness_history must have consecutive, consistent generations."""
    return FitnessHistorySequenceAudit().validate(receipt)


def validate_ledger_call_volume(
    receipt: Mapping[str, Any],
    ledger_path: str | None,
) -> str | None:
    """Advisory: verified ledger should reflect CMA-ES-scale API volume (optional)."""
    return LedgerTrainingVolumeAudit().validate(receipt, ledger_path)


def validate_head_routing_diversity(head_weights: np.ndarray) -> str | None:
    """Advisory: agent logit rows should not collapse to identical routing."""
    return HeadRoutingDiversityAudit().validate(head_weights)
