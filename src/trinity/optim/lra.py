"""Learning-rate adaptation (LRA) for separable CMA-ES on noisy objectives.

Implements a lightweight black-box SNR estimator that scales pycma's effective
learning rates generation-to-generation. When per-generation mean fitness is
flat (pure noise from binary trajectory rewards), η shrinks so the strategy
does not thrash; when a clear signal appears, η recovers.

This is improvement #4 from ``docs/IMPROVEMENTS.md``. The adaptation layer is
optional and default-off so existing training runs remain byte-identical until
``configs/trinity.yaml`` enables it.

Reference: Nomura et al., "Learning Rate Adaptation for CMA-ES", ACM TELO 2025.
The implementation here is deliberately self-contained and uses only generation
mean shifts — no gradients through the LLM pool.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "LRAConfig",
    "LRAState",
    "LRA_OPT_KEYS",
    "capture_baseline_opts",
    "apply_learning_rate",
    "estimate_snr",
    "adapt_eta",
]

# pycma knobs that directly control mean / covariance learning rates.
_LRA_OPT_KEYS: tuple[str, ...] = (
    "CMA_cmean",
    "CMA_cs",
    "CMA_cc",
    "CMA_ccov1",
    "CMA_rankmu",
    "CMA_rankone",
)

LRA_OPT_KEYS = _LRA_OPT_KEYS


@dataclass(frozen=True)
class LRAConfig:
    """Hyperparameters for generation-to-generation learning-rate adaptation."""

    enabled: bool = False
    target_snr: float = 1.0
    beta: float = 0.2
    eta_min: float = 0.05
    eta_max: float = 1.0
    initial_eta: float = 1.0
    eps: float = 1e-12

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> LRAConfig:
        """Parse an optional yaml mapping; missing/empty -> defaults (disabled)."""
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            target_snr=float(raw.get("target_snr", 1.0)),
            beta=float(raw.get("beta", 0.2)),
            eta_min=float(raw.get("eta_min", 0.05)),
            eta_max=float(raw.get("eta_max", 1.0)),
            initial_eta=float(raw.get("initial_eta", 1.0)),
            eps=float(raw.get("eps", 1e-12)),
        )


@dataclass
class LRAState:
    """Mutable adaptation state carried across ``tell`` calls."""

    eta: float = 1.0
    s1: float = 0.0
    s2: float = 0.0
    prev_mean: float | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    def reset(self, cfg: LRAConfig) -> None:
        """Reinitialize state for a fresh optimizer run."""
        self.eta = max(cfg.eta_min, min(cfg.eta_max, cfg.initial_eta))
        self.s1 = 0.0
        self.s2 = 0.0
        self.prev_mean = None
        self.history.clear()

    def observe(
        self,
        gen_mean: float,
        cfg: LRAConfig,
        es: Any,
        baseline: dict[str, float],
        *,
        iteration: int,
    ) -> dict[str, float]:
        """Update SNR EMAs, adapt η, apply scaled learning rates, return diagnostics."""
        if self.prev_mean is None:
            self.prev_mean = float(gen_mean)
            diag = {
                "iteration": float(iteration),
                "gen_mean_fitness": float(gen_mean),
                "delta_mean": 0.0,
                "snr": 0.0,
                "eta": self.eta,
            }
            self.history.append(diag)
            apply_learning_rate(es, baseline, self.eta)
            return diag

        delta = float(gen_mean) - self.prev_mean
        self.prev_mean = float(gen_mean)
        self.s1 = (1.0 - cfg.beta) * self.s1 + cfg.beta * delta
        self.s2 = (1.0 - cfg.beta) * self.s2 + cfg.beta * delta * delta
        snr = estimate_snr(self.s1, self.s2, eps=cfg.eps)
        self.eta = adapt_eta(snr, cfg, self.eta)
        apply_learning_rate(es, baseline, self.eta)
        diag = {
            "iteration": float(iteration),
            "gen_mean_fitness": float(gen_mean),
            "delta_mean": delta,
            "snr": snr,
            "eta": self.eta,
        }
        self.history.append(diag)
        return diag


def estimate_snr(s1: float, s2: float, *, eps: float = 1e-12) -> float:
    """Return |EMA(mean shift)| / sqrt(EMA(squared shift))."""
    return abs(s1) / math.sqrt(max(s2, eps))


def adapt_eta(snr: float, cfg: LRAConfig, prev_eta: float) -> float:
    """Multiplicatively nudge η toward ``target_snr`` and clamp to bounds."""
    if cfg.target_snr <= 0:
        ratio = 1.0
    else:
        ratio = snr / cfg.target_snr
    # Square-root damping keeps adaptation stable on binary noise.
    proposed = prev_eta * math.sqrt(max(ratio, cfg.eps))
    return max(cfg.eta_min, min(cfg.eta_max, proposed))


def capture_baseline_opts(es: Any) -> dict[str, float]:
    """Snapshot pycma learning-rate knobs at initialization time."""
    baseline: dict[str, float] = {}
    opts = getattr(es, "opts", None)
    if opts is None:
        return baseline
    getter = opts.get if hasattr(opts, "get") else lambda k, d=None: getattr(opts, k, d)
    for key in _LRA_OPT_KEYS:
        val = getter(key, None)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            baseline[key] = float(val)
    return baseline


def apply_learning_rate(es: Any, baseline: dict[str, float], eta: float) -> None:
    """Scale captured pycma learning rates by ``eta``."""
    if not baseline:
        return
    opts = es.opts
    for key, base in baseline.items():
        opts[key] = base * float(eta)
