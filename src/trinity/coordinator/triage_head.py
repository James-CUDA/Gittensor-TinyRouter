"""Milestone-1 triage heads: domain + difficulty from SLM hidden state.

Sibling of :class:`~trinity.coordinator.head.LinearHead`, but for prompt
**triage** (not model×role routing)::

    h ∈ R^{d_h}
    z_domain = W_domain · h ,   W_domain ∈ R^{C × d_h}
    z_diff   = W_diff   · h ,   W_diff   ∈ R^{5 × d_h}

Two separate softmaxes (same style as LinearHead's agent/role split). No bias,
no activation. Zero-init → uniform over domains and difficulties.

Two concrete heads ship for the Hub configs on ``James-Cuda/tinyrouter-m1``:

* :class:`TriageHead5Domain`  — ``C=5``  (``5-domain`` config)
* :class:`TriageHead20Domain` — ``C=20`` (``20-domain`` / GCI-Bench topics)

Not wired into the submission / CMA-ES path — supervised CE training only.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import torch
from torch import Tensor
from torch import nn

from trinity.m1.domains import DOMAINS_20, DOMAINS_5, N_DIFFICULTY

# Re-export for callers that imported from triage_head.
__all__ = [
    "DOMAINS_5",
    "DOMAINS_20",
    "N_DIFFICULTY",
    "TriageHead",
    "TriageHead5Domain",
    "TriageHead20Domain",
    "make_triage_head",
]


class TriageHead(nn.Module):
    """Bias-free dual linear head: domain + difficulty.

    Parameters
    ----------
    domains:
        Closed ordered domain label list. Index ``i`` ↔ ``domains[i]``.
    d_h:
        SLM hidden size (Qwen3-0.6B = 1024).
    n_diff:
        Number of difficulty classes (default 5 → labels 1..5).
    """

    def __init__(
        self,
        domains: Sequence[str],
        *,
        d_h: int = 1024,
        n_diff: int = N_DIFFICULTY,
    ) -> None:
        super().__init__()
        if len(domains) < 1:
            raise ValueError("domains must be non-empty")
        if n_diff < 1:
            raise ValueError("n_diff must be >= 1")
        if len(set(domains)) != len(domains):
            raise ValueError("domains must be unique")
        self.domains: tuple[str, ...] = tuple(domains)
        self.n_domains = len(self.domains)
        self.d_h = int(d_h)
        self.n_diff = int(n_diff)
        self._domain_to_idx = {d: i for i, d in enumerate(self.domains)}
        # Zero-init = uniform triage policy (matches LinearHead).
        self.W_domain = nn.Parameter(torch.zeros(self.n_domains, self.d_h))
        self.W_diff = nn.Parameter(torch.zeros(self.n_diff, self.d_h))

    @property
    def n_params(self) -> int:
        return int(self.W_domain.numel() + self.W_diff.numel())

    def domain_index(self, domain: str) -> int:
        try:
            return self._domain_to_idx[domain]
        except KeyError as e:
            raise KeyError(f"unknown domain {domain!r}; expected one of {self.domains}") from e

    def difficulty_to_index(self, difficulty: int) -> int:
        """Map difficulty label ``1..n_diff`` → logit index ``0..n_diff-1``."""
        d = int(difficulty)
        if d < 1 or d > self.n_diff:
            raise ValueError(f"difficulty must be in 1..{self.n_diff}, got {difficulty}")
        return d - 1

    def index_to_difficulty(self, idx: int) -> int:
        i = int(idx)
        if i < 0 or i >= self.n_diff:
            raise ValueError(f"difficulty index must be in 0..{self.n_diff - 1}, got {idx}")
        return i + 1

    @torch.no_grad()
    def load_weights(
        self,
        W_domain: "np.ndarray | Tensor",
        W_diff: "np.ndarray | Tensor",
    ) -> None:
        """Install ``(n_domains, d_h)`` and ``(n_diff, d_h)`` weight matrices."""
        self._copy_param(self.W_domain, W_domain, (self.n_domains, self.d_h), "W_domain")
        self._copy_param(self.W_diff, W_diff, (self.n_diff, self.d_h), "W_diff")

    @staticmethod
    def _copy_param(
        param: nn.Parameter,
        W: "np.ndarray | Tensor",
        shape: tuple[int, int],
        name: str,
    ) -> None:
        if isinstance(W, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(W))
        else:
            t = W
        if tuple(t.shape) != shape:
            raise ValueError(f"{name} shape {tuple(t.shape)} != expected {shape}")
        param.copy_(t.to(dtype=param.dtype, device=param.device))

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(domain_logits, difficulty_logits)``.

        ``h`` shape ``(d_h,)`` or ``(..., d_h)``. Leading dims preserved.
        """
        z_dom = torch.matmul(h, self.W_domain.t())
        z_diff = torch.matmul(h, self.W_diff.t())
        return z_dom, z_diff

    @torch.no_grad()
    def select(
        self,
        h: Tensor,
        *,
        sample: bool = False,
        rng: "torch.Generator | None" = None,
    ) -> tuple[str, int, dict[str, Any]]:
        """Pick ``(domain, difficulty)`` from one hidden state.

        Eval (``sample=False``): argmax each group.
        Train optional (``sample=True``): multinomial each group.
        ``difficulty`` is returned as a **1-based** label.
        """
        h = h.squeeze(0) if h.dim() == 2 and h.shape[0] == 1 else h
        if h.dim() != 1:
            raise ValueError(
                f"select expects a single hidden state of shape (d_h,), got {tuple(h.shape)}"
            )

        dom_logits, diff_logits = self.forward(h)
        dom_probs = torch.softmax(dom_logits, dim=-1)
        diff_probs = torch.softmax(diff_logits, dim=-1)

        if sample:
            device = rng.device if rng is not None else dom_probs.device
            dom_idx = int(torch.multinomial(dom_probs.to(device), 1, generator=rng).item())
            diff_idx = int(torch.multinomial(diff_probs.to(device), 1, generator=rng).item())
        else:
            dom_idx = int(torch.argmax(dom_logits, dim=-1).item())
            diff_idx = int(torch.argmax(diff_logits, dim=-1).item())

        domain = self.domains[dom_idx]
        difficulty = self.index_to_difficulty(diff_idx)
        debug: dict[str, Any] = {
            "domain_logits": dom_logits.detach().to("cpu").float().numpy(),
            "difficulty_logits": diff_logits.detach().to("cpu").float().numpy(),
            "domain_probs": dom_probs.detach().to("cpu").float().numpy(),
            "difficulty_probs": diff_probs.detach().to("cpu").float().numpy(),
            "domain_idx": dom_idx,
            "difficulty_idx": diff_idx,
            "domain": domain,
            "difficulty": difficulty,
            "sampled": sample,
        }
        return domain, difficulty, debug

    def pack(self) -> np.ndarray:
        """Flat θ = ``[W_domain row-major | W_diff row-major]``."""
        wd = self.W_domain.detach().to("cpu").float().numpy().reshape(-1)
        wf = self.W_diff.detach().to("cpu").float().numpy().reshape(-1)
        return np.concatenate([wd, wf]).astype(np.float64)

    @torch.no_grad()
    def unpack_into(self, theta: np.ndarray) -> None:
        """Inverse of :meth:`pack`."""
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta.size != self.n_params:
            raise ValueError(f"theta size {theta.size} != n_params {self.n_params}")
        n_dom = self.n_domains * self.d_h
        self.load_weights(
            theta[:n_dom].reshape(self.n_domains, self.d_h),
            theta[n_dom:].reshape(self.n_diff, self.d_h),
        )


class TriageHead5Domain(TriageHead):
    """Triage head for Hub config ``5-domain`` (C=5, difficulty 1–5)."""

    DOMAINS: tuple[str, ...] = DOMAINS_5

    def __init__(self, *, d_h: int = 1024, n_diff: int = N_DIFFICULTY) -> None:
        super().__init__(self.DOMAINS, d_h=d_h, n_diff=n_diff)


class TriageHead20Domain(TriageHead):
    """Triage head for Hub config ``20-domain`` (GCI-Bench topics, C=20)."""

    DOMAINS: tuple[str, ...] = DOMAINS_20

    def __init__(self, *, d_h: int = 1024, n_diff: int = N_DIFFICULTY) -> None:
        super().__init__(self.DOMAINS, d_h=d_h, n_diff=n_diff)


def make_triage_head(config: str, *, d_h: int = 1024) -> TriageHead:
    """Factory: ``config`` ∈ {``5-domain``, ``20-domain``, ``5``, ``20``}."""
    key = str(config).strip().lower().replace("_", "-")
    if key in ("5-domain", "5", "five"):
        return TriageHead5Domain(d_h=d_h)
    if key in ("20-domain", "20", "twenty", "gci"):
        return TriageHead20Domain(d_h=d_h)
    raise ValueError(
        f"unknown triage config {config!r}; expected '5-domain' or '20-domain'"
    )
