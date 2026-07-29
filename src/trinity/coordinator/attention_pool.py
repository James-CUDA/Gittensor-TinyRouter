"""Learned attention pooling over encoder token states.

Paper / submission path uses a **single penultimate token** ``h = H[-2]``
(:mod:`trinity.coordinator.slm`). This module is an optional M1 (and ablation)
path::

    H ∈ R^{K × d_h}  →  AttentivePool  →  h ∈ R^{d_h}  →  head

``h = Σ_k α_k H_k`` with ``α = softmax(H q)`` (or batched). Zero-init query →
uniform weights (= mean pool) at start.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class AttentivePool(nn.Module):
    """Single-query additive attention pool over a token sequence.

    Parameters
    ----------
    d_h:
        Hidden size (must match encoder / head).
    use_key_proj:
        If True, scores use ``(H W_k) q``; else ``H q`` (fewer params).
    """

    def __init__(self, d_h: int = 1024, *, use_key_proj: bool = False) -> None:
        super().__init__()
        self.d_h = int(d_h)
        self.use_key_proj = bool(use_key_proj)
        # Zero query → uniform α at init (mean-pool start).
        self.query = nn.Parameter(torch.zeros(self.d_h))
        if self.use_key_proj:
            self.W_k = nn.Linear(self.d_h, self.d_h, bias=False)
            nn.init.zeros_(self.W_k.weight)  # identity-ish start via residual? 
            # zeros W_k → scores 0 → still uniform; OK
        else:
            self.W_k = None

    @property
    def n_params(self) -> int:
        n = self.query.numel()
        if self.W_k is not None:
            n += self.W_k.weight.numel()
        return int(n)

    def forward(
        self,
        H: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Pool ``H`` → ``(h, alpha)``.

        Parameters
        ----------
        H:
            ``(K, d_h)`` or ``(B, K, d_h)``.
        mask:
            Optional bool / 0-1 mask, same leading dims as ``H`` without ``d_h``.
            True / 1 = keep token.

        Returns
        -------
        h:
            ``(d_h,)`` or ``(B, d_h)``.
        alpha:
            Attention weights, ``(K,)`` or ``(B, K)``.
        """
        if H.dim() == 2:
            return self._forward_one(H, mask)
        if H.dim() == 3:
            hs = []
            alphas = []
            for i in range(H.shape[0]):
                m = None if mask is None else mask[i]
                h_i, a_i = self._forward_one(H[i], m)
                hs.append(h_i)
                alphas.append(a_i)
            return torch.stack(hs, dim=0), torch.stack(alphas, dim=0)
        raise ValueError(f"H must be 2-D or 3-D, got shape {tuple(H.shape)}")

    def _forward_one(self, H: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
        # H: (K, d_h)
        if H.dim() != 2 or H.shape[-1] != self.d_h:
            raise ValueError(f"expected H (K, {self.d_h}), got {tuple(H.shape)}")
        keys = self.W_k(H) if self.W_k is not None else H
        scores = torch.matmul(keys, self.query)  # (K,)
        if mask is not None:
            m = mask.to(dtype=torch.bool, device=scores.device)
            if m.shape != scores.shape:
                raise ValueError(f"mask shape {tuple(m.shape)} != {(scores.shape[0],)}")
            scores = scores.masked_fill(~m, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=-1)
        h = torch.matmul(alpha, H)  # (d_h,)
        return h, alpha

    def pack(self) -> np.ndarray:
        parts = [self.query.detach().cpu().float().numpy().reshape(-1)]
        if self.W_k is not None:
            parts.append(self.W_k.weight.detach().cpu().float().numpy().reshape(-1))
        return np.concatenate(parts).astype(np.float64)

    @torch.no_grad()
    def unpack_into(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta.size != self.n_params:
            raise ValueError(f"theta size {theta.size} != n_params {self.n_params}")
        self.query.copy_(
            torch.from_numpy(theta[: self.d_h]).to(
                dtype=self.query.dtype, device=self.query.device
            )
        )
        if self.W_k is not None:
            w = theta[self.d_h :].reshape(self.d_h, self.d_h)
            self.W_k.weight.copy_(
                torch.from_numpy(w).to(
                    dtype=self.W_k.weight.dtype, device=self.W_k.weight.device
                )
            )


class AttentiveTriageRouter(nn.Module):
    """encode-sequence → AttentivePool → TriageHead.

    Not the paper submission path. For M1 triage on full token states.
    """

    def __init__(
        self,
        triage_head: nn.Module,
        *,
        d_h: int = 1024,
        use_key_proj: bool = False,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.pool = AttentivePool(d_h, use_key_proj=use_key_proj)
        self.head = triage_head
        self.l2_normalize = bool(l2_normalize)
        self.d_h = int(d_h)

    @property
    def n_params(self) -> int:
        head_n = getattr(self.head, "n_params", sum(p.numel() for p in self.head.parameters()))
        return int(self.pool.n_params + head_n)

    def pool_hidden(
        self,
        H: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """``H → (h, alpha)`` with optional L2 on ``h``."""
        h, alpha = self.pool(H, mask)
        if self.l2_normalize:
            if h.dim() == 1:
                n = torch.linalg.vector_norm(h)
                h = h / n if float(n.detach()) > 0 else h
            else:
                h = F.normalize(h, p=2, dim=-1)
        return h, alpha

    def forward(
        self,
        H: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(domain_logits, difficulty_logits, alpha)``."""
        h, alpha = self.pool_hidden(H, mask)
        dom, diff = self.head.forward(h)
        return dom, diff, alpha

    @torch.no_grad()
    def select(
        self,
        H: Tensor,
        mask: Tensor | None = None,
        *,
        sample: bool = False,
        rng: "torch.Generator | None" = None,
    ) -> tuple[str, int, dict[str, Any]]:
        h, alpha = self.pool_hidden(H, mask)
        domain, difficulty, dbg = self.head.select(h, sample=sample, rng=rng)
        dbg = dict(dbg)
        dbg["alpha"] = alpha.detach().to("cpu").float().numpy()
        dbg["pool"] = "attentive"
        return domain, difficulty, dbg


def make_attentive_triage(
    config: str,
    *,
    d_h: int = 1024,
    use_key_proj: bool = False,
    l2_normalize: bool = True,
) -> AttentiveTriageRouter:
    """Factory for attentive 5-domain / 20-domain triage routers."""
    from .triage_head import make_triage_head

    head = make_triage_head(config, d_h=d_h)
    return AttentiveTriageRouter(
        head, d_h=d_h, use_key_proj=use_key_proj, l2_normalize=l2_normalize
    )
