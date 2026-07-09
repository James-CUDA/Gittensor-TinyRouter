"""Canonical OpenRouter pricing ($/1M tokens, input/output).

Single source of truth for per-model rates. Scripts and the Fugu cost module
import from here so pricing cannot drift across copies.
"""
from __future__ import annotations

from pathlib import Path

from .cost_ledger import read_ledger_entries, verify_ledger_chain

__all__ = [
    "OPENROUTER_PRICES",
    "PRICES",
    "blended_rates",
    "token_cost_usd",
    "ledger_total_usd",
]

# OpenRouter list prices for the default pool ($ per 1M tokens), (input, output).
OPENROUTER_PRICES: dict[str, tuple[float, float]] = {
    "qwen3.5-35b-a3b": (0.14, 1.00),
    "minimax-m3": (0.30, 1.20),
    "deepseek-v4-flash": (0.09, 0.18),
}

# Back-compat alias used by older call sites / docs.
PRICES = OPENROUTER_PRICES


def blended_rates(prices: dict[str, tuple[float, float]] | None = None) -> tuple[float, float]:
    """Mean (input, output) $/1M across the price table."""
    table = prices if prices is not None else OPENROUTER_PRICES
    if not table:
        return 0.0, 0.0
    pin = sum(v[0] for v in table.values()) / len(table)
    pout = sum(v[1] for v in table.values()) / len(table)
    return pin, pout


def token_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    prices: dict[str, tuple[float, float]] | None = None,
    *,
    default_in: float | None = None,
    default_out: float | None = None,
) -> float:
    """Price one API call from per-model token counts."""
    table = prices if prices is not None else OPENROUTER_PRICES
    di, dout = blended_rates(table)
    pin, pout = table.get(
        model,
        (
            default_in if default_in is not None else di,
            default_out if default_out is not None else dout,
        ),
    )
    return prompt_tokens / 1e6 * pin + completion_tokens / 1e6 * pout


def ledger_total_usd(
    path: str | Path,
    prices: dict[str, tuple[float, float]] | None = None,
    *,
    verify_chain: bool = False,
) -> float:
    """Sum USD from a TRINITY_COST_LEDGER jsonl file.

    Hash verification uses :mod:`trinity.llm.cost_ledger` (canonical payload),
    not a re-serialized ``json.dumps`` string.
    """
    if verify_chain:
        valid, _, _ = verify_ledger_chain(path)
        if not valid:
            return 0.0

    table = prices if prices is not None else OPENROUTER_PRICES
    total = 0.0
    for entry in read_ledger_entries(path):
        total += token_cost_usd(
            entry.model, entry.prompt_tokens, entry.completion_tokens, table,
        )
    return round(total, 4)
