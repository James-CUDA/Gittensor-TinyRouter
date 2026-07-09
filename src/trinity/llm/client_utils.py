"""Shared helpers for LLM client wrappers."""
from __future__ import annotations

import inspect

__all__ = ["filter_supported_kwargs"]


def filter_supported_kwargs(fn, kwargs: dict) -> dict:
    """Drop kwargs a (possibly stub) callable does not accept."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}
