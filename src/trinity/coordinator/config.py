"""Load coordinator settings from YAML and build a policy."""
from __future__ import annotations

from pathlib import Path

import yaml

from .params import ParamSpec
from .policy import CoordinatorPolicy

__all__ = ["load_coordinator_section", "build_policy_from_config"]


def load_coordinator_section(path: str | Path) -> dict:
    """Return the ``coordinator`` block from a trinity.yaml config."""
    return yaml.safe_load(Path(path).read_text())["coordinator"]


def build_policy_from_config(
    cc: dict,
    *,
    n_models: int,
    n_roles: int | None = None,
) -> tuple[CoordinatorPolicy, ParamSpec]:
    """Build a :class:`CoordinatorPolicy` from a coordinator config dict."""
    kwargs: dict = dict(
        model_name=cc["encoder_model"],
        device=cc.get("device", "cuda:0"),
        dtype=cc.get("dtype", "bfloat16"),
        target_layer=cc["svf"]["target_layer"],
        svf_matrices=cc["svf"].get("matrices"),
        n_models=n_models,
        l2_normalize=cc["hidden_state"].get("l2_normalize", True),
    )
    if n_roles is not None:
        kwargs["n_roles"] = n_roles
    elif "head" in cc:
        kwargs["n_roles"] = cc["head"].get("n_roles", 3)
    return CoordinatorPolicy.build(**kwargs)
