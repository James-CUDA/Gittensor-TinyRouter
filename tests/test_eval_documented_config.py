"""Regression: published eval commands must match trinity.eval defaults (#157)."""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def test_eval_cli_defaults_to_trinity_config():
    eval_py = (_REPO / "src" / "trinity" / "eval.py").read_text(encoding="utf-8")
    assert 'default=str(_REPO / "configs" / "trinity.yaml")' in eval_py


def test_published_eval_docs_use_trinity_config():
    agents = (_REPO / "AGENTS.md").read_text(encoding="utf-8")
    run_remote = (_REPO / "scripts/run_remote.sh").read_text(encoding="utf-8")
    assert "run_remote.sh eval --config configs/trinity.yaml" in agents
    assert "run_remote.sh eval  --config configs/trinity.yaml" in run_remote
    assert "run_remote.sh eval" not in agents.replace(
        "run_remote.sh eval --config configs/trinity.yaml", ""
    )
    assert "benchmarks.yaml" not in run_remote.split("eval", 1)[-1]
