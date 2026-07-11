"""Tests for pack_submission generation auto-detect."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from pack_submission import next_generation


def _touch_dir(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_next_generation_missing_dir(tmp_path):
    assert next_generation(tmp_path / "alice") == 1


def test_next_generation_empty_dir(tmp_path):
    d = tmp_path / "alice"
    d.mkdir()
    assert next_generation(d) == 1


def test_next_generation_gap(tmp_path):
    root = tmp_path / "alice"
    _touch_dir(root, "1")
    _touch_dir(root, "3")
    assert next_generation(root) == 4


def test_next_generation_contiguous(tmp_path):
    root = tmp_path / "alice"
    for n in ("1", "2", "3"):
        _touch_dir(root, n)
    assert next_generation(root) == 4


def test_next_generation_ignores_stray_entries(tmp_path):
    root = tmp_path / "alice"
    _touch_dir(root, "1")
    _touch_dir(root, "2")
    (root / "README").write_text("notes")
    _touch_dir(root, "draft")
    assert next_generation(root) == 3


def test_next_generation_single_high(tmp_path):
    root = tmp_path / "alice"
    _touch_dir(root, "7")
    assert next_generation(root) == 8
