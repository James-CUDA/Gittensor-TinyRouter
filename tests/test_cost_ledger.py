"""Tests for cost-ledger hash-chain integrity."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

from cost_report import verify_ledger_chain
from trinity.llm.fireworks_client import _ledger_append


def _write_chain(path: Path, entries: list[tuple[str, int, int]]) -> None:
    prev_hash = ""
    with path.open("w") as f:
        for model, pt, ct in entries:
            body = {"m": model, "p": pt, "c": ct}
            payload = json.dumps(body, sort_keys=True)
            h = hashlib.sha256((prev_hash + payload).encode()).hexdigest()
            f.write(json.dumps({**body, "h": h}, sort_keys=True) + "\n")
            prev_hash = h


def test_verify_ledger_chain_accepts_canonical_entries(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _write_chain(ledger, [("glm-5p2", 100, 50), ("deepseek-v4-pro", 200, 80)])

    valid, count, err = verify_ledger_chain(str(ledger))
    assert valid, err
    assert count == 2


def test_ledger_append_matches_verifier(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TRINITY_COST_LEDGER", str(ledger))

    _ledger_append("accounts/fireworks/models/glm-5p2", 120, 40)
    _ledger_append("accounts/fireworks/models/kimi-k2p6", 300, 60)

    valid, count, err = verify_ledger_chain(str(ledger))
    assert valid, err
    assert count == 2


def test_broken_chain_rejected(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"c": 1, "h": "deadbeef", "m": "glm-5p2", "p": 1}\n')

    valid, _, err = verify_ledger_chain(str(ledger))
    assert not valid
    assert "hash mismatch" in err
