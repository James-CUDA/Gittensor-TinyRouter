"""Offline tests for the on-disk LLM cache audit. No network, no GPU."""
from __future__ import annotations

import json

import pytest

from trinity.llm.cache import ResponseCache
from trinity.llm.cache_audit import audit_cache, render

# OpenRouter pool rates ($/1M in, out): deepseek (0.09, 0.18) -> $0.27 for 1M+1M;
# qwen (0.14, 1.00) -> $1.14 for 1M+1M.
_M = 1_000_000


def _cache(tmp_path):
    return ResponseCache(tmp_path)


def _write(cache, key, model, pt, ct, text="ok", finish="stop"):
    cache.put(key, {"model": model, "text": text, "prompt_tokens": pt,
                    "completion_tokens": ct, "finish_reason": finish})


def _populate(tmp_path):
    c = _cache(tmp_path)
    _write(c, "aa01", "deepseek-v4-flash", _M, _M)                 # $0.27
    _write(c, "bb02", "qwen3.5-35b-a3b", _M, _M)                   # $1.14
    _write(c, "cc03", "qwen3.5-35b-a3b", 0, 0, text="", finish="error")   # polluted (error)
    _write(c, "dd04", "deepseek-v4-flash", 0, 0, text="   ")       # polluted (blank text)
    # a stale-schema record ResponseCache.get would ignore
    (tmp_path / "ee" ).mkdir(parents=True, exist_ok=True)
    (tmp_path / "ee" / "ee05.json").write_text(json.dumps({"v": 999, "model": "x"}))
    # an unreadable file
    (tmp_path / "ff").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ff" / "ff06.json").write_text("not json{")
    return tmp_path


def test_valid_entries_and_per_model_dollar_value(tmp_path):
    a = audit_cache(_populate(tmp_path))
    assert a.n_files == 6
    assert a.n_valid == 4                 # the 2 clean + 2 polluted are all current-schema
    assert a.n_stale == 1 and a.n_unreadable == 1
    by_model = {m.model: m for m in a.per_model}
    assert by_model["qwen3.5-35b-a3b"].entries == 2       # clean + polluted-error
    assert by_model["deepseek-v4-flash"].entries == 2
    assert by_model["qwen3.5-35b-a3b"].dollar_value == pytest.approx(1.14)
    assert by_model["deepseek-v4-flash"].dollar_value == pytest.approx(0.27)
    assert a.total_dollar_value == pytest.approx(1.41)
    # sorted by dollar value descending
    assert [m.model for m in a.per_model] == ["qwen3.5-35b-a3b", "deepseek-v4-flash"]


def test_pollution_is_counted_and_sampled(tmp_path):
    a = audit_cache(_populate(tmp_path))
    assert a.n_polluted == 2              # the error entry + the blank-text entry
    assert set(a.polluted_keys) == {"cc03", "dd04"}


def test_token_totals(tmp_path):
    a = audit_cache(_populate(tmp_path))
    assert a.total_prompt_tokens == 2 * _M and a.total_completion_tokens == 2 * _M


def test_empty_or_absent_cache(tmp_path):
    a = audit_cache(tmp_path / "does_not_exist")
    assert a.n_files == 0 and a.n_valid == 0 and a.per_model == []
    assert a.total_dollar_value == 0.0
    assert "empty or absent" in render(a)


def test_clean_cache_reports_no_pollution(tmp_path):
    c = _cache(tmp_path)
    _write(c, "aa01", "deepseek-v4-flash", 100, 200)
    a = audit_cache(tmp_path)
    assert a.n_polluted == 0 and a.polluted_keys == []
    assert "no polluted entries" in render(a)


def test_unknown_model_is_priced_via_blended_fallback(tmp_path):
    # An unknown slug must not be dropped: it prices through the blended default.
    c = _cache(tmp_path)
    _write(c, "aa01", "some-new-model", _M, 0)
    a = audit_cache(tmp_path)
    assert a.per_model[0].model == "some-new-model"
    assert a.per_model[0].dollar_value > 0.0     # blended in-rate applied, row not lost


def test_render_and_to_dict(tmp_path):
    a = audit_cache(_populate(tmp_path))
    md = render(a)
    assert "cache audit" in md.lower() and "value on re-serve" in md
    assert "polluted entries" in md and "qwen3.5-35b-a3b" in md
    d = a.to_dict()
    assert json.loads(json.dumps(d))["n_polluted"] == 2
    assert d["per_model"][0]["dollar_value"] == pytest.approx(1.14)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
