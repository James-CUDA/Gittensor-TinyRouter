"""Frozen-file tamper detection (COMPETITION_RULES §Frozen files).

The load-bearing test here is ``test_rules_match_the_published_table``: it parses
the frozen-file table out of ``docs/COMPETITION_RULES.md`` and asserts the code's
``FROZEN_RULES`` matches it exactly. The document stays the single source of
truth, and the two cannot drift.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.submission.frozen_files import (
    BENCHMARK_DIR_ENV,
    FROZEN_RULES,
    FrozenRule,
    audit_frozen_files,
    frozen_violations,
    match_frozen,
    normalize_path,
)
from trinity.submission.gates import (
    OFFLINE_ADVISORIES,
    OFFLINE_GATES,
    PreflightContext,
)

_REPO = Path(__file__).resolve().parents[1]
_RULES_MD = _REPO / "docs" / "COMPETITION_RULES.md"
_SCRIPT = _REPO / "scripts" / "frozen_files_report.py"
_SRC = str(_REPO / "src")


# --------------------------------------------------------------------------
# the published table is the source of truth
# --------------------------------------------------------------------------


def _published_rows() -> list[tuple[str, str]]:
    """Parse the ``## Frozen files`` markdown table into (pattern, reason)."""
    text = _RULES_MD.read_text()
    section = text.split("## Frozen files", 1)[1].split("\n##", 1)[0]
    rows: list[tuple[str, str]] = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2 or cells[0] == "File":
            continue
        rows.append((cells[0], cells[1]))
    return rows


def test_the_table_is_parseable_and_non_trivial():
    rows = _published_rows()
    assert len(rows) >= 6, rows


def test_rules_match_the_published_table():
    """FROZEN_RULES must equal the document, row for row.

    The ``$TINYROUTER_BENCHMARK_DIR`` row is excluded: its location is only known
    from the environment, so it is resolved at match time rather than listed.
    """
    published = [
        (p, r) for p, r in _published_rows() if BENCHMARK_DIR_ENV not in p
    ]
    stripped = [(p.strip("`"), r) for p, r in published]
    coded = [(rule.pattern, rule.reason) for rule in FROZEN_RULES]
    assert coded == stripped


def test_the_benchmark_dir_row_exists_in_the_table():
    """If this row disappears, the env-var branch is dead code."""
    assert any(BENCHMARK_DIR_ENV in p for p, _ in _published_rows())


def test_competition_rules_still_defers_this_to_the_maintainer():
    """The gap this module closes. If the doc gains a gate number, revisit."""
    text = _RULES_MD.read_text()
    assert "Modifying frozen files in a submission PR" in text


# --------------------------------------------------------------------------
# normalize_path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,want",
    [
        ("scripts/pr_eval.py", "scripts/pr_eval.py"),
        ("./scripts/pr_eval.py", "scripts/pr_eval.py"),
        (".//scripts/pr_eval.py", "scripts/pr_eval.py"),
        ("scripts\\pr_eval.py", "scripts/pr_eval.py"),
        ("  scripts/pr_eval.py  ", "scripts/pr_eval.py"),
        ("/scripts/pr_eval.py", "scripts/pr_eval.py"),
        ("scripts//pr_eval.py", "scripts/pr_eval.py"),
        ("", ""),
    ],
)
def test_normalize_path(raw, want):
    assert normalize_path(raw) == want


# --------------------------------------------------------------------------
# match_frozen
# --------------------------------------------------------------------------


def test_exact_frozen_path_matches():
    m = match_frozen("scripts/pr_eval.py")
    assert m is not None
    assert m.rule.reason == "The evaluation orchestrator"


def test_the_shared_grader_is_frozen():
    assert match_frozen("src/trinity/orchestration/reward.py") is not None


def test_glob_matches_any_submission_module():
    m = match_frozen("src/trinity/submission/gates.py")
    assert m is not None
    assert m.rule.pattern == "src/trinity/submission/*.py"


def test_the_most_specific_rule_wins():
    """constants.py is covered by the glob but has its own published reason."""
    m = match_frozen("src/trinity/submission/constants.py")
    assert m is not None
    assert m.rule.reason == "Frozen constants (pool, margin, params)"


def test_glob_is_not_recursive():
    """The published pattern is ``submission/*.py`` -- one level, shell semantics."""
    assert match_frozen("src/trinity/submission/nested/deep.py") is None


def test_non_python_file_in_submission_is_not_matched_by_the_glob():
    assert match_frozen("src/trinity/submission/README.md") is None


def test_ordinary_paths_are_not_frozen():
    for p in [
        "src/trinity/analysis/transcript_budget.py",
        "tests/test_frozen_files.py",
        "README.md",
        "submissions/minion/theta.npy",
    ]:
        assert match_frozen(p) is None, p


def test_leaderboard_json_is_frozen_only_at_the_repo_root():
    assert match_frozen("leaderboard.json") is not None
    assert match_frozen("submissions/mine/leaderboard.json") is None


def test_normalization_is_applied_before_matching():
    assert match_frozen("./scripts/pr_eval.py") is not None
    assert match_frozen("scripts\\pr_eval.py") is not None


def test_empty_path_is_not_a_match():
    assert match_frozen("") is None
    assert match_frozen("   ") is None


# --------------------------------------------------------------------------
# $TINYROUTER_BENCHMARK_DIR
# --------------------------------------------------------------------------


def test_benchmark_dir_files_are_frozen_when_the_env_var_is_set(monkeypatch):
    monkeypatch.setenv(BENCHMARK_DIR_ENV, "secret_bench")
    m = match_frozen("secret_bench/mmlu.enc")
    assert m is not None
    assert m.rule.reason == "Encrypted hidden benchmarks"


def test_benchmark_dir_rule_is_inert_when_unset(monkeypatch):
    monkeypatch.delenv(BENCHMARK_DIR_ENV, raising=False)
    assert match_frozen("secret_bench/mmlu.enc") is None


def test_benchmark_dir_matches_nested_files(monkeypatch):
    monkeypatch.setenv(BENCHMARK_DIR_ENV, "bench")
    assert match_frozen("bench/a/b/c.enc") is not None


def test_a_sibling_prefix_is_not_inside_the_benchmark_dir(monkeypatch):
    """``bench_old/x`` must not match a root of ``bench`` -- prefix != subtree."""
    monkeypatch.setenv(BENCHMARK_DIR_ENV, "bench")
    assert match_frozen("bench_old/x.enc") is None


def test_benchmark_dir_env_is_normalized(monkeypatch):
    monkeypatch.setenv(BENCHMARK_DIR_ENV, "./bench/")
    assert match_frozen("bench/x.enc") is not None


def test_blank_benchmark_dir_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv(BENCHMARK_DIR_ENV, "   ")
    assert match_frozen("x/y.enc") is None


# --------------------------------------------------------------------------
# frozen_violations / audit_frozen_files
# --------------------------------------------------------------------------


def test_violations_preserve_input_order():
    hits = frozen_violations(
        ["README.md", "leaderboard.json", "scripts/pr_eval.py"]
    )
    assert [h.path for h in hits] == ["leaderboard.json", "scripts/pr_eval.py"]


def test_violations_are_deduplicated():
    hits = frozen_violations(["scripts/pr_eval.py", "./scripts/pr_eval.py"])
    assert len(hits) == 1


def test_clean_diff_produces_no_advisory():
    assert audit_frozen_files(["README.md", "src/trinity/eval.py"]) is None


def test_empty_or_missing_diff_produces_no_advisory():
    assert audit_frozen_files([]) is None
    assert audit_frozen_files(None) is None


def test_advisory_names_every_offender_and_its_reason():
    msg = audit_frozen_files(["scripts/pr_eval.py", "leaderboard.json"])
    assert msg is not None
    assert "2 frozen files modified" in msg
    assert "scripts/pr_eval.py" in msg
    assert "leaderboard.json" in msg
    assert "The evaluation orchestrator" in msg


def test_advisory_is_singular_for_one_offender():
    msg = audit_frozen_files(["leaderboard.json"])
    assert msg is not None and "1 frozen file modified" in msg


def test_violation_to_dict_is_serializable():
    hit = frozen_violations(["scripts/pr_eval.py"])[0]
    json.dumps(hit.to_dict())
    assert hit.to_dict()["pattern"] == "scripts/pr_eval.py"


def test_frozen_rule_glob_flag():
    assert FrozenRule("a/*.py", "r").is_glob is True
    assert FrozenRule("a/b.py", "r").is_glob is False


# --------------------------------------------------------------------------
# gate-chain wiring -- must not change any existing behaviour
# --------------------------------------------------------------------------


def test_the_check_is_an_advisory_not_a_blocking_gate():
    names = [g.name for g in OFFLINE_ADVISORIES]
    assert "frozen_files" in names
    assert "frozen_files" not in [g.name for g in OFFLINE_GATES]


def test_the_blocking_gate_chain_is_unchanged():
    """The rejection criteria must be exactly what shipped before this PR."""
    assert [g.name for g in OFFLINE_GATES] == [
        "rate_limit", "weights", "duplicate", "receipt",
        "ledger_cost", "pack_schema", "theta_integrity",
    ]


def test_context_defaults_to_no_diff():
    ctx = PreflightContext(benchmark="mmlu", leaderboard={}, submissions_root=Path("."))
    assert ctx.changed_paths == ()


def test_advisory_is_silent_without_a_diff():
    """Every existing caller constructs the context without changed_paths."""
    ctx = PreflightContext(benchmark="mmlu", leaderboard={}, submissions_root=Path("."))
    gate = next(g for g in OFFLINE_ADVISORIES if g.name == "frozen_files")
    assert gate.run(object(), ctx).ok is True


def test_advisory_reports_when_a_diff_is_supplied():
    ctx = PreflightContext(
        benchmark="mmlu", leaderboard={}, submissions_root=Path("."),
        changed_paths=("scripts/pr_eval.py",),
    )
    gate = next(g for g in OFFLINE_ADVISORIES if g.name == "frozen_files")
    result = gate.run(object(), ctx)
    assert result.ok is False
    assert result.reason is not None and "pr_eval.py" in result.reason


# --------------------------------------------------------------------------
# report script
# --------------------------------------------------------------------------


def _run(*args, stdin: str | None = None):
    env = {**os.environ, "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True, text=True, env=env, input=stdin,
    )


def test_script_clean_diff_exits_zero(tmp_path):
    p = tmp_path / "changed.txt"
    p.write_text("README.md\nsrc/trinity/eval.py\n")
    r = _run("--changed", str(p))
    assert r.returncode == 0, r.stderr
    assert "no frozen files" in r.stdout.lower()


def test_script_flags_a_frozen_file(tmp_path):
    p = tmp_path / "changed.txt"
    p.write_text("scripts/pr_eval.py\n")
    r = _run("--changed", str(p))
    assert r.returncode == 1
    assert "pr_eval.py" in r.stdout


def test_script_reads_stdin():
    r = _run("--changed", "-", stdin="leaderboard.json\n")
    assert r.returncode == 1
    assert "leaderboard.json" in r.stdout


def test_script_json_output(tmp_path):
    p = tmp_path / "changed.txt"
    p.write_text("scripts/pr_eval.py\n")
    r = _run("--changed", str(p), "--json")
    assert r.returncode == 1
    payload = json.loads(r.stdout)
    assert payload["violations"][0]["path"] == "scripts/pr_eval.py"


def test_script_missing_file_is_graceful(tmp_path):
    r = _run("--changed", str(tmp_path / "nope.txt"))
    assert r.returncode == 2
    assert "no such file" in r.stderr


def test_script_ignores_blank_lines(tmp_path):
    p = tmp_path / "changed.txt"
    p.write_text("\n\nREADME.md\n\n")
    r = _run("--changed", str(p))
    assert r.returncode == 0


# --------------------------------------------------------------------------
# import cost
# --------------------------------------------------------------------------


def test_module_imports_without_torch():
    env = {**os.environ, "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    code = ("import sys; import trinity.submission.frozen_files; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True, env=env)
    assert out.stdout.strip() == "False", out.stdout


def test_no_regex_backtracking_in_patterns():
    """Patterns are fnmatch globs, not regexes -- keep it that way."""
    for rule in FROZEN_RULES:
        assert not re.search(r"[\[\](){}+^$]", rule.pattern), rule.pattern
