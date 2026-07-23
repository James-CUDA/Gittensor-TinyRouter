"""SPEC §4.4 prompt conformance.

The load-bearing test is ``test_shipped_prompts_match_the_spec``: it reads the
repo's own ``docs/SPEC.md`` and asserts the shipped system prompts reproduce it.
Everything else exercises the machinery that makes that assertion meaningful --
in particular that a real drift would actually be caught.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.roles.conformance import (
    PLACEHOLDER_LABELS,
    ROLE_HEADINGS,
    ConformanceReport,
    check,
    default_spec_path,
    expected_system_prompt,
    parse_spec_prompts,
    render,
    split_scaffold,
)
from trinity.roles.prompts import THINKER_SYSTEM, VERIFIER_SYSTEM, WORKER_SYSTEM

_REPO = Path(__file__).resolve().parents[1]
_SPEC = _REPO / "docs" / "SPEC.md"
_SCRIPT = _REPO / "scripts" / "prompt_conformance_report.py"
_SRC = str(_REPO / "src")

SHIPPED = {"THINKER": THINKER_SYSTEM, "WORKER": WORKER_SYSTEM, "VERIFIER": VERIFIER_SYSTEM}

_MINI_SPEC = """\
### 4.3 Something else
text

### 4.4 Prompt templates
intro line

**THINKER**
```
You are the THINKER. Think.
QUERY:
{Q}
TRANSCRIPT SO FAR:
{C_prev}
Return only your plan.
```

**WORKER**
```
You are the WORKER. Work.
QUERY:
{Q}
TRANSCRIPT SO FAR:
{C_prev}
Return your solution work.
```

**VERIFIER**
```
You are the VERIFIER. Verify.
QUERY:
{Q}
TRANSCRIPT SO FAR:
{C_prev}
```

### 4.5 Next section
more text
"""


# --------------------------------------------------------------------------
# the claim, checked against the real document
# --------------------------------------------------------------------------


def test_shipped_prompts_match_the_spec():
    """roles/prompts.py says 'word-for-word from SPEC §4.4'. Verify it."""
    report = check()
    assert report.ok, render(report)


def test_every_role_is_covered():
    report = check()
    assert [r.role for r in report.roles] == list(ROLE_HEADINGS)


def test_the_repo_spec_is_where_the_module_looks():
    assert default_spec_path() == _SPEC
    assert _SPEC.exists()


def test_scaffold_labels_are_delivered_in_the_user_message():
    """§4.4's QUERY/TRANSCRIPT blocks must still reach the model, via the user turn."""
    for r in check().roles:
        assert r.scaffold_labels_delivered, r.role


def test_the_known_scaffold_divergence_is_reported_not_hidden():
    """build_messages adds a blank line §4.4 lacks; it must be surfaced."""
    notes = check().notes
    assert any("blank line" in n for n in notes), notes


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_parse_extracts_all_three_blocks():
    blocks = parse_spec_prompts(_MINI_SPEC)
    assert set(blocks) == set(ROLE_HEADINGS)
    assert blocks["THINKER"].startswith("You are the THINKER.")


def test_parse_stops_at_the_next_section():
    """A block in §4.5 must not leak into §4.4's set."""
    blocks = parse_spec_prompts(_MINI_SPEC)
    assert "more text" not in "".join(blocks.values())


def test_parse_rejects_a_spec_without_the_section():
    with pytest.raises(ValueError, match="not found"):
        parse_spec_prompts("### 9.9 Nothing here\ntext\n\n### 10 End\n")


def test_parse_rejects_a_missing_role_block():
    trimmed = _MINI_SPEC.replace("**VERIFIER**", "**SOMETHINGELSE**")
    with pytest.raises(ValueError, match="VERIFIER"):
        parse_spec_prompts(trimmed)


def test_parse_uses_the_real_spec_too():
    blocks = parse_spec_prompts(_SPEC.read_text())
    assert set(blocks) == set(ROLE_HEADINGS)
    for body in blocks.values():
        assert "{Q}" in body and "{C_prev}" in body


# --------------------------------------------------------------------------
# scaffold splitting
# --------------------------------------------------------------------------


def test_split_removes_only_the_scaffold_run():
    block = parse_spec_prompts(_MINI_SPEC)["THINKER"]
    contract, scaffold = split_scaffold(block)
    assert contract == "You are the THINKER. Think.\nReturn only your plan."
    assert scaffold.splitlines() == ["QUERY:", "{Q}", "TRANSCRIPT SO FAR:", "{C_prev}"]


def test_split_keeps_a_trailing_instruction():
    """The line after {C_prev} belongs to the system prompt, not the scaffold."""
    contract, _ = split_scaffold(parse_spec_prompts(_MINI_SPEC)["WORKER"])
    assert contract.endswith("Return your solution work.")


def test_split_handles_a_block_with_no_trailer():
    contract, _ = split_scaffold(parse_spec_prompts(_MINI_SPEC)["VERIFIER"])
    assert contract == "You are the VERIFIER. Verify."


def test_split_rejects_a_block_without_a_scaffold():
    with pytest.raises(ValueError, match="QUERY:"):
        split_scaffold("You are the THINKER.\nNo scaffold here.")


def test_expected_prompt_is_the_block_minus_scaffold():
    block = parse_spec_prompts(_MINI_SPEC)["THINKER"]
    assert expected_system_prompt(block) == split_scaffold(block)[0]


@pytest.mark.parametrize("role", ROLE_HEADINGS)
def test_expected_matches_the_shipped_constant_per_role(role):
    block = parse_spec_prompts(_SPEC.read_text())[role]
    assert expected_system_prompt(block) == SHIPPED[role]


# --------------------------------------------------------------------------
# drift is actually detected
# --------------------------------------------------------------------------


def test_a_reworded_prompt_is_caught():
    drifted = dict(SHIPPED, WORKER=WORKER_SYSTEM.replace("concrete", "some"))
    report = check(shipped=drifted)
    assert report.ok is False
    assert report.drifted == ["WORKER"]


def test_a_single_character_change_is_caught():
    drifted = dict(SHIPPED, THINKER=THINKER_SYSTEM + " ")
    assert check(shipped=drifted).ok is False


def test_a_dropped_line_is_caught():
    lines = VERIFIER_SYSTEM.split("\n")
    drifted = dict(SHIPPED, VERIFIER="\n".join(lines[:-1]))
    report = check(shipped=drifted)
    assert report.ok is False
    assert "VERIFIER" in report.drifted


def test_the_diff_names_the_offending_line():
    drifted = dict(SHIPPED, WORKER=WORKER_SYSTEM.replace("WORKER", "WORKERR"))
    report = check(shipped=drifted)
    diff = next(r for r in report.roles if r.role == "WORKER").diff_lines
    assert diff and any("WORKERR" in ln for ln in diff)


def test_a_conformant_role_has_no_diff():
    for r in check().roles:
        assert r.diff_lines == []


def test_report_serializes():
    payload = check().to_dict()
    json.dumps(payload)
    assert payload["ok"] is True
    assert len(payload["roles"]) == 3


def test_render_reports_drift_visibly():
    drifted = dict(SHIPPED, THINKER="totally different")
    out = render(check(shipped=drifted))
    assert "DRIFTED" in out and "THINKER" in out


def test_render_reports_conformance():
    assert "prompts match the document" in render(check())


def test_empty_report_is_ok():
    assert ConformanceReport().ok is True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _run(*args):
    env = {**os.environ, "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, env=env
    )


def test_cli_passes_on_the_real_spec():
    r = _run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "prompt conformance" in r.stdout


def test_cli_json_output():
    r = _run("--json")
    assert r.returncode == 0
    assert json.loads(r.stdout)["ok"] is True


def test_cli_flags_drift_against_a_modified_spec(tmp_path):
    spec = tmp_path / "SPEC.md"
    spec.write_text(_SPEC.read_text().replace(
        "You are the WORKER. Make concrete progress toward the final answer.",
        "You are the WORKER. Make vague progress toward the final answer.",
    ))
    r = _run("--spec", str(spec))
    assert r.returncode == 1
    assert "DRIFTED" in r.stdout


def test_cli_missing_spec_is_graceful(tmp_path):
    r = _run("--spec", str(tmp_path / "nope.md"))
    assert r.returncode == 2
    assert "no such file" in r.stderr


def test_cli_unparseable_spec_is_graceful(tmp_path):
    spec = tmp_path / "SPEC.md"
    spec.write_text("# nothing resembling section 4.4\n")
    r = _run("--spec", str(spec))
    assert r.returncode == 2
    assert "could not parse" in r.stderr


# --------------------------------------------------------------------------
# import cost
# --------------------------------------------------------------------------


def test_module_imports_without_torch():
    env = {**os.environ, "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    code = ("import sys; import trinity.roles.conformance; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True, env=env)
    assert out.stdout.strip() == "False", out.stdout


def test_placeholder_labels_are_what_the_spec_uses():
    body = parse_spec_prompts(_SPEC.read_text())["THINKER"]
    for label in PLACEHOLDER_LABELS:
        assert label in body
