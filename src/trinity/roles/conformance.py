"""Verify the role prompts still match SPEC §4.4 word-for-word.

``roles/prompts.py`` states the contract in its own module docstring:

    "The instruction text is preserved word-for-word from SPEC §4.4. The {Q} /
    {C_prev} blocks named in the spec are supplied via the user message, so the
    system prompt holds only the stable role contract."

Nothing checked it. ``tests/test_role_prompts.py`` covers plumbing — transcript
rendering, message layout, that the three system prompts differ — and never
compares a character against the document. The role prompts *are* the multi-agent
behaviour: they define what Thinker, Worker and Verifier do. If they drift from
§4.4, every number in ``RESULTS.md`` quietly stops describing the documented
system, and no test fails.

Why a naive equality check does not work
----------------------------------------
SPEC §4.4 publishes one block per role, and each block interleaves the role
contract with the query/transcript scaffold::

    You are the THINKER. Do NOT solve the task end-to-end.
    ... instruction lines ...
    QUERY:
    {Q}
    TRANSCRIPT SO FAR:
    {C_prev}
    Return only your plan/critique.

The implementation deliberately splits that: the instruction lines become the
**system** message, while ``QUERY:``/``TRANSCRIPT SO FAR:`` are assembled into the
**user** message by ``build_messages``. So ``THINKER_SYSTEM`` is the published
block *minus* the scaffold segment — comparing the two directly fails on correct
code, which is why the check was never hand-written.

This module reconstructs the comparison the claim actually makes:

* the published block minus its scaffold segment must equal the shipped system
  prompt, character for character;
* the scaffold labels must appear in the assembled user message, so the parts
  §4.4 moved out are genuinely still delivered.

Known, deliberate divergence
----------------------------
``build_messages`` renders the user turn as ``QUERY:\\n{q}\\n\\nTRANSCRIPT SO
FAR:\\n{transcript}`` — a blank line between the query and the transcript label
that §4.4's block does not have. That is scaffold *formatting*, not instruction
text, so :func:`check` reports it separately rather than failing on it. It is
surfaced instead of normalized away, because "we normalize whitespace" is how a
real drift eventually hides.

Pure stdlib. No torch, no network.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "PLACEHOLDER_LABELS",
    "ROLE_HEADINGS",
    "SPEC_SECTION",
    "ConformanceReport",
    "RoleConformance",
    "check",
    "default_spec_path",
    "expected_system_prompt",
    "parse_spec_prompts",
    "render",
    "split_scaffold",
]

#: The SPEC section that publishes the templates.
SPEC_SECTION = "4.4"

#: Bold headings introducing each role's fenced block, in canonical order.
ROLE_HEADINGS: tuple[str, ...] = ("THINKER", "WORKER", "VERIFIER")

#: The scaffold labels §4.4 places inside each block and the implementation
#: moves into the user message.
PLACEHOLDER_LABELS: tuple[str, ...] = ("QUERY:", "TRANSCRIPT SO FAR:")

#: The placeholder tokens themselves.
_PLACEHOLDERS: tuple[str, ...] = ("{Q}", "{C_prev}")

_SECTION_RE = re.compile(r"^###\s+4\.4\b.*?$(.*?)(?=^###\s)", re.S | re.M)
_BLOCK_RE = re.compile(r"^\*\*(?P<role>[A-Z]+)\*\*\s*\n```(?P<body>.*?)```", re.S | re.M)


def default_spec_path() -> Path:
    """``docs/SPEC.md`` relative to the installed package."""
    return Path(__file__).resolve().parents[3] / "docs" / "SPEC.md"


def parse_spec_prompts(spec_text: str) -> dict[str, str]:
    """Extract §4.4's fenced prompt block for each role.

    Args:
        spec_text: the full contents of ``docs/SPEC.md``.

    Returns:
        ``{"THINKER": block, ...}`` with each block stripped of surrounding
        blank lines but otherwise verbatim.

    Raises:
        ValueError: if §4.4 is absent, or a role's block is missing. Both mean
            the document moved and this checker needs updating -- failing loudly
            beats silently verifying nothing.
    """
    section = _SECTION_RE.search(spec_text)
    if section is None:
        raise ValueError(f"SPEC §{SPEC_SECTION} section not found")
    blocks = {m.group("role"): m.group("body").strip("\n") for m in _BLOCK_RE.finditer(section.group(1))}
    missing = [r for r in ROLE_HEADINGS if r not in blocks]
    if missing:
        raise ValueError(f"SPEC §{SPEC_SECTION} has no fenced block for: {', '.join(missing)}")
    return {r: blocks[r] for r in ROLE_HEADINGS}


def split_scaffold(block: str) -> tuple[str, str]:
    """Split a §4.4 block into (instructions+trailer, scaffold).

    The scaffold is the contiguous ``QUERY: / {Q} / TRANSCRIPT SO FAR: /
    {C_prev}`` run. Everything before it is the role contract; anything after it
    is a trailing instruction (e.g. *"Return only your plan/critique."*) which
    the implementation also keeps in the system prompt.

    Returns:
        ``(contract, scaffold)`` where ``contract`` is the lines outside the
        scaffold joined by newlines, and ``scaffold`` is the removed run.

    Raises:
        ValueError: if the block has no recognizable scaffold.
    """
    lines = block.split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.strip() == PLACEHOLDER_LABELS[0]), None)
    if start is None:
        raise ValueError(f"no {PLACEHOLDER_LABELS[0]!r} line in block")
    end = start
    scaffold_tokens = set(PLACEHOLDER_LABELS) | set(_PLACEHOLDERS)
    while end < len(lines) and lines[end].strip() in scaffold_tokens:
        end += 1
    if end == start:
        raise ValueError("empty scaffold segment")
    contract = lines[:start] + lines[end:]
    return "\n".join(contract).strip("\n"), "\n".join(lines[start:end])


def expected_system_prompt(block: str) -> str:
    """The system prompt §4.4 implies for a role: the block minus its scaffold."""
    contract, _scaffold = split_scaffold(block)
    return contract


@dataclass(frozen=True)
class RoleConformance:
    """Whether one role's shipped system prompt matches SPEC §4.4."""

    role: str
    ok: bool
    expected: str
    actual: str
    scaffold_labels_delivered: bool

    @property
    def diff_lines(self) -> list[str]:
        """A unified-ish line diff, empty when conformant."""
        if self.ok:
            return []
        exp, act = self.expected.split("\n"), self.actual.split("\n")
        out: list[str] = []
        for i in range(max(len(exp), len(act))):
            e = exp[i] if i < len(exp) else "<missing>"
            a = act[i] if i < len(act) else "<missing>"
            if e != a:
                out.append(f"    line {i + 1}:")
                out.append(f"      spec: {e!r}")
                out.append(f"      code: {a!r}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "ok": self.ok,
            "scaffold_labels_delivered": self.scaffold_labels_delivered,
            "expected": self.expected,
            "actual": self.actual,
            "diff": self.diff_lines,
        }


@dataclass(frozen=True)
class ConformanceReport:
    """Per-role results plus the overall verdict."""

    roles: list[RoleConformance] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok and r.scaffold_labels_delivered for r in self.roles)

    @property
    def drifted(self) -> list[str]:
        return [r.role for r in self.roles if not r.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "drifted": self.drifted,
            "roles": [r.to_dict() for r in self.roles],
            "notes": list(self.notes),
        }


def _assembled_user_message() -> str:
    """The user turn ``build_messages`` produces, with stand-in content."""
    from .prompts import build_messages
    from ..types import Role

    msgs = build_messages(Role.WORKER, "Q?", [])
    return str(msgs[1]["content"])


def check(
    spec_text: str | None = None,
    *,
    spec_path: Path | str | None = None,
    shipped: Mapping[str, str] | None = None,
) -> ConformanceReport:
    """Compare the shipped system prompts against SPEC §4.4.

    Args:
        spec_text: SPEC contents. When ``None``, read from ``spec_path`` or
            :func:`default_spec_path`.
        spec_path: where to read the SPEC from.
        shipped: ``{"THINKER": prompt, ...}`` to check instead of the live
            constants. Used by tests to exercise drift without editing the repo.

    Returns:
        A :class:`ConformanceReport`.

    Raises:
        ValueError: if the SPEC section or a role block cannot be parsed.
        FileNotFoundError: if the SPEC file is missing.
    """
    if spec_text is None:
        path = Path(spec_path) if spec_path is not None else default_spec_path()
        if not path.exists():
            raise FileNotFoundError(str(path))
        spec_text = path.read_text()

    if shipped is None:
        from .prompts import THINKER_SYSTEM, VERIFIER_SYSTEM, WORKER_SYSTEM

        shipped = {
            "THINKER": THINKER_SYSTEM,
            "WORKER": WORKER_SYSTEM,
            "VERIFIER": VERIFIER_SYSTEM,
        }

    blocks = parse_spec_prompts(spec_text)
    user_message = _assembled_user_message()

    roles: list[RoleConformance] = []
    for role in ROLE_HEADINGS:
        expected = expected_system_prompt(blocks[role])
        actual = str(shipped.get(role, ""))
        delivered = all(label in user_message for label in PLACEHOLDER_LABELS)
        roles.append(
            RoleConformance(
                role=role,
                ok=expected == actual,
                expected=expected,
                actual=actual,
                scaffold_labels_delivered=delivered,
            )
        )

    notes: list[str] = []
    spec_scaffold = split_scaffold(blocks["THINKER"])[1]
    if spec_scaffold.replace("{Q}", "Q?") not in user_message:
        notes.append(
            "user-message scaffold is not byte-identical to §4.4's block "
            "(build_messages inserts a blank line before 'TRANSCRIPT SO FAR:'); "
            "this is scaffold formatting, not instruction text, and is reported "
            "rather than failed on."
        )
    return ConformanceReport(roles=roles, notes=notes)


def render(report: ConformanceReport) -> str:
    """Human-readable conformance report."""
    lines = ["SPEC §4.4 prompt conformance"]
    for r in report.roles:
        mark = "ok" if r.ok else "DRIFTED"
        lines.append(f"  {r.role:9s} {mark}")
        lines.extend(r.diff_lines)
        if not r.scaffold_labels_delivered:
            lines.append("      scaffold labels MISSING from the assembled user message")
    for note in report.notes:
        lines.append(f"  note: {note}")
    lines.append(
        "  verdict: prompts match the document"
        if report.ok
        else f"  verdict: DRIFT in {', '.join(report.drifted) or 'scaffold delivery'}"
    )
    return "\n".join(lines)
