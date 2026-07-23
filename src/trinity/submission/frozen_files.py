"""Frozen-file tamper detection for submission PRs (COMPETITION_RULES §Frozen files).

``docs/COMPETITION_RULES.md`` tabulates what each cheat attempt runs into. Every
row names an automated gate — *"Gate 3 rejects"*, *"Gate 4 rejects"*, *"Gate 5
hard-rejects"* — with exactly one exception:

    | Modifying frozen files in a submission PR | Rejected by maintainer |

That is the only row in the anti-cheat table whose enforcement is a human reading
a diff. The frozen list itself is published one section earlier (§"Frozen files
(do not touch)"), so the check is entirely mechanical; nothing in ``submission/``
or ``scripts/repo_governance/`` performs it.

This module performs it, over a list of changed paths.

Advisory, not a gate
--------------------
It is wired into :data:`~trinity.submission.gates.OFFLINE_ADVISORIES`, the
"report but never reject" tier from issue #208 — not into ``OFFLINE_GATES``.
Two reasons:

* ``4bb03a7`` deliberately relaxed the gate chain "to attract miners, not repel
  them"; a *new blocking* check runs against that, an advisory costs a miner
  nothing.
* The check's input is a changed-file list, which the preflight context cannot
  always supply (a local miner running preflight has no PR diff). A gate that
  silently passes when its input is missing is a gate in name only; an advisory
  that stays quiet is behaving exactly as documented.

Promoting it to a blocking gate is a one-line change and the maintainer's call.

Path matching
-------------
The published list mixes three kinds of entry, all supported here:

* exact repo-relative paths (``scripts/pr_eval.py``)
* a directory glob (``src/trinity/submission/*.py`` — files directly inside, not
  recursive, matching the shell semantics the table implies)
* an environment-rooted subtree (``$TINYROUTER_BENCHMARK_DIR/``), resolved from
  the environment at call time so a run that does not set it simply has no
  benchmark rule to match

Paths are normalized (``./`` prefixes and backslash separators) before matching,
so a caller can pass ``git diff --name-only`` output verbatim.
"""
from __future__ import annotations

import fnmatch
import os
import posixpath
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "BENCHMARK_DIR_ENV",
    "FROZEN_RULES",
    "FrozenMatch",
    "FrozenRule",
    "audit_frozen_files",
    "frozen_violations",
    "match_frozen",
    "normalize_path",
]

#: Environment variable naming the encrypted hidden-benchmark directory. The
#: published rule is "any file under ``$TINYROUTER_BENCHMARK_DIR/``".
BENCHMARK_DIR_ENV = "TINYROUTER_BENCHMARK_DIR"


@dataclass(frozen=True)
class FrozenRule:
    """One row of the published frozen-file table."""

    pattern: str
    reason: str

    @property
    def is_glob(self) -> bool:
        return "*" in self.pattern or "?" in self.pattern


@dataclass(frozen=True)
class FrozenMatch:
    """A changed path that matched a frozen rule."""

    path: str
    rule: FrozenRule

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "pattern": self.rule.pattern, "reason": self.rule.reason}


#: The frozen list, transcribed from ``docs/COMPETITION_RULES.md`` §"Frozen files
#: (do not touch)". ``tests/test_frozen_files.py`` parses that table and asserts
#: this tuple matches it exactly, so the two cannot drift -- the document stays
#: the single source of truth without this module having to parse markdown at
#: runtime.
#:
#: The ``$TINYROUTER_BENCHMARK_DIR/`` row is handled separately (see
#: :func:`match_frozen`) because its location is only known from the environment.
FROZEN_RULES: tuple[FrozenRule, ...] = (
    FrozenRule("scripts/pr_eval.py", "The evaluation orchestrator"),
    FrozenRule("scripts/build_benchmark.py", "The hidden benchmark builder"),
    FrozenRule("src/trinity/orchestration/reward.py", "The shared grader"),
    FrozenRule("src/trinity/submission/*.py", "The anti-cheat gates"),
    FrozenRule("src/trinity/submission/constants.py", "Frozen constants (pool, margin, params)"),
    FrozenRule("leaderboard.json", "King-of-the-hill state (maintainer-only writes)"),
)


def normalize_path(path: str) -> str:
    """Normalize a changed path for matching.

    Converts backslashes to ``/``, strips a leading ``./``, and collapses ``..``
    / duplicate separators. The result is repo-relative and comparable with the
    published patterns.
    """
    p = str(path).strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    return posixpath.normpath(p) if p else ""


def _benchmark_rule() -> FrozenRule | None:
    """The ``$TINYROUTER_BENCHMARK_DIR`` rule, if the variable is set."""
    root = os.environ.get(BENCHMARK_DIR_ENV, "").strip()
    if not root:
        return None
    return FrozenRule(
        f"{normalize_path(root)}/**", "Encrypted hidden benchmarks"
    )


def _under(path: str, root: str) -> bool:
    """Whether ``path`` lies inside directory ``root`` (not merely prefixed).

    ``benchmarks_old/x`` must NOT match a root of ``benchmarks``.
    """
    if not root:
        return False
    return path == root or path.startswith(root + "/")


def match_frozen(path: str) -> FrozenMatch | None:
    """Return the frozen rule ``path`` violates, or ``None``.

    The most specific rule wins: an exact entry is preferred over a glob, so
    ``src/trinity/submission/constants.py`` reports its own "Frozen constants"
    reason rather than the broader "anti-cheat gates" one.
    """
    norm = normalize_path(path)
    if not norm:
        return None

    bench = _benchmark_rule()
    if bench is not None:
        root = bench.pattern[: -len("/**")]
        if _under(norm, root):
            return FrozenMatch(path=norm, rule=bench)

    exact = [r for r in FROZEN_RULES if not r.is_glob and r.pattern == norm]
    if exact:
        return FrozenMatch(path=norm, rule=exact[0])

    for rule in FROZEN_RULES:
        if rule.is_glob and _glob_matches(norm, rule.pattern):
            return FrozenMatch(path=norm, rule=rule)
    return None


def _glob_matches(path: str, pattern: str) -> bool:
    """Shell-style glob match in which ``*`` does NOT cross a ``/``.

    ``fnmatch`` alone is wrong here: its ``*`` spans separators, so
    ``src/trinity/submission/*.py`` would also match
    ``src/trinity/submission/nested/deep.py``. The published pattern is written
    in shell semantics and means "the ``.py`` files directly inside
    ``submission/``", so the directory part is compared exactly and only the
    final component is globbed.

    Consequence, stated rather than silently widened: a hypothetical nested
    package under ``submission/`` would not match the published pattern. Widening
    the rule is a documentation change, not a matcher change.
    """
    p_dir, _, p_base = path.rpartition("/")
    g_dir, _, g_base = pattern.rpartition("/")
    return p_dir == g_dir and fnmatch.fnmatch(p_base, g_base)


def frozen_violations(paths: Iterable[str]) -> tuple[FrozenMatch, ...]:
    """Every changed path that touches a frozen file, in input order.

    Duplicate paths are reported once.
    """
    seen: set[str] = set()
    out: list[FrozenMatch] = []
    for p in paths:
        m = match_frozen(p)
        if m is None or m.path in seen:
            continue
        seen.add(m.path)
        out.append(m)
    return tuple(out)


def audit_frozen_files(paths: Sequence[str] | None) -> str | None:
    """Advisory message naming every frozen file touched, or ``None`` if clean.

    Returns ``None`` for an empty or absent path list: with no diff to inspect
    there is nothing to report, and the advisory stays silent rather than
    guessing.
    """
    if not paths:
        return None
    hits = frozen_violations(paths)
    if not hits:
        return None
    detail = "; ".join(f"{m.path} ({m.rule.reason})" for m in hits)
    plural = "s" if len(hits) > 1 else ""
    return (
        f"{len(hits)} frozen file{plural} modified: {detail}. "
        "COMPETITION_RULES.md lists these as frozen — modifying them in a "
        "submission PR is cheating."
    )
