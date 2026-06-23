#!/usr/bin/env python3
"""
provenance.py — tool-repo-anchored helpers shared across the toolchain.

Two concerns, one anchor. Everything here is keyed off THIS module's __file__ (which lives in
T/tools/), so the answers are invariant to the current working directory:

  * tool_provenance() — the git SHA + dirty flag of the *tool* repo (T,
    rtx3090-ai-training-tools), the repo that holds the code + eval inputs that produced a
    result. NEVER the SHA of the cwd. Post-split the tools run with cwd = R
    (rtx3090-ai-training) because results are written into R; a cwd-based git read would pin the
    data repo instead of the code, re-introducing the dirty-by-sibling friction the split
    removed. Anchoring to __file__ records T's SHA from anywhere.

  * resolve_input() / tool_repo_root() — locate the bundled eval inputs (prompts/, probes/,
    rubrics/) that ship in T alongside the tools, so a caller can pass a bare repo-relative path
    (e.g. "prompts/operator-copilot-rca-system-prompt.md") and it resolves whether or not the
    cwd happens to contain it.

The provenance keys (`tool_git_*`) name the fact honestly: this is the tool repo's state, and a
dirty flag here is the discipline gate that now matters — R being dirty at capture time is
expected and irrelevant to what produced the result.

Usage (sibling import; tools run as `python3 tools/<tool>.py`, so tools/ is sys.path[0]):

    from provenance import tool_provenance, resolve_input
    prov = tool_provenance()                  # {"tool_git_sha": "...", "tool_git_dirty": False}
    sp   = resolve_input(args.system_prompt)   # Path, found in cwd or in the tool repo
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

# The tool repo root is wherever this file lives (T/tools/...). `git -C` walks up to the
# enclosing .git, so the exact subdir is immaterial — only that it is inside T, never cwd.
_TOOL_REPO_DIR = Path(__file__).resolve().parent


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_TOOL_REPO_DIR), *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort; never fail a run over it
        return ""


def tool_provenance() -> dict[str, Any]:
    """Git SHA + dirty flag of the tool repo this module lives in (never cwd).

    Returns {"tool_git_sha": <sha|None>, "tool_git_dirty": <bool|None>}. Both are None if
    the tool repo has no HEAD yet (e.g. T freshly created with no initial commit) or git is
    otherwise unavailable — in which case commit T and re-run to get a real SHA.
    """
    sha = _git(["rev-parse", "HEAD"])
    if not sha:
        return {"tool_git_sha": None, "tool_git_dirty": None}
    return {"tool_git_sha": sha, "tool_git_dirty": bool(_git(["status", "--porcelain"]))}


def tool_repo_root() -> Path:
    """Filesystem root of the tool repo (T) — the parent of tools/."""
    return _TOOL_REPO_DIR.parent


def resolve_input(path_str: str) -> Path:
    """Resolve a bundled-input path (a prompt / probe / rubric file).

    Resolution order: an absolute path is used as given; a relative path is used as-is if it
    exists relative to the CWD, otherwise it is resolved against the tool repo root — so the
    eval inputs that ship in T (prompts/, probes/, rubrics/) resolve from any working directory
    without the caller spelling out the tool-repo path. If neither location exists, the path is
    returned unchanged so the caller's not-found error names exactly what was requested.
    """
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    candidate = tool_repo_root() / p
    return candidate if candidate.exists() else p
