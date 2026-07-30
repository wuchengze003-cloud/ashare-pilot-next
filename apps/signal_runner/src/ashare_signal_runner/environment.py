"""Resolve runtime identity from the checked-out repository."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

GIT_SHA_PATTERN = re.compile(r"^[a-f0-9]{40}$")


@dataclass(frozen=True)
class RuntimeEnvironment:
    git_sha: str
    lockfile_sha256: str


def resolve_runtime_environment(
    *,
    repository_root: Path,
    deployment_git_sha: str,
) -> RuntimeEnvironment:
    """Verify the deployment claim and hash the actual workspace lockfile."""
    if not GIT_SHA_PATTERN.fullmatch(deployment_git_sha):
        raise ValueError("deployment_git_sha must be a lowercase 40-character Git SHA")
    resolved_root = repository_root.resolve(strict=True)
    result = subprocess.run(
        ["git", "-C", str(resolved_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_git_sha = result.stdout.strip()
    if actual_git_sha != deployment_git_sha:
        raise ValueError("deployment Git SHA does not match the checked-out repository")

    lockfile = resolved_root / "uv.lock"
    if lockfile.is_symlink():
        raise ValueError("uv.lock cannot be a symlink")
    resolved_lockfile = lockfile.resolve(strict=True)
    if not resolved_lockfile.is_relative_to(resolved_root):
        raise ValueError("uv.lock escapes the repository root")
    lockfile_sha256 = hashlib.sha256(resolved_lockfile.read_bytes()).hexdigest()
    return RuntimeEnvironment(
        git_sha=actual_git_sha,
        lockfile_sha256=lockfile_sha256,
    )
