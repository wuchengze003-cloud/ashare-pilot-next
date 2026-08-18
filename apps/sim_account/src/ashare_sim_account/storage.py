"""Atomic append-only publication for simulated account states."""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ashare_signal_runner import canonical_json_bytes, canonical_json_sha256

ACCOUNT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
STATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,159}$")


def _account_root(runtime_root: Path, account_id: str) -> Path:
    if not ACCOUNT_ID.fullmatch(account_id):
        raise ValueError("invalid simulated account id")
    runtime = Path(runtime_root)
    accounts = runtime / "sim-accounts"
    unresolved_root = accounts / account_id
    if accounts.is_symlink() or unresolved_root.is_symlink():
        raise ValueError("simulated account path must not contain a symlink")
    root = runtime.resolve() / "sim-accounts" / account_id
    candidate = root.resolve(strict=False)
    expected_parent = (Path(runtime_root).resolve() / "sim-accounts").resolve(strict=False)
    if candidate.parent != expected_parent:
        raise ValueError("simulated account path escaped runtime root")
    return candidate


def load_current_account_state(
    *,
    runtime_root: Path,
    account_id: str,
) -> Mapping[str, Any] | None:
    root = _account_root(runtime_root, account_id)
    head_path = root / "current-state.json"
    if head_path.is_symlink():
        raise ValueError("simulated account head must not be a symlink")
    if not head_path.exists():
        return None
    head_bytes = head_path.read_bytes()
    import json

    head = json.loads(head_bytes)
    if canonical_json_bytes(head) != head_bytes:
        raise ValueError("simulated account head is not canonical")
    if head.get("account_id") != account_id:
        raise ValueError("simulated account head account_id mismatch")
    state_id = str(head["state_id"])
    if not STATE_ID.fullmatch(state_id):
        raise ValueError("invalid simulated account state id in head")
    run_dir = root / "runs" / state_id
    commit_path = run_dir / "COMMITTED"
    if run_dir.is_symlink() or commit_path.is_symlink() or not commit_path.is_file():
        raise ValueError("simulated account run is not committed")
    state_path = run_dir / "state.json"
    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError("simulated account state must be a regular file")
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    if canonical_json_bytes(state) != state_bytes:
        raise ValueError("simulated account state is not canonical")
    if canonical_json_sha256(state) != head["state_sha256"]:
        raise ValueError("simulated account head hash mismatch")
    committed_hash = commit_path.read_text(encoding="ascii")
    if committed_hash != head["state_sha256"]:
        raise ValueError("simulated account commit marker hash mismatch")
    if state.get("previous_state_sha256") != head.get("previous_state_sha256"):
        raise ValueError("simulated account previous-state binding mismatch")
    return state


def commit_account_state(
    *,
    runtime_root: Path,
    document: Mapping[str, Any],
) -> Path:
    """Commit one immutable run, then atomically advance the account head."""
    account_id = str(document["account_id"])
    root = _account_root(runtime_root, account_id)
    runs_root = root / "runs"
    root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".publish.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = load_current_account_state(runtime_root=runtime_root, account_id=account_id)
        expected_previous = canonical_json_sha256(current) if current is not None else None
        if document["previous_state_sha256"] != expected_previous:
            raise ValueError("simulated account previous state changed concurrently")
        expected_sequence = int(current["account_sequence"]) + 1 if current is not None else 1
        if int(document["account_sequence"]) != expected_sequence:
            raise ValueError("simulated account sequence must advance by exactly one")

        state_id = str(document["state_id"])
        if not STATE_ID.fullmatch(state_id):
            raise ValueError("invalid simulated account state id")
        final_dir = runs_root / state_id
        state_bytes = canonical_json_bytes(document)
        state_sha256 = canonical_json_sha256(document)
        if final_dir.is_symlink():
            raise ValueError("simulated account run must not be a symlink")
        if final_dir.exists():
            existing_state = final_dir / "state.json"
            existing_commit = final_dir / "COMMITTED"
            if existing_state.is_symlink() or existing_commit.is_symlink():
                raise ValueError("existing simulated account run contains a symlink")
            existing = existing_state.read_bytes()
            if existing != state_bytes or not existing_commit.is_file():
                raise ValueError("conflicting simulated account state id")
        else:
            staging = runs_root / f".{state_id}.staging-{os.getpid()}"
            if staging.exists():
                raise ValueError("simulated account staging path already exists")
            staging.mkdir()
            (staging / "state.json").write_bytes(state_bytes)
            (staging / "COMMITTED").write_text(state_sha256, encoding="ascii")
            os.replace(staging, final_dir)

        head = {
            "contract_id": "simulated-account-head",
            "schema_version": "1.0.0",
            "account_id": account_id,
            "account_sequence": int(document["account_sequence"]),
            "state_id": state_id,
            "state_sha256": state_sha256,
            "previous_state_sha256": expected_previous,
            "as_of": str(document["as_of"]),
            "generated_at": str(document["generated_at"]),
        }
        temporary_head = root / f".current-state-{os.getpid()}.json"
        temporary_head.write_bytes(canonical_json_bytes(head))
        os.replace(temporary_head, root / "current-state.json")
        return final_dir
