from __future__ import annotations

from pathlib import Path

import pytest
from ashare_signal_runner import canonical_json_sha256
from ashare_sim_account import commit_account_state, load_current_account_state


def _state(*, sequence: int, previous: dict | None) -> dict:
    previous_hash = canonical_json_sha256(previous) if previous is not None else None
    return {
        "contract_id": "simulated-account-state",
        "schema_version": "1.0.0",
        "state_id": f"paper-main-2026-08-{17 + sequence:02d}-seq{sequence}",
        "account_id": "paper-main",
        "account_sequence": sequence,
        "as_of": f"2026-08-{17 + sequence:02d}",
        "generated_at": f"2026-08-{17 + sequence:02d}T09:31:00+08:00",
        "previous_state_sha256": previous_hash,
        "source_signal": {"sequence": sequence},
        "initial_cash": 100000.0,
        "cash": 100000.0,
        "holdings": [],
    }


def test_account_state_publication_is_append_only_and_advances_the_head(tmp_path: Path) -> None:
    first = _state(sequence=1, previous=None)
    first_dir = commit_account_state(runtime_root=tmp_path, document=first)
    assert (first_dir / "COMMITTED").is_file()
    assert load_current_account_state(runtime_root=tmp_path, account_id="paper-main") == first

    second = _state(sequence=2, previous=first)
    second_dir = commit_account_state(runtime_root=tmp_path, document=second)
    assert second_dir != first_dir
    assert (first_dir / "state.json").is_file()
    assert load_current_account_state(runtime_root=tmp_path, account_id="paper-main") == second


def test_concurrent_or_stale_previous_state_is_rejected(tmp_path: Path) -> None:
    first = _state(sequence=1, previous=None)
    commit_account_state(runtime_root=tmp_path, document=first)

    stale = _state(sequence=2, previous=None)
    with pytest.raises(ValueError, match="changed concurrently"):
        commit_account_state(runtime_root=tmp_path, document=stale)


def test_symlink_account_head_is_rejected(tmp_path: Path) -> None:
    account_root = tmp_path / "sim-accounts" / "paper-main"
    account_root.mkdir(parents=True)
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    (account_root / "current-state.json").symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        load_current_account_state(runtime_root=tmp_path, account_id="paper-main")


def test_symlink_account_directory_is_rejected(tmp_path: Path) -> None:
    accounts = tmp_path / "sim-accounts"
    accounts.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (accounts / "paper-main").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain a symlink"):
        commit_account_state(
            runtime_root=tmp_path,
            document=_state(sequence=1, previous=None),
        )


def test_symlink_commit_marker_is_rejected(tmp_path: Path) -> None:
    state = _state(sequence=1, previous=None)
    run_dir = commit_account_state(runtime_root=tmp_path, document=state)
    commit_marker = run_dir / "COMMITTED"
    target = tmp_path / "outside-hash"
    target.write_text(canonical_json_sha256(state), encoding="ascii")
    commit_marker.unlink()
    commit_marker.symlink_to(target)

    with pytest.raises(ValueError, match="not committed"):
        load_current_account_state(runtime_root=tmp_path, account_id="paper-main")
