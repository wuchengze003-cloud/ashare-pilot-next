from __future__ import annotations

from datetime import date

import pytest
from ashare_sim_account.cli import _validate_execution_dataset_lineage


def _signal() -> dict:
    return {
        "contract_set": {
            "dataset_manifest_sha256": "a" * 64,
        }
    }


def _manifest(*, as_of: str = "2026-08-18", parent: str = "a" * 64) -> dict:
    return {
        "as_of": as_of,
        "parent_manifest_sha256": parent,
    }


def test_execution_dataset_must_directly_extend_signal_dataset() -> None:
    _validate_execution_dataset_lineage(
        production_signal=_signal(),
        dataset_manifest=_manifest(),
        execution_date=date(2026, 8, 18),
    )


def test_execution_dataset_rejects_unrelated_parent() -> None:
    with pytest.raises(ValueError, match="directly extend"):
        _validate_execution_dataset_lineage(
            production_signal=_signal(),
            dataset_manifest=_manifest(parent="b" * 64),
            execution_date=date(2026, 8, 18),
        )


def test_execution_dataset_rejects_later_manifest() -> None:
    with pytest.raises(ValueError, match="as_of must equal execution_date"):
        _validate_execution_dataset_lineage(
            production_signal=_signal(),
            dataset_manifest=_manifest(as_of="2026-08-19"),
            execution_date=date(2026, 8, 18),
        )
