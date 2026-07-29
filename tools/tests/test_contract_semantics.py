from pathlib import Path

import pytest

from tools.validate_contracts import validate_semantics

DOCUMENT = Path("synthetic-document.json")


def test_flat_signal_cannot_have_nonzero_target() -> None:
    signal = {
        "state": "FLAT",
        "champion": None,
        "previous_signal_sha256": None,
        "target_positions": [{"symbol": "600000.SH", "target_weight": 0.1}],
    }

    with pytest.raises(ValueError, match="FLAT requires zero target weights"):
        validate_semantics("production-signal", signal, DOCUMENT)


def test_hold_signal_requires_previous_signal() -> None:
    signal = {
        "state": "HOLD",
        "champion": None,
        "previous_signal_sha256": None,
        "target_positions": [],
    }

    with pytest.raises(ValueError, match="HOLD requires a previous signal"):
        validate_semantics("production-signal", signal, DOCUMENT)


def test_universe_rejects_duplicate_symbols() -> None:
    universe = {
        "as_of": "2026-01-02",
        "members": [
            {
                "symbol": "600000.SH",
                "valid_from": "2026-01-01",
                "valid_to": None,
            },
            {
                "symbol": "600000.SH",
                "valid_from": "2026-01-01",
                "valid_to": None,
            },
        ],
    }

    with pytest.raises(ValueError, match="duplicate universe symbol"):
        validate_semantics("universe", universe, DOCUMENT)


def test_experiment_windows_must_not_overlap() -> None:
    experiment = {
        "windows": {
            "development": {"start": "2025-01-01", "end": "2025-06-30"},
            "validation": {"start": "2025-06-30", "end": "2025-09-30"},
            "final": {"start": "2025-10-01", "end": "2025-12-31"},
        }
    }

    with pytest.raises(ValueError, match="ordered and disjoint"):
        validate_semantics("experiment-config", experiment, DOCUMENT)
