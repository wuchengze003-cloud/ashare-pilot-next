import copy
import json
from pathlib import Path

import pytest

from tools.validate_contracts import validate_semantics

DOCUMENT = Path("synthetic-document.json")
ROOT = Path(__file__).resolve().parents[2]


def cost_model() -> dict[str, object]:
    return json.loads(
        (ROOT / "contracts/examples/cost-model.example.json").read_text(encoding="utf-8")
    )


def contract_set(*, champion_sha256: str | None = None) -> dict[str, str | None]:
    return {
        "dataset_manifest_sha256": "a" * 64,
        "dataset_snapshot_sha256": "b" * 64,
        "universe_snapshot_sha256": "8" * 64,
        "champion_sha256": champion_sha256,
        "cost_model_sha256": "c" * 64,
        "market_rules_sha256": "d" * 64,
        "execution_policy_sha256": "e" * 64,
        "portfolio_risk_sha256": "f" * 64,
        "code_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "lockfile_sha256": "3" * 64,
    }


def signal_document(
    *,
    state: str,
    targets: list[dict[str, object]],
    previous: str | None = None,
) -> dict[str, object]:
    return {
        "state": state,
        "as_of": "2026-01-02",
        "latest_complete_date": "2026-01-02",
        "generated_at": "2026-01-02T08:00:00Z",
        "champion": None,
        "previous_signal_sha256": previous,
        "target_positions": targets,
        "contract_set": contract_set(),
    }


def test_flat_signal_cannot_have_nonzero_target() -> None:
    signal = signal_document(
        state="FLAT",
        targets=[{"symbol": "600000.SH", "target_weight": 0.1}],
    )

    with pytest.raises(ValueError, match="FLAT requires an empty target"):
        validate_semantics("production-signal", signal, DOCUMENT)


def test_hold_signal_requires_previous_signal() -> None:
    signal = signal_document(state="HOLD", targets=[])

    with pytest.raises(ValueError, match="HOLD requires a previous signal"):
        validate_semantics("production-signal", signal, DOCUMENT)


def test_reduce_only_signal_requires_previous_signal() -> None:
    signal = signal_document(state="REDUCE_ONLY", targets=[])

    with pytest.raises(ValueError, match="REDUCE_ONLY requires a previous signal"):
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


def test_cost_model_rejects_timeline_gap() -> None:
    model = cost_model()
    segments = model["segments"]
    assert isinstance(segments, list)
    segments[0]["effective_to"] = "2023-08-26"

    with pytest.raises(ValueError, match="contiguous and non-overlapping"):
        validate_semantics("cost-model", model, DOCUMENT)


def test_cost_model_rejects_timeline_overlap() -> None:
    model = cost_model()
    segments = model["segments"]
    assert isinstance(segments, list)
    segments[1]["effective_from"] = "2023-08-27"

    with pytest.raises(ValueError, match="contiguous and non-overlapping"):
        validate_semantics("cost-model", model, DOCUMENT)


def test_cost_model_rejects_unknown_source() -> None:
    model = cost_model()
    segments = model["segments"]
    assert isinstance(segments, list)
    segments[0]["source_ids"].append("missing-source")

    with pytest.raises(ValueError, match="unknown sources"):
        validate_semantics("cost-model", model, DOCUMENT)


def test_cost_model_rejects_closed_final_segment() -> None:
    model = cost_model()
    closed_model = copy.deepcopy(model)
    segments = closed_model["segments"]
    assert isinstance(segments, list)
    segments[-1]["effective_to"] = "2026-12-31"

    with pytest.raises(ValueError, match="must be open-ended"):
        validate_semantics("cost-model", closed_model, DOCUMENT)
