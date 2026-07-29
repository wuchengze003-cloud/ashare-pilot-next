import hashlib
import json
import shutil
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from ashare_quant_core import TargetPosition
from ashare_signal_runner import (
    build_active_run,
    canonical_json_bytes,
    canonical_json_sha256,
    publish_run,
)
from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = ROOT / "contracts/golden_fixtures/vertical-slice"
EXAMPLES = ROOT / "contracts/examples"
SCHEMAS = ROOT / "contracts/schemas"
PROMOTION_CONTRACT_FIELDS = {
    "cost-model": "cost_model_sha256",
    "dataset-manifest": "dataset_manifest_sha256",
    "execution-policy": "execution_policy_sha256",
    "market-rules": "market_rules_sha256",
    "portfolio-risk": "portfolio_risk_sha256",
    "universe": "universe_sha256",
}


def load_json(path: Path) -> dict[str, object] | list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_schemas() -> dict[str, dict[str, object]]:
    registry = load_json(ROOT / "contracts/registry.json")
    assert isinstance(registry, dict)
    schemas: dict[str, dict[str, object]] = {}
    for entry in registry["contracts"]:
        schema = load_json(ROOT / "contracts" / entry["schema"])
        assert isinstance(schema, dict)
        schemas[entry["contract_id"]] = schema
    return schemas


def load_documents() -> dict[str, dict[str, object]]:
    paths = {
        "champion": FIXTURE_ROOT / "champion.json",
        "cost-model": EXAMPLES / "cost-model.example.json",
        "dataset-manifest": FIXTURE_ROOT / "dataset-manifest.json",
        "execution-policy": EXAMPLES / "execution-policy.example.json",
        "market-rules": EXAMPLES / "market-rules.example.json",
        "portfolio-risk": EXAMPLES / "portfolio-risk.example.json",
        "universe": FIXTURE_ROOT / "universe.json",
    }
    documents: dict[str, dict[str, object]] = {}
    for contract_id, path in paths.items():
        document = load_json(path)
        assert isinstance(document, dict)
        documents[contract_id] = document
    return documents


class FixtureReferenceStrategy:
    """Test-only momentum adapter; never eligible for production promotion."""

    strategy_id = "fixture-reference"
    strategy_version = "v1"
    adapter_id = "fixture-reference-adapter/v1"
    adapter_sha256 = "a" * 64

    def __init__(
        self,
        *,
        bars: list[dict[str, object]],
        eligible_symbols: set[str],
    ) -> None:
        self._bars = bars
        self._eligible_symbols = eligible_symbols

    def target_positions(
        self,
        *,
        as_of: date,
        dataset_id: str,
        universe_id: str,
    ) -> tuple[TargetPosition, ...]:
        if not dataset_id.startswith("fixture-daily-"):
            raise ValueError("unexpected fixture dataset")
        if universe_id != "fixture-csi800-pit/v1":
            raise ValueError("unexpected fixture universe")

        history: dict[str, list[tuple[date, float]]] = defaultdict(list)
        for row in self._bars:
            symbol = str(row["symbol"])
            trade_date = date.fromisoformat(str(row["trade_date"]))
            if symbol in self._eligible_symbols and trade_date <= as_of:
                history[symbol].append((trade_date, float(row["close"])))

        momentum: list[tuple[float, str]] = []
        for symbol, observations in history.items():
            ordered = sorted(observations)
            if len(ordered) >= 2:
                momentum.append((ordered[-1][1] / ordered[-2][1] - 1, symbol))
        if not momentum:
            return ()
        _, selected = max(momentum, key=lambda item: (item[0], item[1]))
        return (TargetPosition(selected, 0.15),)


def reference_strategy(
    *,
    bars: list[dict[str, object]] | None = None,
) -> FixtureReferenceStrategy:
    universe = load_json(FIXTURE_ROOT / "universe.json")
    assert isinstance(universe, dict)
    eligible = {
        str(member["symbol"])
        for member in universe["members"]
        if member["eligible"]
    }
    if bars is None:
        raw_bars = load_json(FIXTURE_ROOT / "daily-bars.json")
        assert isinstance(raw_bars, list)
        bars = raw_bars
    return FixtureReferenceStrategy(bars=bars, eligible_symbols=eligible)


def bind_champion_to_documents(
    documents: dict[str, dict[str, object]],
    *,
    lockfile_sha256: str = "3" * 64,
) -> None:
    champion = documents["champion"]
    promotion_contract_set = champion["promotion_contract_set"]
    assert isinstance(promotion_contract_set, dict)
    for contract_id, champion_field in PROMOTION_CONTRACT_FIELDS.items():
        promotion_contract_set[champion_field] = canonical_json_sha256(
            documents[contract_id]
        )
    promotion_contract_set["lockfile_sha256"] = lockfile_sha256


def build_fixture_run(
    *,
    dataset_root: Path = FIXTURE_ROOT,
    documents: dict[str, dict[str, object]] | None = None,
    strategy: FixtureReferenceStrategy | None = None,
    lockfile_sha256: str = "3" * 64,
):
    return build_active_run(
        strategy=strategy or reference_strategy(),
        as_of=date(2026, 7, 29),
        generated_at=datetime(2026, 7, 29, 8, 10, tzinfo=UTC),
        signal_id="fixture-signal-2026-07-29",
        run_id="fixture-run-2026-07-29",
        git_sha="1" * 40,
        lockfile_sha256=lockfile_sha256,
        dataset_root=dataset_root,
        documents=documents or load_documents(),
        schemas=load_schemas(),
    )


def test_vertical_slice_is_byte_deterministic_and_manifest_binds_output(
    tmp_path: Path,
) -> None:
    first = build_fixture_run()
    second = build_fixture_run()

    assert canonical_json_bytes(first.production_signal) == canonical_json_bytes(
        second.production_signal
    )
    assert canonical_json_bytes(first.runtime_manifest) == canonical_json_bytes(
        second.runtime_manifest
    )
    assert first.production_signal["state"] == "ACTIVE"
    assert first.production_signal["target_positions"] == [
        {"symbol": "000001.SZ", "target_weight": 0.15}
    ]
    promotion_contract_set = load_documents()["champion"]["promotion_contract_set"]
    assert isinstance(promotion_contract_set, dict)
    assert first.production_signal["contract_set"]["code_sha256"] == (
        promotion_contract_set["code_sha256"]
    )
    assert first.production_signal["contract_set"]["config_sha256"] == (
        promotion_contract_set["config_sha256"]
    )

    output_dir = tmp_path / "fixture-run"
    publish_run(output_dir=output_dir, artifacts=first)
    signal_bytes = (output_dir / "production-signal.json").read_bytes()
    manifest = json.loads((output_dir / "runtime-manifest.json").read_text())
    assert hashlib.sha256(signal_bytes).hexdigest() == manifest["outputs"][0]["sha256"]
    assert not (tmp_path / ".fixture-run.tmp").exists()

    with pytest.raises(FileExistsError, match="already exists"):
        publish_run(output_dir=output_dir, artifacts=first)


def test_dataset_content_hash_is_verified_before_strategy_runs(tmp_path: Path) -> None:
    copied_fixture = tmp_path / "vertical-slice"
    shutil.copytree(FIXTURE_ROOT, copied_fixture)
    (copied_fixture / "daily-bars.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        build_fixture_run(dataset_root=copied_fixture)


def test_dataset_row_count_is_verified_before_strategy_runs() -> None:
    documents = load_documents()
    manifest = documents["dataset-manifest"]
    files = manifest["files"]
    assert isinstance(files, list)
    files[0]["row_count"] = 999
    bind_champion_to_documents(documents)

    with pytest.raises(ValueError, match="row count mismatch"):
        build_fixture_run(documents=documents)


def test_portfolio_guard_rejects_strategy_target_outside_pit_universe() -> None:
    class IneligibleStrategy(FixtureReferenceStrategy):
        def target_positions(
            self,
            *,
            as_of: date,
            dataset_id: str,
            universe_id: str,
        ) -> tuple[TargetPosition, ...]:
            return (TargetPosition("600000.SH", 0.15),)

    with pytest.raises(ValueError, match="ineligible"):
        build_fixture_run(
            strategy=IneligibleStrategy(bars=[], eligible_symbols=set()),
        )


def test_duplicate_pit_universe_symbol_is_rejected() -> None:
    documents = load_documents()
    universe = documents["universe"]
    members = universe["members"]
    assert isinstance(members, list)
    duplicate = dict(members[0])
    duplicate["reason_codes"] = ["DUPLICATE_FIXTURE"]
    members.append(duplicate)
    bind_champion_to_documents(documents)

    with pytest.raises(ValueError, match="duplicate symbol"):
        build_fixture_run(documents=documents)


def test_target_requires_cost_segment_for_signal_date() -> None:
    documents = load_documents()
    cost_model = documents["cost-model"]
    segments = cost_model["segments"]
    assert isinstance(segments, list)
    segments[-1]["effective_to"] = "2024-01-01"
    bind_champion_to_documents(documents)

    with pytest.raises(ValueError, match="match exactly one segment"):
        build_fixture_run(documents=documents)


def mutate_bound_contract(
    documents: dict[str, dict[str, object]],
    *,
    contract_id: str,
) -> None:
    document = documents[contract_id]
    if contract_id == "dataset-manifest":
        document["source"] = "synthetic-fixture-v2"
    elif contract_id == "universe":
        document["source_version"] = "fixture-v2"
    elif contract_id == "cost-model":
        commission = document["commission"]
        assert isinstance(commission, dict)
        commission["rate"] = 0.00031
    elif contract_id == "market-rules":
        segments = document["segments"]
        assert isinstance(segments, list)
        segments[0]["lot_size"] = 200
    elif contract_id == "execution-policy":
        document["slippage_bps"] = 11
    elif contract_id == "portfolio-risk":
        document["max_positions"] = 9
    else:
        raise AssertionError(f"unsupported contract fixture: {contract_id}")


@pytest.mark.parametrize("contract_id", sorted(PROMOTION_CONTRACT_FIELDS))
def test_champion_rejects_drift_from_promoted_contract_set(contract_id: str) -> None:
    documents = load_documents()
    mutate_bound_contract(documents, contract_id=contract_id)

    with pytest.raises(ValueError, match=f"current {contract_id}"):
        build_fixture_run(documents=documents)


def test_champion_rejects_lockfile_drift() -> None:
    with pytest.raises(ValueError, match="current lockfile"):
        build_fixture_run(lockfile_sha256="4" * 64)


def test_champion_requires_complete_promotion_contract_set() -> None:
    documents = load_documents()
    promotion_contract_set = documents["champion"]["promotion_contract_set"]
    assert isinstance(promotion_contract_set, dict)
    promotion_contract_set.pop("portfolio_risk_sha256")

    with pytest.raises(ValidationError, match="required property"):
        build_fixture_run(documents=documents)


@pytest.mark.parametrize(
    ("adapter_field", "adapter_value"),
    [
        ("adapter_id", "different-adapter/v1"),
        ("adapter_sha256", "e" * 64),
    ],
)
def test_champion_rejects_declared_adapter_drift(
    adapter_field: str,
    adapter_value: str,
) -> None:
    strategy = reference_strategy()
    setattr(strategy, adapter_field, adapter_value)

    with pytest.raises(ValueError, match=adapter_field):
        build_fixture_run(strategy=strategy)


def test_champion_rejects_strategy_without_adapter_identity() -> None:
    class UndeclaredAdapter:
        strategy_id = "fixture-reference"
        strategy_version = "v1"

        def target_positions(
            self,
            *,
            as_of: date,
            dataset_id: str,
            universe_id: str,
        ) -> tuple[TargetPosition, ...]:
            return ()

    with pytest.raises(ValueError, match="adapter_id"):
        build_fixture_run(strategy=UndeclaredAdapter())


def test_future_rows_do_not_change_historical_reference_target() -> None:
    raw_bars = load_json(FIXTURE_ROOT / "daily-bars.json")
    assert isinstance(raw_bars, list)
    baseline = reference_strategy(bars=raw_bars).target_positions(
        as_of=date(2026, 7, 29),
        dataset_id="fixture-daily-2026-07-29-b9591a573343",
        universe_id="fixture-csi800-pit/v1",
    )
    with_future = [
        *raw_bars,
        {
            "symbol": "000001.SZ",
            "trade_date": "2026-07-30",
            "close": 1.0,
        },
    ]
    replayed = reference_strategy(bars=with_future).target_positions(
        as_of=date(2026, 7, 29),
        dataset_id="fixture-daily-2026-07-30-future",
        universe_id="fixture-csi800-pit/v1",
    )

    assert replayed == baseline
