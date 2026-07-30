import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from ashare_quant_core import (
    DAILY_BAR_SCHEMA_ID,
    DAILY_BAR_SCHEMA_SHA256,
    DailyBar,
    DatasetSnapshot,
    UniverseMember,
    UniverseSnapshot,
)

ROOT = Path(__file__).resolve().parents[3]


def bar(
    symbol: str,
    trade_date: date,
    *,
    close: float = 10.0,
) -> DailyBar:
    return DailyBar(
        symbol=symbol,
        trade_date=trade_date,
        open=close,
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        volume=100.0,
        amount=1000.0,
    )


def dataset_snapshot(
    *,
    dataset_id: str,
    records: tuple[DailyBar, ...],
) -> DatasetSnapshot:
    return DatasetSnapshot(
        dataset_id=dataset_id,
        dataset_family_id="fixture-daily/v1",
        manifest_sha256="a" * 64,
        as_of=date(2026, 7, 29),
        data_schema_id=DAILY_BAR_SCHEMA_ID,
        data_schema_sha256=DAILY_BAR_SCHEMA_SHA256,
        normalization_version="fixture-normalizer/v1",
        records=records,
    )


def test_tracked_daily_bar_descriptor_matches_runtime_schema_hash() -> None:
    descriptor = json.loads(
        (
            ROOT
            / "contracts/data_schemas/normalized-daily-bar-v1.json"
        ).read_text(encoding="utf-8")
    )
    canonical = json.dumps(
        descriptor,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()

    assert hashlib.sha256(canonical).hexdigest() == DAILY_BAR_SCHEMA_SHA256


def test_dataset_snapshot_is_order_independent_and_content_addressed() -> None:
    records = (
        bar("000001.SZ", date(2026, 7, 29), close=10.5),
        bar("000001.SZ", date(2026, 7, 28), close=10.0),
    )

    first = dataset_snapshot(dataset_id="dataset-a", records=records)
    second = dataset_snapshot(dataset_id="dataset-b", records=tuple(reversed(records)))

    assert first.records == second.records
    assert first.snapshot_sha256 == second.snapshot_sha256


def test_dataset_snapshot_rejects_duplicate_primary_key() -> None:
    duplicate = bar("000001.SZ", date(2026, 7, 29))

    with pytest.raises(ValueError, match="duplicate daily bar primary key"):
        dataset_snapshot(dataset_id="dataset-a", records=(duplicate, duplicate))


def test_dataset_snapshot_rejects_future_record_and_query() -> None:
    with pytest.raises(ValueError, match="after as_of"):
        dataset_snapshot(
            dataset_id="dataset-a",
            records=(bar("000001.SZ", date(2026, 7, 30)),),
        )

    snapshot = dataset_snapshot(
        dataset_id="dataset-a",
        records=(bar("000001.SZ", date(2026, 7, 29)),),
    )
    with pytest.raises(ValueError, match="after snapshot as_of"):
        snapshot.bars(through=date(2026, 7, 30))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symbol": "BAD"},
        {"close": float("nan")},
        {"close": 0.0},
    ],
)
def test_daily_bar_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "symbol": "000001.SZ",
        "trade_date": date(2026, 7, 29),
        "open": 10.0,
        "high": 10.1,
        "low": 9.9,
        "close": 10.0,
        "volume": 100.0,
        "amount": 1000.0,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        DailyBar(**values)  # type: ignore[arg-type]


def test_universe_snapshot_is_order_independent() -> None:
    members = (
        UniverseMember(
            symbol="600000.SH",
            valid_from=date(2026, 1, 1),
            valid_to=None,
            eligible=False,
            reason_codes=("SUSPENDED",),
        ),
        UniverseMember(
            symbol="000001.SZ",
            valid_from=date(2026, 1, 1),
            valid_to=None,
            eligible=True,
            reason_codes=("DATA_COMPLETE", "BASE_MEMBER"),
        ),
    )
    first = UniverseSnapshot(
        universe_id="fixture-csi800-pit/v1",
        universe_policy_id="fixture-csi800",
        universe_policy_version="v1",
        source_sha256="b" * 64,
        as_of=date(2026, 7, 29),
        members=members,
    )
    second = UniverseSnapshot(
        universe_id="different-snapshot-id",
        universe_policy_id="fixture-csi800",
        universe_policy_version="v1",
        source_sha256="c" * 64,
        as_of=date(2026, 7, 29),
        members=tuple(reversed(members)),
    )

    assert first.snapshot_sha256 == second.snapshot_sha256
    assert first.eligible_symbols == frozenset({"000001.SZ"})
