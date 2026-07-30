from datetime import date

import pytest
from ashare_quant_core import (
    DAILY_BAR_SCHEMA_ID,
    DAILY_BAR_SCHEMA_SHA256,
    DailyBar,
    DatasetSnapshot,
)
from ashare_research_app import build_feature_replay


def snapshot(*, as_of: date, bars: tuple[DailyBar, ...]) -> DatasetSnapshot:
    return DatasetSnapshot(
        dataset_id=f"fixture-{as_of.isoformat()}",
        dataset_family_id="fixture-daily/v1",
        manifest_sha256="a" * 64,
        as_of=as_of,
        data_schema_id=DAILY_BAR_SCHEMA_ID,
        data_schema_sha256=DAILY_BAR_SCHEMA_SHA256,
        normalization_version="fixture-normalizer/v1",
        records=tuple(bar for bar in bars if bar.trade_date <= as_of),
    )


BARS = (
    DailyBar("000001.SZ", date(2026, 7, 27), 10, 11, 9, 10, 100, 1000),
    DailyBar("000001.SZ", date(2026, 7, 28), 10, 12, 10, 11, 120, 1320),
    DailyBar("000001.SZ", date(2026, 7, 29), 11, 13, 10, 12, 150, 1800),
    DailyBar("600000.SH", date(2026, 7, 28), 8, 9, 7, 8, 80, 640),
)


def test_replay_is_byte_deterministic() -> None:
    first = build_feature_replay(snapshot(as_of=date(2026, 7, 29), bars=BARS))
    second = build_feature_replay(
        snapshot(as_of=date(2026, 7, 29), bars=tuple(reversed(BARS)))
    )

    assert first.canonical_bytes == second.canonical_bytes
    assert first.replay_sha256 == second.replay_sha256


def test_future_rows_do_not_change_historical_features() -> None:
    future = DailyBar(
        "000001.SZ",
        date(2026, 7, 30),
        4,
        5,
        3,
        4,
        500,
        2000,
    )
    early = build_feature_replay(
        snapshot(as_of=date(2026, 7, 29), bars=BARS)
    )
    late = build_feature_replay(
        snapshot(as_of=date(2026, 7, 29), bars=(*BARS, future))
    )

    assert early.canonical_bytes == late.canonical_bytes
    assert early.rows == late.rows


def test_replay_rejects_a_snapshot_that_leaks_future_rows() -> None:
    future = DailyBar(
        "000001.SZ",
        date(2026, 7, 30),
        4,
        5,
        3,
        4,
        500,
        2000,
    )

    class UnsafeSnapshot:
        as_of = date(2026, 7, 29)
        snapshot_sha256 = "b" * 64

        def bars(self, *, through: date):
            assert through == self.as_of
            return (*BARS, future)

    with pytest.raises(ValueError, match="future bar"):
        build_feature_replay(UnsafeSnapshot())  # type: ignore[arg-type]
