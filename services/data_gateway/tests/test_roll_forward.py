from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from ashare_data_gateway.coverage import SecurityLifecycle
from ashare_data_gateway.dataset_publication import (
    DatasetSourceIdentity,
    load_published_dataset,
    prepare_normalized_dataset,
    publish_normalized_dataset,
)
from ashare_data_gateway.normalization import NormalizedDailyBar
from ashare_data_gateway.roll_forward import prepare_roll_forward, publish_roll_forward


def _bar(symbol: str, trade_date: date, price: float = 10.0) -> NormalizedDailyBar:
    return NormalizedDailyBar(
        symbol=symbol,
        trade_date=trade_date,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1000.0,
        amount=price * 1000.0,
    )


def _parent(tmp_path: Path):
    symbols = tuple(f"{index:06d}.SZ" for index in range(1, 21))
    parent_day = date(2026, 8, 17)
    prepared = prepare_normalized_dataset(
        (_bar(symbol, parent_day) for symbol in symbols),
        as_of=parent_day,
        generated_at=datetime(2026, 8, 17, 12, tzinfo=UTC),
        source_identity=DatasetSourceIdentity(
            base_url_host="api.tushare.pro",
            is_official_vendor=True,
            manifest_source="tushare-official",
        ),
        source_version="test-parent/v1",
        dataset_family_id="fixed-test/v1",
        normalization_version="fixed-test/v1",
    )
    paths = publish_normalized_dataset(publication_root=tmp_path, prepared=prepared)
    parent = load_published_dataset(publication_root=tmp_path, dataset_id=paths.dataset_id)
    universe = {
        "contract_id": "universe",
        "schema_version": "2.0.0",
        "universe_id": "fixed-test-pit/v1",
        "universe_policy_id": "fixed-test",
        "universe_policy_version": "v1",
        "as_of": parent_day.isoformat(),
        "generated_at": "2026-08-17T12:00:00Z",
        "source": "test",
        "source_version": "v1",
        "quality_status": "pass",
        "members": [
            {
                "symbol": symbol,
                "valid_from": "2026-08-17",
                "valid_to": None,
                "eligible": True,
                "reason_codes": ["DATA_CURRENT"],
            }
            for symbol in symbols
        ],
    }
    return parent, universe, symbols


def _lifecycles(symbols: tuple[str, ...]) -> tuple[SecurityLifecycle, ...]:
    return tuple(
        SecurityLifecycle(symbol=symbol, listed_on=date(2000, 1, 1), delisted_on=None)
        for symbol in symbols
    )


def test_roll_forward_publishes_new_dataset_without_mutating_parent(tmp_path: Path) -> None:
    parent, universe, symbols = _parent(tmp_path)
    parent_bytes = parent.data_bytes
    target_day = date(2026, 8, 18)
    incremental = tuple(_bar(symbol, target_day, 10.5) for symbol in symbols[:18])
    prepared, next_universe = prepare_roll_forward(
        parent=parent,
        universe=universe,
        incremental_records=incremental,
        current_st_symbols={symbols[0]},
        as_of=target_day,
        generated_at=datetime(2026, 8, 18, 12, tzinfo=UTC),
        source_base_url="https://api.tushare.pro",
        trading_days=(target_day,),
        suspension_keys=frozenset((symbol, target_day) for symbol in symbols[18:]),
        lifecycles=_lifecycles(symbols),
    )
    universe_path = tmp_path / "universes" / "2026-08-18.json"
    paths = publish_roll_forward(
        publication_root=tmp_path,
        universe_out=universe_path,
        prepared=prepared,
        universe=next_universe,
    )

    reloaded_parent = load_published_dataset(
        publication_root=tmp_path, dataset_id=parent.dataset_id
    )
    child = load_published_dataset(publication_root=tmp_path, dataset_id=paths.dataset_id)
    assert reloaded_parent.data_bytes == parent_bytes
    assert len(child.records) == 38
    stored_universe = json.loads(universe_path.read_bytes())
    first = stored_universe["members"][0]
    assert first["eligible"] is False
    assert "CURRENT_ST_EXCLUDED" in first["reason_codes"]
    assert stored_universe["members"][-1]["eligible"] is False


def test_roll_forward_rejects_partial_vendor_day(tmp_path: Path) -> None:
    parent, universe, symbols = _parent(tmp_path)
    with pytest.raises(ValueError, match="coverage"):
        prepare_roll_forward(
            parent=parent,
            universe=universe,
            incremental_records=tuple(
                _bar(symbol, date(2026, 8, 18)) for symbol in symbols[:17]
            ),
            current_st_symbols=(),
            as_of=date(2026, 8, 18),
            generated_at=datetime(2026, 8, 18, 12, tzinfo=UTC),
            source_base_url="https://api.tushare.pro",
            trading_days=(date(2026, 8, 18),),
            suspension_keys=frozenset(),
            lifecycles=_lifecycles(symbols),
        )


def test_roll_forward_rejects_unexplained_missing_member_day(tmp_path: Path) -> None:
    parent, universe, symbols = _parent(tmp_path)
    target_day = date(2026, 8, 18)
    with pytest.raises(ValueError, match="member-day coverage failed"):
        prepare_roll_forward(
            parent=parent,
            universe=universe,
            incremental_records=tuple(_bar(symbol, target_day) for symbol in symbols[:18]),
            current_st_symbols=(),
            as_of=target_day,
            generated_at=datetime(2026, 8, 18, 12, tzinfo=UTC),
            source_base_url="https://api.tushare.pro",
            trading_days=(target_day,),
            suspension_keys=frozenset(),
            lifecycles=_lifecycles(symbols),
        )
