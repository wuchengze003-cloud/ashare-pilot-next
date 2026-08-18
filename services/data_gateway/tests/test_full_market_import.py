from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from ashare_data_gateway.dataset_publication import validate_prepared_dataset
from ashare_data_gateway.full_market_import import prepare_full_market_import
from ashare_data_gateway.normalization import NormalizedDailyBar
from ashare_data_gateway.tushare_models import StockBasicRecord


def _master(symbol: str, *, delist_date: date | None = None) -> StockBasicRecord:
    return StockBasicRecord(
        ts_code=symbol,
        symbol=symbol.split(".", 1)[0],
        name=symbol,
        area="",
        industry="",
        market="main",
        list_date=date(2020, 1, 1),
        delist_date=delist_date,
        list_status="D" if delist_date is not None else "L",
    )


def _bar(symbol: str, day: date, price: float = 10.0) -> NormalizedDailyBar:
    return NormalizedDailyBar(
        symbol=symbol,
        trade_date=day,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=100.0,
        amount=price * 100.0,
    )


def _write_export(root: Path, symbol: str, days: list[date]) -> None:
    rows = [
        {
            "trade_date": day.isoformat(),
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 100.0,
            "amount": 1000.0,
        }
        for day in days
    ]
    (root / f"{symbol.split('.', 1)[0]}.json").write_text(
        json.dumps({"symbol": symbol, "data": rows}),
        encoding="utf-8",
    )


def test_import_selects_at_start_and_keeps_later_delisted_member(tmp_path: Path) -> None:
    selection = date(2023, 4, 27)
    as_of = date(2023, 4, 28)
    source = tmp_path / "source"
    source.mkdir()
    symbols = tuple(f"{index:06d}.SZ" for index in range(1, 23))
    security_master = tuple(
        _master(symbol, delist_date=as_of if symbol == symbols[0] else None)
        for symbol in symbols
    )

    daily_basic = tmp_path / "daily-basic.json"
    daily_basic.write_text(
        json.dumps(
            {
                "fields": ["ts_code", "trade_date", "circ_mv"],
                "items": [
                    [symbol, selection.strftime("%Y%m%d"), 1000 - index]
                    for index, symbol in enumerate(symbols)
                ],
            }
        ),
        encoding="utf-8",
    )
    # The highest-ranked name is absent locally to exercise delisted-history supplementation.
    for symbol in symbols[1:]:
        days = [selection] if symbol == symbols[2] else [selection, as_of]
        _write_export(source, symbol, days)

    def supplement(symbol: str, start: date, end: date):
        if symbol == symbols[0]:
            assert (start, end) == (selection, as_of)
            return (_bar(symbol, selection), _bar(symbol, as_of))
        assert (symbol, start, end) == (symbols[2], as_of, as_of)
        return (_bar(symbol, as_of),)

    result = prepare_full_market_import(
        source_root=source,
        selection_daily_basic=daily_basic,
        security_master=security_master,
        selection_st_symbols={symbols[1]},
        current_st_symbols={symbols[3]},
        security_master_rejections=(),
        selection_date=selection,
        as_of=as_of,
        generated_at=datetime(2023, 4, 28, 9, tzinfo=UTC),
        top_n=20,
        supplement_loader=supplement,
        trading_days=(selection, as_of),
        suspension_keys=frozenset(),
    )

    records = validate_prepared_dataset(result.prepared)
    assert len(result.selected_symbols) == 20
    assert symbols[0] in result.selected_symbols
    assert symbols[1] not in result.selected_symbols
    assert result.supplemented_symbols == (symbols[0], symbols[2])
    assert {record.symbol for record in records} == set(result.selected_symbols)

    members = {member["symbol"]: member for member in result.universe["members"]}
    assert members[symbols[0]]["eligible"] is True
    assert members[symbols[2]]["eligible"] is True
    assert members[symbols[3]]["eligible"] is False
    assert "CURRENT_ST_EXCLUDED" in members[symbols[3]]["reason_codes"]
