import json

import pytest
from ashare_research_app.market_data import COLUMNS, load_full_market


def _write_symbol(root, code: str, name: str, data: list[dict]) -> None:
    (root / f"{code}.json").write_text(
        json.dumps({"symbol": code, "code": code, "name": name, "data": data}),
        encoding="utf-8",
    )


def test_load_full_market_flattens_symbols(tmp_path) -> None:
    _write_symbol(
        tmp_path,
        "600000.SH",
        "浦发银行",
        [
            {"trade_date": "2023-01-03", "open": 10, "high": 11, "low": 9,
             "close": 10.5, "volume": 1000, "amount": 10500},
            {"trade_date": "2023-01-04", "open": 10.5, "high": 11, "low": 10,
             "close": 10.8, "volume": 1200, "amount": 12960},
        ],
    )
    _write_symbol(
        tmp_path,
        "000001.SZ",
        "平安银行",
        [
            {"trade_date": "2023-01-03", "open": 12, "high": 13, "low": 11,
             "close": 12.5, "volume": 2000, "amount": 25000},
        ],
    )

    df = load_full_market(tmp_path)
    assert list(df.columns) == list(COLUMNS)
    assert len(df) == 3
    assert set(df["symbol"]) == {"600000.SH", "000001.SZ"}
    assert str(df["trade_date"].dtype) == "datetime64[ns]"
    # sorted by (trade_date, symbol)
    assert df.iloc[0]["symbol"] == "000001.SZ"
    assert df.iloc[1]["symbol"] == "600000.SH"


def test_load_empty_dir_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        load_full_market(tmp_path)
