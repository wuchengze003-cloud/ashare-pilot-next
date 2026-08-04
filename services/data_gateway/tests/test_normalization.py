"""Deterministic daily-bar normalization tests; no network or wall clock."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date

import pytest
from ashare_data_gateway.normalization import (
    NormalizationError,
    NormalizedDailyBar,
    canonical_daily_bar_bytes,
    normalize_daily_bars,
)
from ashare_data_gateway.tushare_models import DailyBarRecord

START = date(2023, 6, 1)
END = date(2023, 10, 31)
SYMBOLS = ("000001.SZ", "600519.SH")


def raw_bar(
    *,
    symbol: str = "000001.SZ",
    trade_date: date = START,
    open_price: float = 10.0,
    high: float = 11.0,
    low: float = 9.0,
    close: float = 10.5,
    vol: float = 100.0,
    amount: float = 100.0,
) -> DailyBarRecord:
    return DailyBarRecord(
        ts_code=symbol,
        trade_date=trade_date,
        open=open_price,
        high=high,
        low=low,
        close=close,
        vol=vol,
        amount=amount,
    )


def normalize(*records: DailyBarRecord) -> tuple[NormalizedDailyBar, ...]:
    return normalize_daily_bars(
        records,
        requested_symbols=SYMBOLS,
        window_start=START,
        window_end=END,
    )


def test_normalizes_vendor_units_and_primary_key_order() -> None:
    later = raw_bar(symbol="600519.SH", trade_date=date(2023, 6, 2))
    earlier = raw_bar(symbol="000001.SZ", trade_date=date(2023, 6, 1))

    result = normalize(later, earlier)

    assert [(row.symbol, row.trade_date) for row in result] == [
        ("000001.SZ", date(2023, 6, 1)),
        ("600519.SH", date(2023, 6, 2)),
    ]
    assert result[0].volume == 10_000.0
    assert result[0].amount == 100_000.0
    assert result[0].amount / result[0].volume == 10.0


def test_canonical_bytes_ignore_vendor_response_order() -> None:
    rows = (
        raw_bar(symbol="600519.SH", trade_date=date(2023, 6, 2)),
        raw_bar(symbol="000001.SZ", trade_date=date(2023, 6, 1)),
    )

    first = normalize_daily_bars(
        rows,
        requested_symbols=reversed(SYMBOLS),
        window_start=START,
        window_end=END,
    )
    second = normalize_daily_bars(
        reversed(rows),
        requested_symbols=SYMBOLS,
        window_start=START,
        window_end=END,
    )

    assert first == second
    assert canonical_daily_bar_bytes(first) == canonical_daily_bar_bytes(reversed(second))
    document = json.loads(canonical_daily_bar_bytes(first))
    assert list(document[0]) == sorted(document[0])


def test_empty_vendor_result_is_deterministic_but_not_fabricated() -> None:
    result = normalize_daily_bars(
        (),
        requested_symbols=SYMBOLS,
        window_start=START,
        window_end=END,
    )

    assert result == ()
    assert canonical_daily_bar_bytes(result) == b"[]"


def test_duplicate_vendor_primary_key_fails_closed() -> None:
    duplicate = raw_bar()

    with pytest.raises(NormalizationError) as exc_info:
        normalize(duplicate, duplicate)

    assert exc_info.value.reason_code == "DUPLICATE_DAILY_BAR"


def test_vendor_symbol_is_bound_to_explicit_request() -> None:
    with pytest.raises(NormalizationError) as exc_info:
        normalize_daily_bars(
            (raw_bar(symbol="300750.SZ"),),
            requested_symbols=("000001.SZ",),
            window_start=START,
            window_end=END,
        )

    assert exc_info.value.reason_code == "REQUEST_SYMBOL_MISMATCH"


@pytest.mark.parametrize(
    "trade_date",
    [date(2023, 5, 31), date(2023, 11, 1)],
)
def test_vendor_date_is_bound_to_explicit_request(trade_date: date) -> None:
    with pytest.raises(NormalizationError) as exc_info:
        normalize(raw_bar(trade_date=trade_date))

    assert exc_info.value.reason_code == "REQUEST_DATE_MISMATCH"


@pytest.mark.parametrize(
    ("field", "value", "reason_code"),
    [
        ("open", 0.0, "NON_POSITIVE_VALUE"),
        ("high", -1.0, "NON_POSITIVE_VALUE"),
        ("low", math.nan, "NON_FINITE_VALUE"),
        ("close", math.inf, "NON_FINITE_VALUE"),
        ("vol", -1.0, "NEGATIVE_VALUE"),
        ("amount", -1.0, "NEGATIVE_VALUE"),
        ("amount", True, "INVALID_NUMERIC_VALUE"),
    ],
)
def test_invalid_numeric_values_fail_closed(
    field: str,
    value: object,
    reason_code: str,
) -> None:
    record = replace(raw_bar(), **{field: value})

    with pytest.raises(NormalizationError) as exc_info:
        normalize(record)

    assert exc_info.value.reason_code == reason_code


@pytest.mark.parametrize(
    "record",
    [
        raw_bar(open_price=8.0, low=9.0),
        raw_bar(close=12.0, high=11.0),
        raw_bar(high=8.0, low=9.0, open_price=9.5, close=9.5),
    ],
)
def test_invalid_ohlc_fails_closed(record: DailyBarRecord) -> None:
    with pytest.raises(NormalizationError) as exc_info:
        normalize(record)

    assert exc_info.value.reason_code == "INVALID_OHLC"


@pytest.mark.parametrize("amount", [80.0, 120.0])
def test_implied_vwap_outside_five_percent_ohlc_band_fails(amount: float) -> None:
    record = raw_bar(
        open_price=10.0,
        high=10.0,
        low=10.0,
        close=10.0,
        vol=100.0,
        amount=amount,
    )

    with pytest.raises(NormalizationError) as exc_info:
        normalize(record)

    assert exc_info.value.reason_code == "VWAP_OUTSIDE_OHLC_BAND"


def test_zero_volume_and_amount_are_valid_without_vwap_division() -> None:
    rows = normalize(raw_bar(vol=0.0, amount=0.0))

    assert rows[0].volume == 0.0
    assert rows[0].amount == 0.0


@pytest.mark.parametrize(("vol", "amount"), [(0.0, 1.0), (1.0, 0.0)])
def test_zero_turnover_fields_must_be_consistent(vol: float, amount: float) -> None:
    with pytest.raises(NormalizationError) as exc_info:
        normalize(raw_bar(vol=vol, amount=amount))

    assert exc_info.value.reason_code == "INCONSISTENT_ZERO_TURNOVER"


def test_direct_normalized_construction_cannot_bypass_vwap_gate() -> None:
    with pytest.raises(NormalizationError, match="VWAP_OUTSIDE_OHLC_BAND"):
        NormalizedDailyBar(
            symbol="000001.SZ",
            trade_date=START,
            open=10.0,
            high=10.0,
            low=10.0,
            close=10.0,
            volume=10_000.0,
            amount=50_000.0,
        )
