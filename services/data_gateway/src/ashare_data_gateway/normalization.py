"""Deterministic normalization for unadjusted Tushare daily bars.

The vendor boundary uses ``vol`` in lots and ``amount`` in thousands of CNY.
The normalized daily-bar/v1 contract uses shares and CNY.  This module owns
that unit conversion and fails closed before any bytes can be published.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date

from .tushare_models import DailyBarRecord, validate_symbol

VOLUME_SHARES_PER_LOT = 100.0
AMOUNT_CNY_PER_THOUSAND_CNY = 1000.0
VWAP_OHLC_TOLERANCE = 0.05


class NormalizationError(ValueError):
    """A deterministic, machine-classifiable normalization rejection."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        symbol: str | None = None,
        trade_date: date | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.symbol = symbol
        self.trade_date = trade_date
        key = ""
        if symbol is not None or trade_date is not None:
            trade_date_text = (
                trade_date.isoformat() if isinstance(trade_date, date) else repr(trade_date)
            )
            key = f" [{symbol or '?'} {trade_date_text if trade_date is not None else '?'}]"
        super().__init__(f"{reason_code}{key}: {message}")


def _finite_positive_number(
    value: object,
    *,
    field: str,
    symbol: str,
    trade_date: date,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalizationError(
            "INVALID_NUMERIC_VALUE",
            f"{field} must be a number, got {type(value).__name__}",
            symbol=symbol,
            trade_date=trade_date,
        )
    result = float(value)
    if not math.isfinite(result):
        raise NormalizationError(
            "NON_FINITE_VALUE",
            f"{field} must be finite",
            symbol=symbol,
            trade_date=trade_date,
        )
    if result <= 0:
        raise NormalizationError(
            "NON_POSITIVE_VALUE",
            f"{field} must be greater than zero, got {result!r}",
            symbol=symbol,
            trade_date=trade_date,
        )
    return result


def _finite_nonnegative_number(
    value: object,
    *,
    field: str,
    symbol: str,
    trade_date: date,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalizationError(
            "INVALID_NUMERIC_VALUE",
            f"{field} must be a number, got {type(value).__name__}",
            symbol=symbol,
            trade_date=trade_date,
        )
    result = float(value)
    if not math.isfinite(result):
        raise NormalizationError(
            "NON_FINITE_VALUE",
            f"{field} must be finite",
            symbol=symbol,
            trade_date=trade_date,
        )
    if result < 0:
        raise NormalizationError(
            "NEGATIVE_VALUE",
            f"{field} must be non-negative, got {result!r}",
            symbol=symbol,
            trade_date=trade_date,
        )
    return result


def _validate_ohlc(
    *,
    symbol: str,
    trade_date: date,
    open_price: float,
    high: float,
    low: float,
    close: float,
) -> None:
    if low > high:
        raise NormalizationError(
            "INVALID_OHLC",
            "low cannot exceed high",
            symbol=symbol,
            trade_date=trade_date,
        )
    if low > min(open_price, close):
        raise NormalizationError(
            "INVALID_OHLC",
            "low cannot exceed open or close",
            symbol=symbol,
            trade_date=trade_date,
        )
    if high < max(open_price, close):
        raise NormalizationError(
            "INVALID_OHLC",
            "high cannot be below open or close",
            symbol=symbol,
            trade_date=trade_date,
        )


def _validate_implied_vwap(
    *,
    symbol: str,
    trade_date: date,
    low: float,
    high: float,
    volume: float,
    amount: float,
) -> None:
    implied_vwap = amount / volume
    if not math.isfinite(implied_vwap) or implied_vwap <= 0:
        raise NormalizationError(
            "INVALID_IMPLIED_VWAP",
            "amount / volume must be finite and positive",
            symbol=symbol,
            trade_date=trade_date,
        )
    lower_bound = low * (1.0 - VWAP_OHLC_TOLERANCE)
    upper_bound = high * (1.0 + VWAP_OHLC_TOLERANCE)
    if not math.isfinite(lower_bound) or not math.isfinite(upper_bound):
        raise NormalizationError(
            "VWAP_BOUND_OVERFLOW",
            "OHLC tolerance bounds must remain finite",
            symbol=symbol,
            trade_date=trade_date,
        )
    if implied_vwap < lower_bound or implied_vwap > upper_bound:
        raise NormalizationError(
            "VWAP_OUTSIDE_OHLC_BAND",
            (f"implied_vwap={implied_vwap!r} is outside [{lower_bound!r}, {upper_bound!r}]"),
            symbol=symbol,
            trade_date=trade_date,
        )


@dataclass(frozen=True, order=True)
class NormalizedDailyBar:
    """One normalized-daily-bar/v1 row.

    Prices and amount are CNY; volume is shares.  All fields are validated on
    direct construction as well as through :func:`normalize_daily_bars` so a
    caller cannot bypass the quality gate before publication.
    """

    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float

    def __post_init__(self) -> None:
        try:
            validate_symbol(self.symbol, context="normalized_daily_bar")
        except ValueError as exc:
            raise NormalizationError(
                "INVALID_SYMBOL",
                str(exc),
                symbol=self.symbol,
                trade_date=self.trade_date,
            ) from exc
        if type(self.trade_date) is not date:
            raise NormalizationError(
                "INVALID_TRADE_DATE",
                f"trade_date must be date, got {type(self.trade_date).__name__}",
                symbol=self.symbol,
            )

        normalized_values: dict[str, float] = {}
        for field in ("open", "high", "low", "close"):
            normalized_values[field] = _finite_positive_number(
                getattr(self, field),
                field=field,
                symbol=self.symbol,
                trade_date=self.trade_date,
            )
            object.__setattr__(self, field, normalized_values[field])
        for field in ("volume", "amount"):
            normalized_values[field] = _finite_nonnegative_number(
                getattr(self, field),
                field=field,
                symbol=self.symbol,
                trade_date=self.trade_date,
            )
            object.__setattr__(self, field, normalized_values[field])

        zero_volume = normalized_values["volume"] == 0
        zero_amount = normalized_values["amount"] == 0
        if zero_volume != zero_amount:
            raise NormalizationError(
                "INCONSISTENT_ZERO_TURNOVER",
                "volume and amount must either both be zero or both be positive",
                symbol=self.symbol,
                trade_date=self.trade_date,
            )

        _validate_ohlc(
            symbol=self.symbol,
            trade_date=self.trade_date,
            open_price=normalized_values["open"],
            high=normalized_values["high"],
            low=normalized_values["low"],
            close=normalized_values["close"],
        )
        if not zero_volume:
            _validate_implied_vwap(
                symbol=self.symbol,
                trade_date=self.trade_date,
                low=normalized_values["low"],
                high=normalized_values["high"],
                volume=normalized_values["volume"],
                amount=normalized_values["amount"],
            )

    @property
    def document(self) -> dict[str, object]:
        return {
            **asdict(self),
            "trade_date": self.trade_date.isoformat(),
        }


def normalize_daily_bars(
    records: Iterable[DailyBarRecord],
    *,
    requested_symbols: Iterable[str],
    window_start: date,
    window_end: date,
) -> tuple[NormalizedDailyBar, ...]:
    """Validate request binding, convert units, and return primary-key order.

    An empty vendor result is represented by an empty tuple; publication of an
    empty normalized dataset is rejected separately because Manifest 2.0
    requires every listed file to contain at least one row.
    """
    if type(window_start) is not date or type(window_end) is not date:
        raise TypeError("normalization window bounds must be date objects")
    if window_end < window_start:
        raise ValueError("normalization window_end cannot precede window_start")

    symbol_set: set[str] = set()
    for symbol in requested_symbols:
        validate_symbol(symbol, context="normalization request")
        if symbol in symbol_set:
            raise ValueError(f"duplicate requested symbol: {symbol}")
        symbol_set.add(symbol)
    if not symbol_set:
        raise ValueError("normalization requires at least one requested symbol")

    seen: set[tuple[str, date]] = set()
    normalized: list[NormalizedDailyBar] = []
    for record in records:
        if not isinstance(record, DailyBarRecord):
            raise NormalizationError(
                "INVALID_RECORD_TYPE",
                f"expected DailyBarRecord, got {type(record).__name__}",
            )
        if type(record.trade_date) is not date:
            raise NormalizationError(
                "INVALID_TRADE_DATE",
                f"trade_date must be date, got {type(record.trade_date).__name__}",
                symbol=record.ts_code,
            )
        if record.ts_code not in symbol_set:
            raise NormalizationError(
                "REQUEST_SYMBOL_MISMATCH",
                "vendor symbol was not included in the explicit request",
                symbol=record.ts_code,
                trade_date=record.trade_date,
            )
        if record.trade_date < window_start or record.trade_date > window_end:
            raise NormalizationError(
                "REQUEST_DATE_MISMATCH",
                (
                    "vendor trade_date is outside the explicit request window "
                    f"[{window_start.isoformat()}, {window_end.isoformat()}]"
                ),
                symbol=record.ts_code,
                trade_date=record.trade_date,
            )
        key = (record.ts_code, record.trade_date)
        if key in seen:
            raise NormalizationError(
                "DUPLICATE_DAILY_BAR",
                "duplicate normalized primary key",
                symbol=record.ts_code,
                trade_date=record.trade_date,
            )
        seen.add(key)

        raw_open = _finite_positive_number(
            record.open,
            field="open",
            symbol=record.ts_code,
            trade_date=record.trade_date,
        )
        raw_high = _finite_positive_number(
            record.high,
            field="high",
            symbol=record.ts_code,
            trade_date=record.trade_date,
        )
        raw_low = _finite_positive_number(
            record.low,
            field="low",
            symbol=record.ts_code,
            trade_date=record.trade_date,
        )
        raw_close = _finite_positive_number(
            record.close,
            field="close",
            symbol=record.ts_code,
            trade_date=record.trade_date,
        )
        raw_volume = _finite_nonnegative_number(
            record.vol,
            field="vol",
            symbol=record.ts_code,
            trade_date=record.trade_date,
        )
        raw_amount = _finite_nonnegative_number(
            record.amount,
            field="amount",
            symbol=record.ts_code,
            trade_date=record.trade_date,
        )
        volume = raw_volume * VOLUME_SHARES_PER_LOT
        amount = raw_amount * AMOUNT_CNY_PER_THOUSAND_CNY
        if not math.isfinite(volume) or not math.isfinite(amount):
            raise NormalizationError(
                "UNIT_CONVERSION_OVERFLOW",
                "normalized volume or amount is non-finite after unit conversion",
                symbol=record.ts_code,
                trade_date=record.trade_date,
            )

        normalized.append(
            NormalizedDailyBar(
                symbol=record.ts_code,
                trade_date=record.trade_date,
                open=raw_open,
                high=raw_high,
                low=raw_low,
                close=raw_close,
                volume=volume,
                amount=amount,
            )
        )

    return tuple(sorted(normalized, key=lambda row: (row.symbol, row.trade_date)))


def canonical_json_bytes(value: object) -> bytes:
    """Return the repository's deterministic JSON encoding."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_daily_bar_bytes(records: Iterable[NormalizedDailyBar]) -> bytes:
    """Serialize normalized rows in stable primary-key order."""
    materialized = tuple(records)
    if any(not isinstance(record, NormalizedDailyBar) for record in materialized):
        raise TypeError("canonical daily-bar serialization requires NormalizedDailyBar records")
    ordered = tuple(sorted(materialized, key=lambda row: (row.symbol, row.trade_date)))
    seen: set[tuple[str, date]] = set()
    for record in ordered:
        key = (record.symbol, record.trade_date)
        if key in seen:
            raise NormalizationError(
                "DUPLICATE_DAILY_BAR",
                "duplicate normalized primary key",
                symbol=record.symbol,
                trade_date=record.trade_date,
            )
        seen.add(key)
    return canonical_json_bytes([record.document for record in ordered])
