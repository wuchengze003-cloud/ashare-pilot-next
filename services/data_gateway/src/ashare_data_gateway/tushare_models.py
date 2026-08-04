"""Frozen data models for Tushare vendor records.

All records are immutable (frozen dataclass + tuple containers).
No adjusted prices, no derived fields — raw vendor data only.

Field units (verified by live probe 2026-07-30):
- daily.open/high/low/close: 元 (CNY, unadjusted)
- daily.vol: 手 (lots, 1 lot = 100 shares)
- daily.amount: 千元 (thousands of CNY)
- adj_factor.adj_factor: 复权因子 (dimensionless multiplier, NOT applied here)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date

SYMBOL_PATTERN = re.compile(r"^[0-9]{6}\.(SZ|SH|BJ)$")
DATE_PATTERN = re.compile(r"^[0-9]{8}$")


def validate_symbol(value: object, *, context: str) -> str:
    """Validate and return a well-formed A-share symbol."""
    if not isinstance(value, str):
        raise ValueError(f"{context}: symbol must be str, got {type(value).__name__}")
    if not SYMBOL_PATTERN.fullmatch(value):
        raise ValueError(f"{context}: invalid symbol format: {value!r}")
    return value


def validate_trade_date(value: object, *, context: str) -> date:
    """Validate an 8-digit vendor date string and return a date object."""
    if not isinstance(value, str):
        raise ValueError(f"{context}: date must be str, got {type(value).__name__}")
    if not DATE_PATTERN.fullmatch(value):
        raise ValueError(f"{context}: invalid date format: {value!r}")
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except ValueError as exc:
        raise ValueError(f"{context}: impossible calendar date: {value!r}") from exc


def validate_finite_float(value: object, *, context: str, field: str) -> float:
    """Validate a numeric field is finite (no NaN/Inf). Rejects bool."""
    if value is None:
        raise ValueError(f"{context}: field {field} is None")
    if isinstance(value, bool):
        raise ValueError(f"{context}: field {field} must be numeric, got bool")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{context}: field {field} must be numeric, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: field {field} is non-finite: {value!r}")
    return result


@dataclass(frozen=True, order=True)
class TradeCalRecord:
    """One exchange trading-calendar day.

    Source: Tushare trade_cal. Units: N/A (calendar data).
    """

    exchange: str
    cal_date: date
    is_open: int
    pretrade_date: date | None

    def __post_init__(self) -> None:
        if not self.exchange:
            raise ValueError("trade_cal: exchange cannot be empty")
        if self.is_open not in (0, 1):
            raise ValueError(f"trade_cal: is_open must be 0 or 1, got {self.is_open}")


@dataclass(frozen=True, order=True)
class StockBasicRecord:
    """One security master record.

    Source: Tushare stock_basic. Units: N/A (reference data).
    """

    ts_code: str
    symbol: str
    name: str
    area: str
    industry: str
    market: str
    list_date: date | None
    delist_date: date | None
    list_status: str

    def __post_init__(self) -> None:
        validate_symbol(self.ts_code, context="stock_basic")
        if not self.symbol:
            raise ValueError("stock_basic: symbol cannot be empty")
        if not self.name:
            raise ValueError("stock_basic: name cannot be empty")
        if self.list_status not in ("L", "D", "P"):
            raise ValueError(f"stock_basic: invalid list_status: {self.list_status!r}")
        if (
            self.delist_date is not None
            and self.list_date is not None
            and self.delist_date < self.list_date
        ):
            raise ValueError("stock_basic: delist_date cannot precede list_date")


@dataclass(frozen=True, order=True)
class DailyBarRecord:
    """One unadjusted daily bar.

    Source: Tushare daily (verified 2026-07-30).
    Confirmed fields: ts_code, trade_date, open, high, low, close, vol, amount.
    Units:
      - open/high/low/close: 元 (CNY, unadjusted)
      - vol: 手 (lots, 1 lot = 100 shares)
      - amount: 千元 (thousands of CNY)
    """

    ts_code: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    vol: float
    amount: float

    def __post_init__(self) -> None:
        validate_symbol(self.ts_code, context="daily")


@dataclass(frozen=True, order=True)
class AdjFactorRecord:
    """One adjustment-factor record (kept as independent vendor record).

    Source: Tushare adj_factor.
    Units: dimensionless multiplier. This PR does NOT apply the factor.
    """

    ts_code: str
    trade_date: date
    adj_factor: float

    def __post_init__(self) -> None:
        validate_symbol(self.ts_code, context="adj_factor")
        if self.adj_factor <= 0:
            raise ValueError(f"adj_factor: must be positive, got {self.adj_factor}")
