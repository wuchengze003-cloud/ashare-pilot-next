"""Apply adjustment factors to normalized daily bars (forward adjustment).

Tushare's ``adj_factor`` re-bases prices to the *backward*-adjusted series
(``backward_adjusted_price = raw_price * adj_factor``). Forward adjustment
divides every factor by the symbol's latest factor, so historical prices align
with the current (unadjusted) price. Daily returns are identical under either
convention because both scale by the same per-date multiplier.

M1 collected ``adj_factor`` but did not apply it; this module is the M4 step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from .normalization import NormalizationError, NormalizedDailyBar
from .tushare_models import AdjFactorRecord, validate_symbol


@dataclass(frozen=True)
class AdjustedDailyBar:
    """One forward-adjusted daily bar.

    ``open``/``high``/``low``/``close`` are forward-adjusted prices (CNY).
    ``volume`` and ``amount`` are unchanged — adjustment re-bases prices only,
    never turnover. No VWAP check applies here because price and amount no
    longer share the same adjustment base.
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
            validate_symbol(self.symbol, context="adjusted_daily_bar")
        except ValueError as exc:
            raise NormalizationError(
                "INVALID_SYMBOL", str(exc), symbol=self.symbol, trade_date=self.trade_date
            ) from exc
        if type(self.trade_date) is not date:
            raise NormalizationError(
                "INVALID_TRADE_DATE",
                f"trade_date must be date, got {type(self.trade_date).__name__}",
                symbol=self.symbol,
                trade_date=self.trade_date,
            )
        for field in ("open", "high", "low", "close"):
            value = float(getattr(self, field))
            if value <= 0:
                raise NormalizationError(
                    "NON_POSITIVE_PRICE",
                    f"{field} must be greater than zero after adjustment",
                    symbol=self.symbol,
                    trade_date=self.trade_date,
                )
            object.__setattr__(self, field, value)
        if self.low > self.high:
            raise NormalizationError(
                "INVALID_OHLC",
                "low cannot exceed high after adjustment",
                symbol=self.symbol,
                trade_date=self.trade_date,
            )


def forward_adjust(
    bars: Sequence[NormalizedDailyBar],
    factors: Sequence[AdjFactorRecord],
) -> tuple[AdjustedDailyBar, ...]:
    """Forward-adjust bars; fail closed if any bar lacks its adjustment factor.

    Every bar must have a matching ``AdjFactorRecord`` on the same
    ``(symbol, trade_date)``. Missing factors are rejected rather than silently
    publishing unadjusted prices.
    """
    factor_map: dict[tuple[str, date], float] = {}
    latest_factor: dict[str, float] = {}
    latest_date: dict[str, date] = {}
    for factor in factors:
        validate_symbol(factor.ts_code, context="adj_factor")
        if factor.adj_factor <= 0:
            raise NormalizationError(
                "NON_POSITIVE_ADJ_FACTOR",
                f"adj_factor must be positive, got {factor.adj_factor}",
                symbol=factor.ts_code,
                trade_date=factor.trade_date,
            )
        key = (factor.ts_code, factor.trade_date)
        if key in factor_map:
            raise NormalizationError(
                "DUPLICATE_ADJ_FACTOR",
                "duplicate adjustment factor primary key",
                symbol=factor.ts_code,
                trade_date=factor.trade_date,
            )
        factor_map[key] = factor.adj_factor
        if factor.ts_code not in latest_date or factor.trade_date > latest_date[factor.ts_code]:
            latest_factor[factor.ts_code] = factor.adj_factor
            latest_date[factor.ts_code] = factor.trade_date

    adjusted: list[AdjustedDailyBar] = []
    for bar in bars:
        key = (bar.symbol, bar.trade_date)
        if key not in factor_map:
            raise NormalizationError(
                "MISSING_ADJ_FACTOR",
                "daily bar has no matching adjustment factor",
                symbol=bar.symbol,
                trade_date=bar.trade_date,
            )
        if bar.symbol not in latest_factor:
            raise NormalizationError(
                "MISSING_LATEST_ADJ_FACTOR",
                "symbol has no latest adjustment factor",
                symbol=bar.symbol,
                trade_date=bar.trade_date,
            )
        scale = factor_map[key] / latest_factor[bar.symbol]
        adjusted.append(
            AdjustedDailyBar(
                symbol=bar.symbol,
                trade_date=bar.trade_date,
                open=bar.open * scale,
                high=bar.high * scale,
                low=bar.low * scale,
                close=bar.close * scale,
                volume=bar.volume,
                amount=bar.amount,
            )
        )
    return tuple(sorted(adjusted, key=lambda row: (row.symbol, row.trade_date)))
