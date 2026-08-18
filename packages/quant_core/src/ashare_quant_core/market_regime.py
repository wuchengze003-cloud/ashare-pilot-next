"""Causal market-regime semantics shared by research and production."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, order=True)
class MarketRegimePoint:
    trade_date: date
    invested: bool
    market_nav: float
    moving_average: float | None
    deviation: float | None


def hysteresis_market_regime(
    daily_returns: Sequence[tuple[date, float]],
    *,
    moving_average_window: int = 120,
    band: float = 0.03,
    initially_invested: bool = True,
) -> tuple[MarketRegimePoint, ...]:
    """Convert a dated market-return series into a causal invested/flat state.

    A value above ``+band`` turns the state on, a value below ``-band`` turns it
    off, and the state is retained inside the band. The moving average at date
    ``t`` uses only NAV observations on or before ``t``.
    """
    if moving_average_window < 2:
        raise ValueError("moving_average_window must be at least 2")
    if not 0 <= band < 1:
        raise ValueError("band must be in [0, 1)")
    previous_date: date | None = None
    nav = 1.0
    nav_history: list[float] = []
    invested = initially_invested
    result: list[MarketRegimePoint] = []
    for trade_date, daily_return in daily_returns:
        if previous_date is not None and trade_date <= previous_date:
            raise ValueError("market returns must be unique and date-ordered")
        parsed_return = float(daily_return)
        if not math.isfinite(parsed_return) or parsed_return <= -1:
            raise ValueError("market returns must be finite and greater than -1")
        nav *= 1.0 + parsed_return
        nav_history.append(nav)
        moving_average: float | None = None
        deviation: float | None = None
        if len(nav_history) >= moving_average_window:
            window = nav_history[-moving_average_window:]
            moving_average = sum(window) / moving_average_window
            deviation = nav / moving_average - 1.0
            if deviation > band:
                invested = True
            elif deviation < -band:
                invested = False
        result.append(
            MarketRegimePoint(
                trade_date=trade_date,
                invested=invested,
                market_nav=nav,
                moving_average=moving_average,
                deviation=deviation,
            )
        )
        previous_date = trade_date
    return tuple(result)
