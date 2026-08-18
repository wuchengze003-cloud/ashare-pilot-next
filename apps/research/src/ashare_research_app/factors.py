"""Cross-sectional factors computed from OHLCV daily bars.

Every factor at time ``t`` uses only bars on or before ``t``: returns use
``pct_change`` (shift-based) and windows use ``rolling``, both of which never
peek ahead. The panel is therefore causal by construction. Add a new factor
here and it will be caught by the look-ahead gate if it ever reads the future.
"""

from __future__ import annotations

import pandas as pd

COLUMNS: tuple[str, ...] = (
    "symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"
)

FACTOR_NAMES: tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "vol_20d",
    "amt_20d",
    "vol_avg_20d",
)


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add factor columns to a full-market bar DataFrame.

    Expects columns ``symbol``, ``trade_date``, ``open``, ``high``, ``low``,
    ``close``, ``volume``, ``amount``. Returns a new frame sorted by
    ``(symbol, trade_date)`` with the factor columns appended.
    """
    result = df.sort_values(["symbol", "trade_date"]).copy()
    grouped = result.groupby("symbol", sort=False)

    result["ret_1d"] = grouped["close"].pct_change()
    result["ret_5d"] = grouped["close"].pct_change(5)
    result["ret_20d"] = grouped["close"].pct_change(20)
    result["ret_60d"] = grouped["close"].pct_change(60)

    result["vol_20d"] = grouped["ret_1d"].transform(
        lambda series: series.rolling(20, min_periods=20).std()
    )
    result["amt_20d"] = grouped["amount"].transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    result["vol_avg_20d"] = grouped["volume"].transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    return result
