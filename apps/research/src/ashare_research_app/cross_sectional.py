"""Cross-sectional factor-ranking backtest over the full market.

Ranks every stock by a factor each day (cross-section), holds the top/bottom
``top_k`` equal-weight, and rebalances daily with T+1 execution (today's
positions come from yesterday's signal) and per-side costs. Metrics mirror the
research-discipline modules so results can feed Deflated Sharpe and bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def limit_up_threshold(symbol: str) -> float:
    """Approximate A-share limit-up magnitude by board.

    Main board (±10%), ChiNext/STAR (±20%), BSE (±30%). Used to exclude
    limit-up names from the signal because a limit-up stock cannot actually be
    bought (the order queue is full), which would otherwise inflate momentum
    backtests with an unrealizable "chase the limit" premium.
    """
    code = symbol.split(".", 1)[0]
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


@dataclass(frozen=True)
class BacktestResult:
    nav: pd.Series
    annual_return: float
    sharpe: float
    max_drawdown: float
    n_rebalances: int
    turnover_per_year: float

    def describe(self) -> str:
        return (
            f"ann_return={self.annual_return:.2%} sharpe={self.sharpe:.2f} "
            f"max_dd={self.max_drawdown:.2%} turnover={self.turnover_per_year:.1f}/yr"
        )


def run_cross_sectional(
    bars: pd.DataFrame,
    *,
    factor: str,
    top_k: int,
    higher_is_better: bool = True,
    cost_per_side: float = 0.0015,
    initial_capital: float = 1_000_000.0,
    max_abs_return: float = 0.20,
    exclude_limit_up: bool = True,
) -> BacktestResult:
    """Run a daily-rebalanced cross-sectional backtest.

    Args:
        bars: full-market frame already enriched by :func:`add_factors` (must
            contain ``factor``, ``ret_1d``, ``symbol``, ``trade_date``).
        factor: column name to rank cross-sectionally each day.
        top_k: number of names held.
        higher_is_better: select the largest factor values (False selects smallest).
        cost_per_side: cost per traded side (stamp duty + commission + slippage).
        initial_capital: starting NAV.
        max_abs_return: filter out names with an absolute daily return above
            this (new-IPO limit-free spikes and ex-right gaps are not tradable
            and would otherwise pollute momentum factors). Disable with 0.
        exclude_limit_up: drop names that closed at their limit-up (they cannot
            be bought), which otherwise fabricates a "chase the limit" premium.
    """
    if factor not in bars.columns or "ret_1d" not in bars.columns:
        raise ValueError("bars must contain the factor column and ret_1d")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    panel = bars.sort_values(["trade_date", "symbol"]).copy()
    if max_abs_return > 0:
        panel["_abs_ret"] = panel["ret_1d"].abs()
        panel["_abnormal"] = panel.groupby("symbol")["_abs_ret"].transform(
            lambda s: s.rolling(20, min_periods=1).max() > max_abs_return
        )
        panel["_ret_1d"] = panel["ret_1d"].where(
            panel["ret_1d"].abs() <= max_abs_return
        )
    else:
        panel["_abnormal"] = False
        panel["_ret_1d"] = panel["ret_1d"]
    if exclude_limit_up:
        panel["_limit_up"] = panel["symbol"].map(limit_up_threshold)
        panel["_at_limit"] = panel["ret_1d"] >= panel["_limit_up"] - 0.005
    else:
        panel["_at_limit"] = False

    dates = sorted(panel["trade_date"].unique())

    signal_by_date: dict[pd.Timestamp, set[str]] = {}
    for day in dates:
        day_panel = panel[panel["trade_date"] == day]
        valid = day_panel.dropna(subset=[factor])
        valid = valid[~valid["_abnormal"]]
        valid = valid[~valid["_at_limit"]]
        ranked = valid.sort_values(factor, ascending=not higher_is_better, kind="stable")
        signal_by_date[day] = set(ranked.head(top_k)["symbol"])

    ret_1d = panel.pivot_table(index="trade_date", columns="symbol", values="_ret_1d")

    nav = initial_capital
    nav_values: list[float] = []
    prev_positions: set[str] = set()
    total_turnover = 0.0
    n_rebalances = 0

    for i, day in enumerate(dates):
        if i == 0:
            nav_values.append(nav)
            continue
        positions = signal_by_date[dates[i - 1]]
        if not positions:
            # no valid signal yet (factor warm-up) — hold cash
            nav_values.append(nav)
            prev_positions = set()
            continue
        day_ret = ret_1d.loc[day]
        gross = day_ret.reindex(list(positions)).mean(skipna=True)
        if pd.isna(gross):
            gross = 0.0
        new_names = positions - prev_positions
        removed_names = prev_positions - positions
        turnover_ratio = (len(new_names) + len(removed_names)) / max(top_k, 1)
        net = gross - turnover_ratio * cost_per_side
        nav *= 1.0 + net
        nav_values.append(nav)
        total_turnover += turnover_ratio
        n_rebalances += 1
        prev_positions = positions

    nav_index = pd.Series(nav_values, index=pd.DatetimeIndex(dates), name="nav")
    daily_returns = nav_index.pct_change(fill_method=None).dropna()
    if daily_returns.empty or daily_returns.std() == 0:
        raise ValueError("not enough variance to compute backtest metrics")

    annual_return = (nav / initial_capital) ** (TRADING_DAYS_PER_YEAR / len(nav_values)) - 1
    sharpe = float(
        daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    max_drawdown = float((nav_index / nav_index.cummax() - 1).min())
    turnover_per_year = total_turnover / n_rebalances * TRADING_DAYS_PER_YEAR

    return BacktestResult(
        nav=nav_index,
        annual_return=float(annual_return),
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        n_rebalances=n_rebalances,
        turnover_per_year=float(turnover_per_year),
    )
