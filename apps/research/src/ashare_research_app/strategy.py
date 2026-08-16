"""Strategy harness: buy-and-hold with exit rules, realistic fills, validation.

Shared "racetrack" for strategy racing. Every candidate strategy runs through
the same execution model so results are comparable:

- Signal produced at t close (model cross-sectional prediction).
- Execute at t+1 OPEN (realistic fill, not close-to-close).
- Hold until an exit rule fires (stop-loss / take-profit / signal expired),
  NOT daily full turnover. Selection is daily, positions held to an exit.
- Equal-weight portfolio, daily mark-to-market (open-to-open).
- Slippage on every traded side; optional market-trend filter for cash-out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class HoldConfig:
    top_k: int = 50
    stop_loss: float = 0.08
    take_profit: float = 0.25
    buffer_days: int = 3
    max_hold_days: int = 20
    cost_per_side: float = 0.0015
    position_scale: pd.Series | None = None


@dataclass(frozen=True)
class HoldResult:
    nav: pd.Series
    daily_turnover: pd.Series
    n_trades: int
    trades: tuple[dict, ...] = ()

    @property
    def annual_return(self) -> float:
        n = len(self.nav)
        return float(self.nav.iloc[-1] ** (TRADING_DAYS / n) - 1.0)

    @property
    def sharpe(self) -> float:
        r = self.nav.pct_change(fill_method=None).dropna()
        return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))

    @property
    def max_drawdown(self) -> float:
        return float((self.nav / self.nav.cummax() - 1.0).min())

    def describe(self) -> str:
        return (
            f"annual {self.annual_return:7.2%} sharpe {self.sharpe:5.2f} "
            f"maxdd {self.max_drawdown:7.2%} turnover {self.daily_turnover.mean():6.2%} "
            f"trades {self.n_trades}"
        )


def _candidate_pool(signal_day, fit, sym_by_date, top_k: int) -> set[str]:
    if signal_day not in fit.predictions:
        return set()
    pred = fit.predictions[signal_day]
    syms = sym_by_date[signal_day]
    return set(syms[np.argsort(-pred)[:top_k]])


def run_hold(
    prepared: pd.DataFrame,
    fit,
    config: HoldConfig,
) -> HoldResult:
    dates = sorted(prepared["trade_date"].unique())
    open_p = prepared.pivot_table(index="trade_date", columns="symbol", values="open")
    sym_by_date = {d: prepared[prepared["trade_date"] == d]["symbol"].to_numpy() for d in dates}

    nav = 1.0
    nav_values: list[float] = []
    turnover_values: list[float] = []
    n_trades = 0
    trades: list[dict] = []

    # holdings: symbol -> (buy_price, buy_index)
    holdings: dict[str, tuple[float, int]] = {}

    for i in range(1, len(dates) - 1):
        sd = dates[i - 1]        # signal day t
        buy_day = dates[i]       # execute at t+1 open
        sell_day = dates[i + 1]  # value at t+2 open

        scale = 1.0
        if config.position_scale is not None and sd in config.position_scale.index:
            scale = float(config.position_scale.loc[sd])

        pool = _candidate_pool(sd, fit, sym_by_date, config.top_k)
        o = open_p.loc[buy_day]
        n_sells = 0
        n_buys = 0

        # ---- exits at buy_day open ----
        for sym in list(holdings.keys()):
            buy_price, buy_i = holdings[sym]
            held_days = i - buy_i
            price = o.get(sym, np.nan)
            if pd.isna(price):
                continue  # suspended: keep holding
            reason = None
            if price <= buy_price * (1 - config.stop_loss):
                reason = "stop"
            elif price >= buy_price * (1 + config.take_profit):
                reason = "profit"
            elif held_days >= config.max_hold_days:
                reason = "maxhold"
            elif sym not in pool and held_days >= config.buffer_days:
                reason = "signal"
            if scale == 0.0:
                reason = "cashout"
            if reason:
                del holdings[sym]
                n_sells += 1
                n_trades += 1
                trades.append(
                    {
                        "date": str(buy_day.date()),
                        "symbol": sym,
                        "side": "sell",
                        "price": round(float(price), 2),
                        "reason": reason,
                    }
                )

        # ---- entries at buy_day open ----
        if scale > 0.0:
            slots = config.top_k - len(holdings)
            if sd in fit.predictions:
                pred = fit.predictions[sd]
                syms = sym_by_date[sd]
                for sym in syms[np.argsort(-pred)]:
                    if slots <= 0:
                        break
                    if sym in holdings:
                        continue
                    price = o.get(sym, np.nan)
                    if pd.isna(price):
                        continue
                    holdings[sym] = (price, i)
                    n_buys += 1
                    n_trades += 1
                    trades.append(
                        {
                            "date": str(buy_day.date()),
                            "symbol": sym,
                            "side": "buy",
                            "price": round(float(price), 2),
                            "reason": "signal",
                        }
                    )
                    slots -= 1

        # ---- portfolio daily return: holdings from buy_day open to sell_day open ----
        # Daily (1-day) return, NOT cumulative from buy price — otherwise the
        # same gain is compounded repeatedly across days.
        o_cur = open_p.loc[buy_day]
        o_next = open_p.loc[sell_day]
        rets = []
        for sym in holdings:
            p0 = o_cur.get(sym, np.nan)
            p1 = o_next.get(sym, np.nan)
            if pd.isna(p0) or pd.isna(p1):
                rets.append(0.0)  # suspended: flat
                continue
            rets.append(p1 / p0 - 1.0)
        gross = float(np.nanmean(rets)) if rets else 0.0

        turnover = (n_sells + n_buys) / max(config.top_k, 1)
        net = gross - turnover * config.cost_per_side
        nav *= 1.0 + net
        nav_values.append(nav)
        turnover_values.append(turnover)

    return HoldResult(
        nav=pd.Series(nav_values, index=pd.DatetimeIndex(dates[1:-1]), name="nav"),
        daily_turnover=pd.Series(turnover_values, index=pd.DatetimeIndex(dates[1:-1])),
        n_trades=n_trades,
        trades=tuple(trades),
    )
