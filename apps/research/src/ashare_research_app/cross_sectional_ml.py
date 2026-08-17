"""Rolling cross-sectional ML selection model.

Trains a model on a trailing window and re-fits every ``refit_every`` days, so
factor weights adapt to the current regime. This is the "self-evolving"
selection engine: it never trusts one factor forever — it learns, from recent
data, which factor works *now*.

Linear models expose their coefficients so the regime adaptation is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class RollingFit:
    """Predictions and (for linear models) factor-weight history."""

    predictions: dict[pd.Timestamp, np.ndarray]
    weights: dict[pd.Timestamp, dict[str, float]] | None


def prepare_panel(
    panel: pd.DataFrame,
    factors: list[str],
    *,
    label_horizon: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """Add cross-sectional z-scores and a forward-return label.

    The label is the forward ``label_horizon``-day cumulative return minus the
    day's cross-sectional mean (so the model learns relative strength, not the
    market's absolute move). Returns ``(prepared, feature_cols)``.
    """
    result = panel.sort_values(["symbol", "trade_date"]).copy()
    result["_fwd_close"] = result.groupby("symbol")["close"].shift(-label_horizon)
    result["_fwd_ret"] = result["_fwd_close"] / result["close"] - 1.0
    result["_label"] = result.groupby("trade_date")["_fwd_ret"].transform(
        lambda s: s - s.mean()
    )

    feature_cols: list[str] = []
    for factor in factors:
        col = f"{factor}_z"
        result[col] = result.groupby("trade_date")[factor].transform(
            lambda s: (s - s.mean()) / s.std()
        )
        feature_cols.append(col)

    result = result.dropna(subset=feature_cols + ["_label"])
    return result, feature_cols


def _make_model(model: str):
    if model == "ridge":
        return Ridge(alpha=1.0)
    if model == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=20260804,
        )
    raise ValueError(f"unknown model: {model}")


def rolling_fit(
    prepared: pd.DataFrame,
    feature_cols: list[str],
    *,
    window: int = 60,
    refit_every: int = 10,
    model: str = "ridge",
) -> RollingFit:
    """Walk-forward fit/predict: refit every ``refit_every`` days on the
    trailing ``window`` days, predict the following days out-of-sample."""
    dates = sorted(prepared["trade_date"].unique())
    predictions: dict[pd.Timestamp, np.ndarray] = {}
    weights: dict[pd.Timestamp, dict[str, float]] = {}

    for i in range(window, len(dates), refit_every):
        train = prepared[
            (prepared["trade_date"] >= dates[i - window])
            & (prepared["trade_date"] <= dates[i])
        ]
        estimator = _make_model(model)
        estimator.fit(train[feature_cols].to_numpy(), train["_label"].to_numpy())
        if model == "ridge":
            weights[dates[i]] = {
                col.removesuffix("_z"): float(w)
                for col, w in zip(feature_cols, estimator.coef_, strict=True)
            }
        for j in range(i + 1, min(i + refit_every, len(dates))):
            day = dates[j]
            day_frame = prepared[prepared["trade_date"] == day]
            predictions[day] = estimator.predict(day_frame[feature_cols].to_numpy())

    return RollingFit(predictions=predictions, weights=weights or None)


def nav_from_predictions(
    prepared: pd.DataFrame,
    fit: RollingFit,
    *,
    top_k: int = 50,
    cost_per_side: float = 0.0015,
    initial_capital: float = 1.0,
    position_scale: pd.Series | None = None,
) -> pd.Series:
    """Simulate an equal-weight top_k portfolio from out-of-sample predictions.

    ``position_scale`` optionally scales exposure each day (e.g. a market-trend
    filter returning 1.0 or 0.0), implementing simple timing risk control.
    """
    dates = sorted(prepared["trade_date"].unique())
    ret_1d = prepared.pivot_table(index="trade_date", columns="symbol", values="ret_1d")
    symbol_by_date = {
        d: prepared[prepared["trade_date"] == d]["symbol"].to_numpy() for d in dates
    }

    nav = initial_capital
    nav_values: list[float] = []
    prev_positions: set[str] = set()
    for i, day in enumerate(dates):
        if i == 0:
            nav_values.append(nav)
            continue
        signal_day = dates[i - 1]
        if signal_day not in fit.predictions:
            nav_values.append(nav)
            prev_positions = set()
            continue
        day_pred = fit.predictions[signal_day]
        symbols = symbol_by_date[signal_day]
        top = set(symbols[np.argsort(-day_pred)[:top_k]])

        day_ret = ret_1d.loc[day]
        gross = day_ret.reindex(list(top)).mean(skipna=True)
        if pd.isna(gross):
            gross = 0.0
        new_names = top - prev_positions
        removed_names = prev_positions - top
        turnover = (len(new_names) + len(removed_names)) / max(top_k, 1)

        scale = 1.0
        if position_scale is not None and signal_day in position_scale.index:
            scale = float(position_scale.loc[signal_day])
        net = scale * gross - turnover * cost_per_side
        nav *= 1.0 + net
        nav_values.append(nav)
        prev_positions = top

    return pd.Series(nav_values, index=pd.DatetimeIndex(dates), name="nav")


def market_trend_filter(
    prepared: pd.DataFrame, *, ma_window: int = 20
) -> pd.Series:
    """Simple timing filter: 1.0 when the equal-weight market NAV is above its
    moving average, 0.0 below (stay out of broad downtrends)."""
    market = prepared.groupby("trade_date")["ret_1d"].mean()
    nav = (1.0 + market).cumprod()
    ma = nav.rolling(ma_window).mean()
    return (nav > ma).astype(float).rename("position_scale")
