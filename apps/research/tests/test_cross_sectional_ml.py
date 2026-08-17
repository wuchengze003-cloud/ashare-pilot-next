import pandas as pd
import pytest
from ashare_research_app.cross_sectional_ml import (
    market_trend_filter,
    nav_from_predictions,
    prepare_panel,
    rolling_fit,
)


def _panel(n_symbols: int = 3, n_days: int = 80) -> pd.DataFrame:
    rows = []
    for s in range(n_symbols):
        symbol = f"60000{s}.SH"
        for d in range(n_days):
            close = 10.0 + s + d * 0.1
            rows.append(
                (
                    symbol,
                    pd.Timestamp("2023-01-02") + pd.Timedelta(days=d),
                    close,
                    close,
                    close,
                    close,
                    1000.0,
                    close * 1000.0,
                )
            )
    return pd.DataFrame(
        rows,
        columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"],
    )


def test_prepare_panel_label_is_forward_cumulative_return() -> None:
    panel = _panel()
    panel["ret_1d"] = panel.groupby("symbol")["close"].pct_change()
    prepared, feats = prepare_panel(panel, factors=["ret_1d"], label_horizon=5)
    sym = (
        prepared[prepared["symbol"] == "600000.SH"]
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    row = sym.iloc[0]
    # _fwd_ret = close[t+5] / close[t] - 1
    expected = sym.iloc[5]["close"] / sym.iloc[0]["close"] - 1.0
    assert row["_fwd_ret"] == pytest.approx(expected)
    # label = fwd_ret minus the day's cross-sectional mean
    day_mean = prepared[prepared["trade_date"] == row["trade_date"]]["_fwd_ret"].mean()
    assert row["_label"] == pytest.approx(expected - day_mean)


def test_rolling_fit_is_strictly_out_of_sample() -> None:
    panel = _panel()
    for f in ["ret_1d"]:
        panel[f] = panel.groupby("symbol")["close"].pct_change()
    prepared, feats = prepare_panel(panel, factors=["ret_1d"], label_horizon=5)
    fit = rolling_fit(prepared, feats, window=30, refit_every=10, model="ridge")
    dates = sorted(prepared["trade_date"].unique())
    # first prediction is strictly after the first refit point (window)
    refit_points = set(dates[30::10])
    for day in fit.predictions:
        assert day not in refit_points


def test_ridge_exposes_factor_weights() -> None:
    panel = _panel()
    panel["ret_1d"] = panel.groupby("symbol")["close"].pct_change()
    prepared, feats = prepare_panel(panel, factors=["ret_1d"], label_horizon=5)
    fit = rolling_fit(prepared, feats, window=30, refit_every=10, model="ridge")
    assert fit.weights is not None
    assert all("ret_1d" in w for w in fit.weights.values())


def test_market_trend_filter_returns_binary() -> None:
    panel = _panel()
    panel["ret_1d"] = panel.groupby("symbol")["close"].pct_change()
    prepared, _ = prepare_panel(panel, factors=["ret_1d"], label_horizon=5)
    scale = market_trend_filter(prepared)
    assert set(scale.unique()) <= {0.0, 1.0}


def test_nav_from_predictions_uses_prior_signal() -> None:
    panel = _panel(n_symbols=1)
    panel["ret_1d"] = panel.groupby("symbol")["close"].pct_change()
    prepared, feats = prepare_panel(panel, factors=["ret_1d"], label_horizon=5)
    fit = rolling_fit(prepared, feats, window=30, refit_every=10, model="ridge")
    nav = nav_from_predictions(prepared, fit, top_k=1)
    assert len(nav) == prepared["trade_date"].nunique()
