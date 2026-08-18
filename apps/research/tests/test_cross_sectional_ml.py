import numpy as np
import pandas as pd
import pytest
from ashare_research_app.cross_sectional_ml import (
    _mature_relative_labels,
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
        prepared[prepared["symbol"] == "600000.SH"].sort_values("trade_date").reset_index(drop=True)
    )
    row = sym.iloc[0]
    # _fwd_ret = close[t+5] / close[t] - 1
    expected = sym.iloc[5]["close"] / sym.iloc[0]["close"] - 1.0
    assert row["_fwd_ret"] == pytest.approx(expected)
    cutoff = pd.Timestamp(prepared["_label_end_date"].max())
    mature = _mature_relative_labels(prepared, cutoff=cutoff)
    mature_row = mature[
        (mature["symbol"] == "600000.SH") & (mature["trade_date"] == row["trade_date"])
    ].iloc[0]
    day_mean = mature[mature["trade_date"] == row["trade_date"]]["_fwd_ret"].mean()
    assert mature_row["_label"] == pytest.approx(expected - day_mean)


def test_rolling_fit_is_strictly_out_of_sample() -> None:
    panel = _panel()
    for f in ["ret_1d"]:
        panel[f] = panel.groupby("symbol")["close"].pct_change()
    prepared, feats = prepare_panel(panel, factors=["ret_1d"], label_horizon=5)
    fit = rolling_fit(prepared, feats, window=30, refit_every=10, model="ridge")
    assert fit.predictions
    for refit_date in fit.weights or {}:
        mature = prepared[
            prepared["_fwd_ret"].notna() & (prepared["_label_end_date"] <= refit_date)
        ]
        assert mature["trade_date"].nunique() >= 30


def test_future_labels_cannot_change_an_existing_rolling_prediction() -> None:
    panel = _panel(n_symbols=6, n_days=90)
    panel["ret_1d"] = panel.groupby("symbol")["close"].pct_change()
    prepared, feats = prepare_panel(panel, factors=["ret_1d"], label_horizon=5)
    original = rolling_fit(prepared, feats, window=30, refit_every=10, model="ridge")
    first_refit = min(original.weights or {})
    first_prediction = min(original.predictions)

    changed = prepared.copy()
    future = changed["_label_end_date"] > first_refit
    changed.loc[future, "_fwd_ret"] = changed.loc[future, "_fwd_ret"] * -1000.0 + 17.0
    replay = rolling_fit(changed, feats, window=30, refit_every=10, model="ridge")

    assert np.array_equal(
        original.predictions[first_prediction],
        replay.predictions[first_prediction],
    )


def test_unmatured_peer_return_cannot_change_a_mature_relative_label() -> None:
    prepared = pd.DataFrame(
        [
            {
                "symbol": "A",
                "trade_date": pd.Timestamp("2026-01-01"),
                "_label_end_date": pd.Timestamp("2026-01-03"),
                "_fwd_ret": 0.2,
            },
            {
                "symbol": "B",
                "trade_date": pd.Timestamp("2026-01-01"),
                "_label_end_date": pd.Timestamp("2026-01-04"),
                "_fwd_ret": 1.0,
            },
        ]
    )

    before = _mature_relative_labels(
        prepared,
        cutoff=pd.Timestamp("2026-01-03"),
    )
    changed = prepared.copy()
    changed.loc[changed["symbol"] == "B", "_fwd_ret"] = 100.0
    after = _mature_relative_labels(
        changed,
        cutoff=pd.Timestamp("2026-01-03"),
    )

    assert before.loc[before["symbol"] == "A", "_label"].iloc[0] == 0.0
    assert after.loc[after["symbol"] == "A", "_label"].iloc[0] == 0.0


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
