from __future__ import annotations

import numpy as np
import pandas as pd
from ashare_research_app.champion import _prepare_forward_labels, rolling_gbdt

FEATURES = ["feature_a", "feature_b"]


def _panel(*, days: int = 45, symbols: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows: list[dict[str, object]] = []
    for day_index, trade_date in enumerate(dates):
        for symbol_index in range(symbols):
            cycle = np.sin((day_index + symbol_index) / 5.0)
            close = 10.0 + symbol_index * 0.3 + day_index * 0.04 + cycle * 0.2
            rows.append(
                {
                    "symbol": f"{600000 + symbol_index:06d}.SH",
                    "trade_date": trade_date,
                    "close": close,
                    "feature_a": cycle,
                    "feature_b": symbol_index / symbols + day_index / days,
                }
            )
    return pd.DataFrame(rows)


def _fit(panel: pd.DataFrame):
    prepared = _prepare_forward_labels(panel, FEATURES, horizon=5)
    fit = rolling_gbdt(
        prepared,
        panel,
        FEATURES,
        label_horizon=5,
        window=12,
        refit=4,
    )
    return prepared, fit


def test_every_refit_uses_only_labels_mature_by_the_refit_date() -> None:
    panel = _panel()
    _, fit = _fit(panel)

    assert fit.model_evolution
    for card in fit.model_evolution:
        assert pd.Timestamp(card["max_label_end"]) <= pd.Timestamp(card["refit_date"])
        assert pd.Timestamp(card["train_end"]) < pd.Timestamp(card["valid_start"])
        assert card["label_horizon"] == 5


def test_future_labels_cannot_change_an_already_refitted_prediction() -> None:
    panel = _panel()
    prepared, original = _fit(panel)
    first_refit = pd.Timestamp(original.model_evolution[0]["refit_date"])
    prediction_day = min(original.predictions)
    original_symbols, original_scores = original.predictions[prediction_day]

    changed = prepared.copy()
    future = changed["_label_end_date"] > first_refit
    assert future.any()
    changed.loc[future, "_label"] = changed.loc[future, "_label"] * -1000.0 + 77.0
    replay = rolling_gbdt(
        changed,
        panel,
        FEATURES,
        label_horizon=5,
        window=12,
        refit=4,
    )
    replay_symbols, replay_scores = replay.predictions[prediction_day]

    np.testing.assert_array_equal(original_symbols, replay_symbols)
    np.testing.assert_allclose(original_scores, replay_scores, rtol=0.0, atol=0.0)


def test_last_mature_model_scores_the_latest_feature_date() -> None:
    panel = _panel()
    _, fit = _fit(panel)

    assert fit.last_refit_date is not None
    assert max(fit.predictions) == panel["trade_date"].max()
