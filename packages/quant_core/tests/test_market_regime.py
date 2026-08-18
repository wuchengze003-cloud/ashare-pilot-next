"""Tests for causal market-regime hysteresis."""

from datetime import date, timedelta

import pytest
from ashare_quant_core import hysteresis_market_regime


def _dated(values: list[float]) -> list[tuple[date, float]]:
    start = date(2026, 1, 1)
    return [(start + timedelta(days=index), value) for index, value in enumerate(values)]


def test_hysteresis_turns_flat_and_requires_positive_band_to_reenter() -> None:
    points = hysteresis_market_regime(
        _dated([0.0, 0.0, -0.20, 0.01, 0.01, 0.30]),
        moving_average_window=3,
        band=0.03,
    )

    assert [point.invested for point in points] == [True, True, False, False, False, True]


def test_future_returns_cannot_change_past_regime() -> None:
    prefix = _dated([0.01, -0.02, 0.03, -0.04, 0.01])
    full = hysteresis_market_regime(
        [*prefix, (date(2026, 1, 6), 0.5)],
        moving_average_window=3,
        band=0.03,
    )
    truncated = hysteresis_market_regime(
        prefix,
        moving_average_window=3,
        band=0.03,
    )

    assert full[: len(truncated)] == truncated


@pytest.mark.parametrize("bad_return", [float("nan"), float("inf"), -1.0])
def test_invalid_market_returns_fail_closed(bad_return: float) -> None:
    with pytest.raises(ValueError, match="market returns"):
        hysteresis_market_regime(_dated([bad_return]))
