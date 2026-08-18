import math
from datetime import date, timedelta

import pytest
from ashare_quant_core import DailyBar
from ashare_research_app.causality import factor_causality_check


def _series(symbol: str, closes: list[float]) -> list[DailyBar]:
    start = date(2023, 1, 2)
    bars = []
    for i, close in enumerate(closes):
        bars.append(
            DailyBar(
                symbol=symbol,
                trade_date=start + timedelta(days=i),
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1000.0,
                amount=1000.0 * close,
            )
        )
    return bars


def _causal_momentum(bars):
    # trailing 5-day return: uses only the past, so full == truncated.
    out = [None] * len(bars)
    for i in range(len(bars)):
        if i >= 5:
            out[i] = bars[i].close / bars[i - 5].close - 1
    return out


def _lookahead_zscore(bars):
    # z-score normalized over the FULL sample: reads the future.
    closes = [b.close for b in bars]
    mean = sum(closes) / len(closes)
    var = sum((c - mean) ** 2 for c in closes) / len(closes)
    std = math.sqrt(var)
    if std == 0:
        return [0.0] * len(bars)
    return [(c - mean) / std for c in closes]


def _forward_return(bars):
    # uses next bar's close: look-ahead, and the truncated tail is None.
    out = [None] * len(bars)
    for i in range(len(bars) - 1):
        out[i] = bars[i + 1].close / bars[i].close - 1
    return out


def test_causal_factor_passes() -> None:
    closes = [10.0 + 0.1 * i for i in range(60)]
    report = factor_causality_check(
        _causal_momentum, {"600000.SH": _series("600000.SH", closes)}
    )
    assert report.passed
    assert report.checked_points > 0


def test_lookahead_zscore_is_detected() -> None:
    closes = [10.0 + 0.1 * i for i in range(60)]
    report = factor_causality_check(
        _lookahead_zscore, {"600000.SH": _series("600000.SH", closes)}
    )
    assert not report.passed
    assert report.leak is not None


def test_forward_return_factor_is_detected() -> None:
    closes = [10.0 + 0.1 * i for i in range(60)]
    report = factor_causality_check(
        _forward_return, {"600000.SH": _series("600000.SH", closes)}
    )
    assert not report.passed
    assert report.leak is not None


def test_factor_length_mismatch_raises() -> None:
    def bad(bars):
        return [1.0]

    with pytest.raises(ValueError):
        factor_causality_check(bad, {"600000.SH": _series("600000.SH", [10.0] * 40)})


def test_describe_reports_leak_location() -> None:
    closes = [10.0 + 0.1 * i for i in range(60)]
    report = factor_causality_check(
        _lookahead_zscore, {"600000.SH": _series("600000.SH", closes)}
    )
    text = report.describe()
    assert "FAIL" in text
    assert "600000.SH" in text


def test_existing_feature_row_is_causal() -> None:
    from ashare_research_app.features import compute_feature_row

    def ret_1d_factor(bars):
        out = [None] * len(bars)
        for i in range(len(bars)):
            row = compute_feature_row(bars, i)
            out[i] = row[0] if row is not None else None
        return out

    closes = [10.0 + 0.1 * i for i in range(60)]
    report = factor_causality_check(
        ret_1d_factor, {"600000.SH": _series("600000.SH", closes)}
    )
    assert report.passed
    assert report.checked_points > 0
