from datetime import date

import pytest
from ashare_data_gateway.adjustment import forward_adjust
from ashare_data_gateway.normalization import NormalizationError, NormalizedDailyBar
from ashare_data_gateway.tushare_models import AdjFactorRecord


def _bar(symbol: str, d: date, close: float) -> NormalizedDailyBar:
    return NormalizedDailyBar(
        symbol=symbol,
        trade_date=d,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
        amount=close * 1000.0,
    )


def _factor(symbol: str, d: date, adj: float) -> AdjFactorRecord:
    return AdjFactorRecord(ts_code=symbol, trade_date=d, adj_factor=adj)


def test_forward_adjust_rebases_to_latest_factor() -> None:
    d1, d2 = date(2023, 1, 2), date(2023, 1, 3)
    bars = [_bar("600000.SH", d1, 10.0), _bar("600000.SH", d2, 12.0)]
    factors = [_factor("600000.SH", d1, 0.8), _factor("600000.SH", d2, 1.0)]
    result = forward_adjust(bars, factors)
    assert len(result) == 2
    assert result[0].close == pytest.approx(10.0 * 0.8 / 1.0)
    assert result[1].close == pytest.approx(12.0 * 1.0 / 1.0)


def test_returns_identical_under_forward_and_backward() -> None:
    d1, d2 = date(2023, 1, 2), date(2023, 1, 3)
    bars = [_bar("600000.SH", d1, 10.0), _bar("600000.SH", d2, 12.0)]
    factors = [_factor("600000.SH", d1, 0.8), _factor("600000.SH", d2, 1.0)]
    result = forward_adjust(bars, factors)
    forward_ret = result[1].close / result[0].close - 1
    backward_ret = (12.0 * 1.0) / (10.0 * 0.8) - 1
    assert forward_ret == pytest.approx(backward_ret)


def test_volume_and_amount_unchanged() -> None:
    d1, d2 = date(2023, 1, 2), date(2023, 1, 3)
    bars = [_bar("600000.SH", d1, 10.0), _bar("600000.SH", d2, 12.0)]
    factors = [_factor("600000.SH", d1, 0.8), _factor("600000.SH", d2, 1.0)]
    result = forward_adjust(bars, factors)
    assert result[0].volume == 1000.0
    assert result[0].amount == 10.0 * 1000.0


def test_missing_factor_fails_closed() -> None:
    d1, d2 = date(2023, 1, 2), date(2023, 1, 3)
    bars = [_bar("600000.SH", d1, 10.0), _bar("600000.SH", d2, 12.0)]
    factors = [_factor("600000.SH", d1, 0.8)]
    with pytest.raises(NormalizationError) as exc:
        forward_adjust(bars, factors)
    assert exc.value.reason_code == "MISSING_ADJ_FACTOR"


def test_symbols_adjust_independently() -> None:
    d1, d2 = date(2023, 1, 2), date(2023, 1, 3)
    bars = [
        _bar("600000.SH", d1, 10.0),
        _bar("600000.SH", d2, 11.0),
        _bar("000001.SZ", d1, 20.0),
        _bar("000001.SZ", d2, 19.0),
    ]
    factors = [
        _factor("600000.SH", d1, 0.5),
        _factor("600000.SH", d2, 1.0),
        _factor("000001.SZ", d1, 2.0),
        _factor("000001.SZ", d2, 3.0),
    ]
    result = forward_adjust(bars, factors)
    by_key = {(b.symbol, b.trade_date): b.close for b in result}
    assert by_key[("600000.SH", d1)] == pytest.approx(10.0 * 0.5 / 1.0)
    assert by_key[("000001.SZ", d1)] == pytest.approx(20.0 * 2.0 / 3.0)


def test_duplicate_factor_raises() -> None:
    d = date(2023, 1, 2)
    bars = [_bar("600000.SH", d, 10.0)]
    factors = [_factor("600000.SH", d, 0.8), _factor("600000.SH", d, 0.9)]
    with pytest.raises(NormalizationError) as exc:
        forward_adjust(bars, factors)
    assert exc.value.reason_code == "DUPLICATE_ADJ_FACTOR"


def test_future_factor_cannot_rebase_a_historical_snapshot() -> None:
    d1, d2 = date(2023, 1, 2), date(2023, 1, 3)
    bars = [_bar("600000.SH", d1, 10.0)]
    factors = [_factor("600000.SH", d1, 1.0), _factor("600000.SH", d2, 2.0)]
    with pytest.raises(NormalizationError) as exc:
        forward_adjust(bars, factors)
    assert exc.value.reason_code == "FUTURE_ADJ_FACTOR"


def test_nan_adjustment_factor_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _factor("600000.SH", date(2023, 1, 2), float("nan"))
