import pandas as pd
import pytest
from ashare_research_app.cross_sectional import limit_up_threshold, run_cross_sectional


def _panel() -> pd.DataFrame:
    # 600000 has the highest factor and the highest return each day;
    # returns vary across days so the variance is non-zero.
    rets = [0.01, 0.02, -0.01, 0.005]
    rows = []
    for d, r in enumerate(rets):
        for symbol, factor, ret in [
            ("600000.SH", 3.0, r),
            ("600001.SH", 2.0, r - 0.005),
            ("600002.SH", 1.0, r - 0.01),
        ]:
            rows.append(
                (
                    symbol,
                    pd.Timestamp("2023-01-02") + pd.Timedelta(days=d),
                    factor,
                    ret,
                )
            )
    return pd.DataFrame(rows, columns=["symbol", "trade_date", "factor", "ret_1d"])


def test_selects_highest_factor_and_compounds() -> None:
    result = run_cross_sectional(
        _panel(), factor="factor", top_k=1, cost_per_side=0.0015, initial_capital=100.0
    )
    assert len(result.nav) == 4
    assert result.nav.iloc[0] == 100.0
    # day1: signal from day0 picks 600000.SH, earning rets[1] = +2%, first build
    # costs one side (T+1 execution)
    assert result.nav.iloc[1] == pytest.approx(100.0 * (1 + 0.02 - 0.0015))
    # day2: no turnover, earning rets[2] = -1%
    assert result.nav.iloc[2] == pytest.approx(result.nav.iloc[1] * (1 - 0.01))


def test_higher_is_better_false_selects_lowest() -> None:
    result = run_cross_sectional(
        _panel(),
        factor="factor",
        top_k=1,
        higher_is_better=False,
        cost_per_side=0.0,
        initial_capital=100.0,
    )
    # lowest factor is 600002.SH, whose day-1 return is rets[1] - 0.01 = +1%
    assert result.nav.iloc[1] == pytest.approx(100.0 * (1 + 0.01))


def test_missing_factor_raises() -> None:
    panel = _panel().drop(columns=["factor"])
    with pytest.raises(ValueError):
        run_cross_sectional(panel, factor="factor", top_k=1)


def test_non_positive_top_k_raises() -> None:
    with pytest.raises(ValueError):
        run_cross_sectional(_panel(), factor="factor", top_k=0)


def test_result_is_deterministic() -> None:
    a = run_cross_sectional(_panel(), factor="factor", top_k=1)
    b = run_cross_sectional(_panel(), factor="factor", top_k=1)
    pd.testing.assert_series_equal(a.nav, b.nav)


def test_limit_up_threshold_by_board() -> None:
    assert limit_up_threshold("600000.SH") == 0.10
    assert limit_up_threshold("000001.SZ") == 0.10
    assert limit_up_threshold("300001.SZ") == 0.20
    assert limit_up_threshold("688001.SH") == 0.20
    assert limit_up_threshold("830001.BJ") == 0.30


def test_limit_up_name_is_excluded_from_signal() -> None:
    # 600000 closes at its limit every day (highest factor), but it cannot be
    # bought, so the strategy must pick 600001 (whose return varies by day).
    rets = [0.01, 0.02, -0.01, 0.005]
    rows = []
    for d in range(40):
        r = rets[d % len(rets)]
        rows.append(("600000.SH", pd.Timestamp("2023-01-02") + pd.Timedelta(days=d), 9.0, 0.098))
        rows.append(("600001.SH", pd.Timestamp("2023-01-02") + pd.Timedelta(days=d), 2.0, r))
        rows.append(("600002.SH", pd.Timestamp("2023-01-02") + pd.Timedelta(days=d), 1.0, 0.0))
    panel = pd.DataFrame(rows, columns=["symbol", "trade_date", "factor", "ret_1d"])
    result = run_cross_sectional(
        panel, factor="factor", top_k=1, higher_is_better=True, cost_per_side=0.0,
        initial_capital=100.0,
    )
    # day1 picks 600001 (600000 is limit-up and excluded), earning rets[1] = +2%
    assert result.nav.iloc[1] == pytest.approx(100.0 * 1.02)
