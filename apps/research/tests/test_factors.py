import pandas as pd
import pytest
from ashare_research_app.factors import FACTOR_NAMES, add_factors
from ashare_research_app.market_data import COLUMNS


def _sample_df(n_symbols: int = 2, n_days: int = 60) -> pd.DataFrame:
    rows = []
    for s in range(n_symbols):
        symbol = f"60000{s}.SH"
        base = 10.0 + s
        for d in range(n_days):
            close = base + d * 0.1
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
    return pd.DataFrame(rows, columns=COLUMNS)


def test_ret_1d_is_pct_change() -> None:
    result = add_factors(_sample_df(n_symbols=1))
    assert result["ret_1d"].iloc[1] == pytest.approx(0.1 / 10.0)
    assert result["ret_1d"].iloc[0] != result["ret_1d"].iloc[0]  # first row is NaN


def test_ret_20d_is_20_day_return() -> None:
    result = add_factors(_sample_df(n_symbols=1))
    row = result.iloc[20]
    assert row["ret_20d"] == pytest.approx((10.0 + 20 * 0.1) / 10.0 - 1)


def test_vol_20d_needs_20_days() -> None:
    result = add_factors(_sample_df(n_symbols=1))
    # ret_1d[0] is NaN (pct_change), so the first 20-window with 20 valid
    # observations closes at index 20.
    assert result["vol_20d"].iloc[:20].isna().all()
    assert not pd.isna(result["vol_20d"].iloc[20])


def test_all_factor_names_present() -> None:
    result = add_factors(_sample_df())
    for name in FACTOR_NAMES:
        assert name in result.columns


def test_factors_are_causal_under_truncation() -> None:
    df = _sample_df(n_symbols=2, n_days=60)
    full = add_factors(df)
    cutoff = df["trade_date"].min() + pd.Timedelta(days=29)
    truncated = add_factors(df[df["trade_date"] <= cutoff])

    full_early = full[full["trade_date"] <= cutoff].sort_values(["symbol", "trade_date"])
    truncated = truncated.sort_values(["symbol", "trade_date"])
    for name in FACTOR_NAMES:
        pd.testing.assert_series_equal(
            full_early[name].reset_index(drop=True),
            truncated[name].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )
