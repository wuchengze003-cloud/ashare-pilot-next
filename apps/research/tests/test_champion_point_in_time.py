from __future__ import annotations

import pandas as pd
import pytest
from ashare_research_app.champion import _merge_holder_number_as_of


def _panel(*dates: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["003007.SZ"] * len(dates),
            "trade_date": pd.to_datetime(list(dates)),
        }
    )


def test_holder_count_is_not_visible_before_announcement_date() -> None:
    panel = _panel("2024-07-01", "2024-08-27", "2024-08-28", "2024-08-29")
    holder_numbers = pd.DataFrame(
        {
            "ts_code": ["003007.SZ", "003007.SZ"],
            "ann_date": ["20240426", "20240828"],
            "end_date": ["20240331", "20240630"],
            "holder_num": [17312, 15500],
        }
    )

    result = _merge_holder_number_as_of(panel, holder_numbers)

    assert result["holder_num"].tolist() == [17312.0, 17312.0, 17312.0, 15500.0]
    assert result["holder_report_end_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-03-31",
        "2024-03-31",
        "2024-03-31",
        "2024-06-30",
    ]
    assert result["holder_ann_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-04-26",
        "2024-04-26",
        "2024-04-26",
        "2024-08-28",
    ]


def test_holder_count_rejects_missing_or_impossible_announcement_dates() -> None:
    panel = _panel("2024-07-08")
    holder_numbers = pd.DataFrame(
        {
            "ts_code": ["003007.SZ", "003007.SZ"],
            "ann_date": [None, "20240704"],
            "end_date": ["20240630", "20240705"],
            "holder_num": [15500, 70020],
        }
    )

    result = _merge_holder_number_as_of(panel, holder_numbers)

    assert pd.isna(result.iloc[0]["holder_num"])
    assert pd.isna(result.iloc[0]["holder_ann_date"])


def test_same_day_disclosures_prefer_the_latest_reporting_period() -> None:
    panel = _panel("2026-04-29", "2026-04-30")
    holder_numbers = pd.DataFrame(
        {
            "ts_code": ["003007.SZ", "003007.SZ"],
            "ann_date": ["20260429", "20260429"],
            "end_date": ["20251231", "20260331"],
            "holder_num": [15860, 16066],
        }
    )

    result = _merge_holder_number_as_of(panel, holder_numbers)

    assert pd.isna(result.iloc[0]["holder_num"])
    assert result.iloc[1]["holder_num"] == 16066
    assert result.iloc[1]["holder_report_end_date"] == pd.Timestamp("2026-03-31")


def test_holder_source_without_announcement_date_fails_closed() -> None:
    holder_numbers = pd.DataFrame(
        {
            "ts_code": ["003007.SZ"],
            "end_date": ["20240630"],
            "holder_num": [15500],
        }
    )

    with pytest.raises(ValueError, match="ann_date"):
        _merge_holder_number_as_of(_panel("2024-08-28"), holder_numbers)
