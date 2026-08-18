from datetime import date

import pytest
from ashare_quant_core import DailyBar
from ashare_research_app.split import ThreeSegmentSplit, partition_bars


def _bar(symbol: str, trade_date: date, close: float = 10.0) -> DailyBar:
    return DailyBar(
        symbol=symbol,
        trade_date=trade_date,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
        amount=1000.0 * close,
    )


def test_segment_for_uses_inclusive_boundaries() -> None:
    split = ThreeSegmentSplit(train_end=date(2023, 6, 30), valid_end=date(2024, 6, 30))
    assert split.segment_for(date(2023, 1, 1)) == "train"
    assert split.segment_for(date(2023, 6, 30)) == "train"
    assert split.segment_for(date(2023, 7, 1)) == "valid"
    assert split.segment_for(date(2024, 6, 30)) == "valid"
    assert split.segment_for(date(2024, 7, 1)) == "test"


def test_valid_end_must_follow_train_end() -> None:
    with pytest.raises(ValueError):
        ThreeSegmentSplit(train_end=date(2024, 6, 30), valid_end=date(2024, 6, 30))


def test_partition_bars_places_each_symbol_in_its_segment() -> None:
    split = ThreeSegmentSplit(train_end=date(2023, 6, 30), valid_end=date(2023, 12, 31))
    bars = {
        "600000.SH": [
            _bar("600000.SH", date(2023, 3, 1)),
            _bar("600000.SH", date(2023, 8, 1)),
            _bar("600000.SH", date(2024, 1, 1)),
        ],
        "000001.SZ": [
            _bar("000001.SZ", date(2023, 5, 1)),
        ],
    }
    result = partition_bars(bars, split)
    assert set(result["train"]) == {"600000.SH", "000001.SZ"}
    assert set(result["valid"]) == {"600000.SH"}
    assert set(result["test"]) == {"600000.SH"}
    assert len(result["train"]["600000.SH"]) == 1
    assert result["valid"]["600000.SH"][0].trade_date == date(2023, 8, 1)
