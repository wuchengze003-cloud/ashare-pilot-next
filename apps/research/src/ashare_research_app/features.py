"""Point-in-time feature engineering for the baseline ML pilot.

Every feature at time t uses only bars dated on or before t. The panel
is built from an immutable DatasetSnapshot through its own as_of gate,
so future rows cannot enter by construction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from ashare_quant_core import DailyBar, DatasetSnapshot

FEATURE_NAMES: tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_ratio_5d",
    "vol_ratio_20d",
    "amount_z_20d",
    "close_to_high_20d",
    "range_position_20d",
)
MIN_OBSERVATIONS = 21


@dataclass(frozen=True, order=True)
class FeatureRow:
    symbol: str
    trade_date: date
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(FEATURE_NAMES):
            raise ValueError("feature row width does not match FEATURE_NAMES")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("feature row must be finite")


def _return(history: Sequence[DailyBar], index: int, lookback: int) -> float:
    base = history[index - lookback].close
    return history[index].close / base - 1


def compute_feature_row(
    history: Sequence[DailyBar],
    index: int,
) -> tuple[float, ...] | None:
    """Compute one feature vector using only bars up to `index` inclusive."""
    if index + 1 < MIN_OBSERVATIONS:
        return None
    current = history[index]

    recent_5 = history[index - 4 : index + 1]
    recent_20 = history[index - 19 : index + 1]

    average_volume_5 = sum(bar.volume for bar in recent_5) / 5
    average_volume_20 = sum(bar.volume for bar in recent_20) / 20
    vol_ratio_5 = current.volume / average_volume_5 if average_volume_5 > 0 else 1.0
    vol_ratio_20 = current.volume / average_volume_20 if average_volume_20 > 0 else 1.0

    average_amount_20 = sum(bar.amount for bar in recent_20) / 20
    variance = sum((bar.amount - average_amount_20) ** 2 for bar in recent_20) / 20
    std_amount_20 = math.sqrt(variance)
    amount_z_20 = (
        (current.amount - average_amount_20) / std_amount_20 if std_amount_20 > 0 else 0.0
    )

    high_20 = max(bar.high for bar in recent_20)
    low_20 = min(bar.low for bar in recent_20)
    close_to_high_20 = current.close / high_20
    span = high_20 - low_20
    range_position_20 = (current.close - low_20) / span if span > 0 else 0.5

    return (
        _return(history, index, 1),
        _return(history, index, 5),
        _return(history, index, 10),
        _return(history, index, 20),
        vol_ratio_5,
        vol_ratio_20,
        amount_z_20,
        close_to_high_20,
        range_position_20,
    )


def build_feature_panel(
    snapshot: DatasetSnapshot,
    *,
    as_of: date | None = None,
) -> tuple[FeatureRow, ...]:
    """Build the PIT feature panel visible at as_of (defaults to snapshot as_of)."""
    cutoff = as_of if as_of is not None else snapshot.as_of
    if cutoff > snapshot.as_of:
        raise ValueError("feature panel cannot request data after the snapshot as_of")

    by_symbol: dict[str, list[DailyBar]] = {}
    for bar in snapshot.bars(through=cutoff):
        if bar.trade_date > cutoff:
            raise ValueError("snapshot returned a future bar")
        by_symbol.setdefault(bar.symbol, []).append(bar)

    rows: list[FeatureRow] = []
    for symbol in sorted(by_symbol):
        history = sorted(by_symbol[symbol], key=lambda bar: bar.trade_date)
        for index in range(len(history)):
            values = compute_feature_row(history, index)
            if values is None:
                continue
            rows.append(
                FeatureRow(symbol=symbol, trade_date=history[index].trade_date, values=values)
            )
    return tuple(sorted(rows))


def forward_return_label(
    history: Sequence[DailyBar],
    index: int,
    horizon: int,
) -> float | None:
    """Return the close-to-close return over the next `horizon` sessions."""
    if horizon < 1:
        raise ValueError("label horizon must be positive")
    target = index + horizon
    if target >= len(history):
        return None
    return history[target].close / history[index].close - 1
