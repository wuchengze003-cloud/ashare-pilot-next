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
from scipy.stats import rankdata

from .feature_datasets import (
    DAILY_AUXILIARY_FEATURE_NAMES,
    HOLDER_FEATURE_NAMES,
    FeatureDataset,
)

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
MARKET_AUXILIARY_FEATURE_NAMES: tuple[str, ...] = (
    *FEATURE_NAMES,
    *DAILY_AUXILIARY_FEATURE_NAMES,
    *HOLDER_FEATURE_NAMES,
)
MIN_OBSERVATIONS = 21
FEATURE_TRANSFORMS: tuple[str, ...] = (
    "raw",
    "cross_sectional_rank",
    "market_auxiliary_rank",
)


def feature_names_for_transform(feature_transform: str) -> tuple[str, ...]:
    if feature_transform == "market_auxiliary_rank":
        return MARKET_AUXILIARY_FEATURE_NAMES
    if feature_transform in {"raw", "cross_sectional_rank"}:
        return FEATURE_NAMES
    raise ValueError(f"unsupported feature_transform: {feature_transform}")


@dataclass(frozen=True, order=True)
class FeatureRow:
    symbol: str
    trade_date: date
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        allowed_widths = {len(FEATURE_NAMES), len(MARKET_AUXILIARY_FEATURE_NAMES)}
        if len(self.values) not in allowed_widths:
            raise ValueError("feature row width is not registered")
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
    feature_transform: str = "raw",
    feature_dataset: FeatureDataset | None = None,
) -> tuple[FeatureRow, ...]:
    """Build the PIT feature panel visible at as_of (defaults to snapshot as_of)."""
    if feature_transform not in FEATURE_TRANSFORMS:
        raise ValueError(f"unsupported feature_transform: {feature_transform}")
    cutoff = as_of if as_of is not None else snapshot.as_of
    if cutoff > snapshot.as_of:
        raise ValueError("feature panel cannot request data after the snapshot as_of")
    if feature_transform == "market_auxiliary_rank":
        if feature_dataset is None:
            raise ValueError("market_auxiliary_rank requires an immutable feature dataset")
        feature_dataset.assert_matches_snapshot(snapshot)
    elif feature_dataset is not None:
        raise ValueError("feature_dataset is only valid for market_auxiliary_rank")

    by_symbol: dict[str, list[DailyBar]] = {}
    for bar in snapshot.bars(through=cutoff):
        if bar.trade_date > cutoff:
            raise ValueError("snapshot returned a future bar")
        by_symbol.setdefault(bar.symbol, []).append(bar)

    rows: list[tuple[str, date, tuple[float | None, ...]]] = []
    for symbol in sorted(by_symbol):
        history = sorted(by_symbol[symbol], key=lambda bar: bar.trade_date)
        for index in range(len(history)):
            values = compute_feature_row(history, index)
            if values is None:
                continue
            if feature_dataset is not None:
                daily_values = feature_dataset.daily_values(
                    symbol=symbol,
                    trade_date=history[index].trade_date,
                )
                holder_values = feature_dataset.holder_values(
                    symbol=symbol,
                    through=history[index].trade_date,
                )
                values = (*values, *daily_values, *holder_values)
            rows.append((symbol, history[index].trade_date, values))
    ordered = tuple(sorted(rows, key=lambda row: (row[0], row[1])))
    if feature_transform == "raw":
        raw_rows: list[FeatureRow] = []
        for symbol, trade_date, values in ordered:
            if any(value is None for value in values):
                raise ValueError("raw base features cannot be missing")
            raw_rows.append(
                FeatureRow(
                    symbol=symbol,
                    trade_date=trade_date,
                    values=tuple(float(value) for value in values),
                )
            )
        return tuple(raw_rows)

    rows_by_date: dict[date, list[tuple[str, tuple[float | None, ...]]]] = {}
    for symbol, trade_date, values in ordered:
        rows_by_date.setdefault(trade_date, []).append((symbol, values))
    transformed: list[FeatureRow] = []
    for trade_date in sorted(rows_by_date):
        daily_rows = rows_by_date[trade_date]
        columns = tuple(zip(*(values for _symbol, values in daily_rows), strict=True))
        ranked_columns: list[tuple[float, ...]] = []
        for column in columns:
            available_positions = [
                position for position, value in enumerate(column) if value is not None
            ]
            available_values = [float(column[position]) for position in available_positions]
            ranked = [0.0] * len(column)
            if len(available_values) > 1:
                ranks = rankdata(available_values, method="average")
                denominator = len(available_values) - 1
                for position, rank in zip(available_positions, ranks, strict=True):
                    ranked[position] = (float(rank) - 1.0) / denominator - 0.5
            ranked_columns.append(tuple(ranked))
        for index, (symbol, _values) in enumerate(daily_rows):
            transformed.append(
                FeatureRow(
                    symbol=symbol,
                    trade_date=trade_date,
                    values=tuple(column[index] for column in ranked_columns),
                )
            )
    return tuple(sorted(transformed))


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
