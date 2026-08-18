"""Three-segment train/validation/test split.

TRAIN is for fitting, VALID for scoring only (parameter selection), and TEST
for the final out-of-sample exam. The TEST segment must be *physically
isolated* — its data lives outside the runner's working directory so the model
cannot read it even by accident. This module defines the date boundaries and
the partition helper; physical placement is enforced where datasets are
assembled.

The split is date-based, daily, and long-only, matching A-share semantics
(T+1, no intraday; the reference crypto project used a different scheme).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from ashare_quant_core import DailyBar

SEGMENT_NAMES: tuple[str, ...] = ("train", "valid", "test")


@dataclass(frozen=True)
class ThreeSegmentSplit:
    """Date boundaries for the three segments.

    ``train_end`` and ``valid_end`` are inclusive; everything after
    ``valid_end`` is TEST.
    """

    train_end: date
    valid_end: date

    def __post_init__(self) -> None:
        if self.valid_end <= self.train_end:
            raise ValueError("valid_end must be strictly after train_end")

    def segment_for(self, d: date) -> str:
        """Return the segment name a trade date belongs to."""
        if d <= self.train_end:
            return "train"
        if d <= self.valid_end:
            return "valid"
        return "test"


def partition_bars(
    bars_by_symbol: Mapping[str, Sequence[DailyBar]],
    split: ThreeSegmentSplit,
) -> dict[str, dict[str, tuple[DailyBar, ...]]]:
    """Partition bars into ``{segment: {symbol: ordered bars}}``.

    Bars are sorted by trade date within each symbol; a symbol that has no bars
    in a segment is simply absent from that segment's mapping.
    """
    result: dict[str, dict[str, tuple[DailyBar, ...]]] = {name: {} for name in SEGMENT_NAMES}
    for symbol, bars in bars_by_symbol.items():
        ordered = sorted(bars, key=lambda bar: bar.trade_date)
        buckets: dict[str, list[DailyBar]] = {name: [] for name in SEGMENT_NAMES}
        for bar in ordered:
            buckets[split.segment_for(bar.trade_date)].append(bar)
        for name in SEGMENT_NAMES:
            if buckets[name]:
                result[name][symbol] = tuple(buckets[name])
    return result
