"""Deterministic feature replay over an immutable point-in-time snapshot."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date

from ashare_quant_core import DailyBar, DatasetSnapshot


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


@dataclass(frozen=True, order=True)
class ReplayFeature:
    symbol: str
    trade_date: date
    close_return_1d: float | None
    volume_ratio_5d: float | None

    def __post_init__(self) -> None:
        for field_name in ("close_return_1d", "volume_ratio_5d"):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")


@dataclass(frozen=True)
class FeatureReplay:
    as_of: date
    dataset_snapshot_sha256: str
    rows: tuple[ReplayFeature, ...]
    replay_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.rows))
        if any(row.trade_date > self.as_of for row in ordered):
            raise ValueError("feature replay cannot contain rows after as_of")
        object.__setattr__(self, "rows", ordered)
        payload = {
            "as_of": self.as_of.isoformat(),
            "dataset_snapshot_sha256": self.dataset_snapshot_sha256,
            "rows": [
                {**asdict(row), "trade_date": row.trade_date.isoformat()}
                for row in ordered
            ],
        }
        object.__setattr__(
            self,
            "replay_sha256",
            hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        )

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(
            {
                "as_of": self.as_of.isoformat(),
                "dataset_snapshot_sha256": self.dataset_snapshot_sha256,
                "replay_sha256": self.replay_sha256,
                "rows": [
                    {**asdict(row), "trade_date": row.trade_date.isoformat()}
                    for row in self.rows
                ],
            }
        )


def build_feature_replay(snapshot: DatasetSnapshot) -> FeatureReplay:
    """Build a stable panel without any data beyond the snapshot cutoff."""
    bars = snapshot.bars(through=snapshot.as_of)
    if any(bar.trade_date > snapshot.as_of for bar in bars):
        raise ValueError("snapshot returned a future bar")

    rows: list[ReplayFeature] = []
    by_symbol: dict[str, list[DailyBar]] = {}
    for bar in bars:
        by_symbol.setdefault(bar.symbol, []).append(bar)
    for symbol in sorted(by_symbol):
        history = sorted(by_symbol[symbol], key=lambda item: item.trade_date)
        for index, bar in enumerate(history):
            previous = history[index - 1] if index else None
            close_return = (
                round(bar.close / previous.close - 1, 12)
                if previous is not None
                else None
            )
            trailing = history[max(0, index - 4) : index + 1]
            average_volume = sum(item.volume for item in trailing) / len(trailing)
            volume_ratio = (
                round(bar.volume / average_volume, 12)
                if average_volume > 0
                else None
            )
            rows.append(
                ReplayFeature(
                    symbol=symbol,
                    trade_date=bar.trade_date,
                    close_return_1d=close_return,
                    volume_ratio_5d=volume_ratio,
                )
            )
    return FeatureReplay(
        as_of=snapshot.as_of,
        dataset_snapshot_sha256=snapshot.snapshot_sha256,
        rows=tuple(rows),
    )
