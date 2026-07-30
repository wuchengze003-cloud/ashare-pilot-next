"""Synthetic adapter used only by Signal Runner contract tests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import date

from ashare_quant_core import (
    DatasetSnapshot,
    TargetPosition,
    UniverseSnapshot,
)


class FixtureReferenceStrategy:
    strategy_id = "fixture-reference"
    strategy_version = "v1"

    def __init__(self, *, target_weight: float, forced_symbol: str | None) -> None:
        self._target_weight = target_weight
        self._forced_symbol = forced_symbol

    def target_positions(
        self,
        *,
        as_of: date,
        dataset_snapshot: DatasetSnapshot,
        universe_snapshot: UniverseSnapshot,
    ) -> tuple[TargetPosition, ...]:
        if self._forced_symbol is not None:
            return (TargetPosition(self._forced_symbol, self._target_weight),)

        history: dict[str, list[tuple[date, float]]] = defaultdict(list)
        for bar in dataset_snapshot.bars(through=as_of):
            if bar.symbol in universe_snapshot.eligible_symbols:
                history[bar.symbol].append((bar.trade_date, bar.close))

        momentum: list[tuple[float, str]] = []
        for symbol, observations in history.items():
            ordered = sorted(observations)
            if len(ordered) >= 2:
                momentum.append((ordered[-1][1] / ordered[-2][1] - 1, symbol))
        if not momentum:
            return ()
        _, selected = max(momentum, key=lambda item: (item[0], item[1]))
        return (TargetPosition(selected, self._target_weight),)


def build_strategy(config: Mapping[str, object]) -> FixtureReferenceStrategy:
    target_weight = float(config["target_weight"])
    raw_forced_symbol = config.get("forced_symbol")
    forced_symbol = str(raw_forced_symbol) if raw_forced_symbol is not None else None
    return FixtureReferenceStrategy(
        target_weight=target_weight,
        forced_symbol=forced_symbol,
    )
