"""Stable strategy protocol shared by Research and Signal Runner."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from .snapshots import DatasetSnapshot, UniverseSnapshot


@dataclass(frozen=True)
class TargetPosition:
    symbol: str
    target_weight: float

    def __post_init__(self) -> None:
        if not 0 <= self.target_weight <= 1:
            raise ValueError("target_weight must be between 0 and 1")


class Strategy(Protocol):
    strategy_id: str
    strategy_version: str
    adapter_id: str
    adapter_sha256: str

    def target_positions(
        self,
        *,
        as_of: date,
        dataset_snapshot: DatasetSnapshot,
        universe_snapshot: UniverseSnapshot,
    ) -> Sequence[TargetPosition]:
        """Return deterministic targets for explicit immutable inputs."""
        ...
