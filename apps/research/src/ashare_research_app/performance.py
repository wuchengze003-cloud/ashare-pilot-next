"""Shared portfolio-performance calculations for research and promotion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def returns_from_nav_curve(
    nav_curve: Sequence[Mapping[str, object]],
    *,
    field: str = "nav",
) -> tuple[float, ...]:
    values = tuple(float(point[field]) for point in nav_curve)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{field} curve must contain finite positive values")
    return tuple(
        values[index] / values[index - 1] - 1.0
        for index in range(1, len(values))
    )


def annualized_sharpe(returns: Sequence[float], *, annualization: int = 252) -> float:
    materialized = tuple(float(value) for value in returns)
    if annualization < 1:
        raise ValueError("annualization must be positive")
    if any(not math.isfinite(value) for value in materialized):
        raise ValueError("returns must be finite")
    if len(materialized) < 2:
        return 0.0
    mean = sum(materialized) / len(materialized)
    variance = sum((value - mean) ** 2 for value in materialized) / (
        len(materialized) - 1
    )
    return mean / math.sqrt(variance) * math.sqrt(annualization) if variance > 0 else 0.0
