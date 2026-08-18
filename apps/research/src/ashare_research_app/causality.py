"""Look-ahead gate: full-data vs. truncated-data causality comparison.

A factor is causal (no look-ahead) iff its value at time ``t`` is identical
whether it is computed from the full series or from the series truncated at
``t``. Any discrepancy means the factor read the future — the classic case is a
z-score or rank normalized over the whole sample instead of a trailing window.

This is the self-built equivalent of the reference project's
``factor_causality_check.py``. It is more reliable than a backtest framework's
built-in lookahead detector because it directly compares two signal
computations rather than heuristically scanning boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ashare_quant_core import DailyBar

# A factor returns one value per input bar (None where it cannot be computed).
FactorFn = Callable[[Sequence[DailyBar]], Sequence[float | None]]


@dataclass(frozen=True)
class CausalityReport:
    """Outcome of a causality check."""

    passed: bool
    checked_points: int
    max_diff: float
    leak: tuple[str, int, float, float] | None  # symbol, index, full, truncated

    def describe(self) -> str:
        if self.passed:
            return (
                f"PASS: causal (checked {self.checked_points} points, "
                f"max diff {self.max_diff:.3e})"
            )
        symbol, index, full, truncated = self.leak  # type: ignore[misc]
        return (
            f"FAIL: look-ahead at {symbol} index {index} — "
            f"full={full:.6g} truncated={truncated:.6g}"
        )


def factor_causality_check(
    factor_fn: FactorFn,
    bars_by_symbol: Mapping[str, Sequence[DailyBar]],
    *,
    min_history: int = 30,
    sample_points: int = 10,
    tol: float = 1e-9,
) -> CausalityReport:
    """Detect look-ahead by comparing full-series vs. truncated-series signals.

    For each sampled timestamp ``t``, the factor is evaluated once on the full
    series and once on the series truncated at ``t``. If the two values at ``t``
    differ (beyond ``tol``), or if the truncated series cannot produce a value
    the full series can, the factor reads the future.
    """
    checked = 0
    max_diff = 0.0
    for symbol, bars in bars_by_symbol.items():
        ordered = sorted(bars, key=lambda bar: bar.trade_date)
        if len(ordered) < min_history:
            continue
        full = factor_fn(ordered)
        if len(full) != len(ordered):
            raise ValueError("factor_fn must return exactly one value per input bar")
        for t in _sample_points(len(ordered), min_history, sample_points):
            full_value = full[t]
            if full_value is None or not math.isfinite(full_value):
                continue
            truncated = factor_fn(ordered[: t + 1])
            truncated_value = truncated[t]
            if truncated_value is None or not math.isfinite(truncated_value):
                return CausalityReport(
                    passed=False,
                    checked_points=checked,
                    max_diff=math.inf,
                    leak=(symbol, t, float(full_value), math.nan),
                )
            checked += 1
            diff = abs(float(full_value) - float(truncated_value))
            max_diff = max(max_diff, diff)
            if diff > tol:
                return CausalityReport(
                    passed=False,
                    checked_points=checked,
                    max_diff=max_diff,
                    leak=(symbol, t, float(full_value), float(truncated_value)),
                )
    return CausalityReport(
        passed=True, checked_points=checked, max_diff=max_diff, leak=None
    )


def _sample_points(length: int, min_history: int, sample_points: int) -> list[int]:
    """Choose evenly-spaced indices in ``[min_history, length)``."""
    if length <= min_history:
        return []
    if length - min_history < sample_points:
        return list(range(min_history, length))
    step = (length - 1 - min_history) / (sample_points - 1)
    return [min_history + round(step * k) for k in range(sample_points)]
