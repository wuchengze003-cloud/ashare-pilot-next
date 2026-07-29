"""Deterministic portfolio constraints shared by research and production."""

from collections.abc import Collection, Sequence
from math import fsum

from .strategy import TargetPosition


def validate_portfolio_targets(
    targets: Sequence[TargetPosition],
    *,
    eligible_symbols: Collection[str],
    max_positions: int,
    max_single_weight: float,
    max_gross_exposure: float,
) -> None:
    """Reject targets that exceed the bound universe or portfolio contract."""
    if max_positions < 1:
        raise ValueError("max_positions must be positive")
    if not 0 < max_single_weight <= 1:
        raise ValueError("max_single_weight must be in (0, 1]")
    if not 0 <= max_gross_exposure <= 1:
        raise ValueError("max_gross_exposure must be in [0, 1]")

    symbols = [target.symbol for target in targets]
    if len(symbols) != len(set(symbols)):
        raise ValueError("target symbols must be unique")
    if len(targets) > max_positions:
        raise ValueError("target position count exceeds portfolio limit")

    unknown_symbols = sorted(set(symbols) - set(eligible_symbols))
    if unknown_symbols:
        raise ValueError(f"targets contain ineligible symbols: {unknown_symbols}")
    if any(target.target_weight > max_single_weight for target in targets):
        raise ValueError("target weight exceeds single-position limit")
    if fsum(target.target_weight for target in targets) > max_gross_exposure + 1e-12:
        raise ValueError("target weights exceed gross-exposure limit")
