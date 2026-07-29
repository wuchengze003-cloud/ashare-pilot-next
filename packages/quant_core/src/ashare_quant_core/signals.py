"""Cross-signal target semantics shared by research and production."""

from collections.abc import Sequence

from .state import RuntimeState
from .strategy import TargetPosition


def _target_map(positions: Sequence[TargetPosition], *, label: str) -> dict[str, float]:
    targets: dict[str, float] = {}
    for position in positions:
        if position.symbol in targets:
            raise ValueError(f"{label} contains duplicate symbol {position.symbol}")
        targets[position.symbol] = position.target_weight
    if sum(targets.values()) > 1 + 1e-12:
        raise ValueError(f"{label} target weights exceed 100%")
    return targets


def validate_target_transition(
    *,
    state: RuntimeState,
    current: Sequence[TargetPosition],
    previous: Sequence[TargetPosition] | None,
) -> None:
    """Enforce state behavior against the previous effective target."""
    current_targets = _target_map(current, label="current")
    previous_targets = (
        _target_map(previous, label="previous") if previous is not None else None
    )

    if state is RuntimeState.FLAT:
        if current_targets:
            raise ValueError("FLAT requires an empty target")
        return

    if state is RuntimeState.HOLD:
        if previous_targets is None:
            raise ValueError("HOLD requires a previous target")
        if current_targets != previous_targets:
            raise ValueError("HOLD must preserve the previous target")
        return

    if state is RuntimeState.REDUCE_ONLY:
        if previous_targets is None:
            raise ValueError("REDUCE_ONLY requires a previous target")
        new_symbols = current_targets.keys() - previous_targets.keys()
        if new_symbols:
            raise ValueError(
                f"REDUCE_ONLY cannot add symbols: {', '.join(sorted(new_symbols))}"
            )
        increased = sorted(
            symbol
            for symbol, weight in current_targets.items()
            if weight > previous_targets[symbol] + 1e-12
        )
        if increased:
            raise ValueError(
                f"REDUCE_ONLY cannot increase weights: {', '.join(increased)}"
            )
