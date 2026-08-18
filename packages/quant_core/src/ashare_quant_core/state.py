"""Deterministic production-state resolution."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class RuntimeState(StrEnum):
    ACTIVE = "ACTIVE"
    HOLD = "HOLD"
    REDUCE_ONLY = "REDUCE_ONLY"
    FLAT = "FLAT"


class ChampionHealth(StrEnum):
    HEALTHY = "HEALTHY"
    NEVER_ACTIVATED = "NEVER_ACTIVATED"
    WITHDRAWN = "WITHDRAWN"


class RiskAction(StrEnum):
    NONE = "NONE"
    REDUCE = "REDUCE"
    FLAT = "FLAT"


@dataclass(frozen=True)
class HealthSnapshot:
    execution_data_valid: bool
    previous_target_known: bool
    decision_data_valid: bool
    contracts_valid: bool
    hashes_valid: bool
    champion: ChampionHealth
    risk_action: RiskAction = RiskAction.NONE


def resolve_state(snapshot: HealthSnapshot) -> RuntimeState:
    """Resolve concurrent failures in the documented priority order."""
    if not snapshot.execution_data_valid or not snapshot.previous_target_known:
        return RuntimeState.HOLD
    if snapshot.risk_action is RiskAction.FLAT:
        return RuntimeState.FLAT
    if snapshot.champion is ChampionHealth.WITHDRAWN or snapshot.risk_action is RiskAction.REDUCE:
        return RuntimeState.REDUCE_ONLY
    if (
        not snapshot.decision_data_valid
        or not snapshot.contracts_valid
        or not snapshot.hashes_valid
    ):
        return RuntimeState.HOLD
    if snapshot.champion is ChampionHealth.NEVER_ACTIVATED:
        return RuntimeState.FLAT
    return RuntimeState.ACTIVE


def constrain_execution_targets(
    *,
    state: RuntimeState,
    desired_shares: Mapping[str, int],
    current_shares: Mapping[str, int],
) -> dict[str, int]:
    """Apply production-state risk constraints to executable share targets.

    Signal Runner owns target-position publication.  Execution consumers still
    enforce the same state semantics as a defense-in-depth boundary so a
    malformed degraded signal can never increase simulated risk.
    """
    if any(shares < 0 for shares in desired_shares.values()):
        raise ValueError("desired shares must be non-negative")
    if any(shares < 0 for shares in current_shares.values()):
        raise ValueError("current shares must be non-negative")

    symbols = set(desired_shares) | set(current_shares)
    if state is RuntimeState.ACTIVE:
        return {symbol: desired_shares.get(symbol, 0) for symbol in symbols}
    if state is RuntimeState.HOLD:
        return {symbol: current_shares.get(symbol, 0) for symbol in symbols}
    if state is RuntimeState.REDUCE_ONLY:
        return {
            symbol: min(desired_shares.get(symbol, 0), current_shares.get(symbol, 0))
            for symbol in symbols
        }
    if state is RuntimeState.FLAT:
        return {symbol: 0 for symbol in symbols}
    raise ValueError(f"unsupported runtime state: {state}")
