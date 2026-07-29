"""Deterministic production-state resolution."""

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
