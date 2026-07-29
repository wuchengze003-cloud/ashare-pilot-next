"""Authoritative, deterministic financial semantics."""

from .costs import CostBreakdown, calculate_cost
from .signals import validate_target_transition
from .state import ChampionHealth, HealthSnapshot, RiskAction, RuntimeState, resolve_state
from .strategy import Strategy, TargetPosition

__all__ = [
    "ChampionHealth",
    "CostBreakdown",
    "HealthSnapshot",
    "RiskAction",
    "RuntimeState",
    "Strategy",
    "TargetPosition",
    "calculate_cost",
    "resolve_state",
    "validate_target_transition",
]
