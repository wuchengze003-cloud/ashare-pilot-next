"""Authoritative, deterministic financial semantics."""

from .costs import CostBreakdown, calculate_cost, select_cost_segment
from .portfolio import validate_portfolio_targets
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
    "select_cost_segment",
    "validate_portfolio_targets",
    "validate_target_transition",
]
