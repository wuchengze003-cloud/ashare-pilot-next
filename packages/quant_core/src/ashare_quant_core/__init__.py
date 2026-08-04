"""Authoritative, deterministic financial semantics."""

from .costs import CostBreakdown, calculate_cost, select_cost_segment
from .portfolio import validate_portfolio_targets
from .signals import validate_target_transition
from .sim_portfolio import (
    DailyBarView,
    ExecutionDay,
    Holding,
    MarketRuleSegment,
    SimulatedExecutionError,
    SimulatedPortfolioState,
    SimulatedSkip,
    SimulatedTrade,
    classify_board,
    execute_buy,
    execute_sell,
    is_limit_down,
    is_limit_up,
    mark_to_market,
    parse_market_rules,
    settle_t_plus_one,
)
from .snapshots import (
    DAILY_BAR_SCHEMA_ID,
    DAILY_BAR_SCHEMA_SHA256,
    DailyBar,
    DatasetSnapshot,
    UniverseMember,
    UniverseSnapshot,
)
from .state import ChampionHealth, HealthSnapshot, RiskAction, RuntimeState, resolve_state
from .strategy import Strategy, TargetPosition

__all__ = [
    "ChampionHealth",
    "CostBreakdown",
    "DAILY_BAR_SCHEMA_ID",
    "DAILY_BAR_SCHEMA_SHA256",
    "DailyBar",
    "DailyBarView",
    "DatasetSnapshot",
    "ExecutionDay",
    "HealthSnapshot",
    "Holding",
    "MarketRuleSegment",
    "RiskAction",
    "RuntimeState",
    "SimulatedExecutionError",
    "SimulatedPortfolioState",
    "SimulatedSkip",
    "SimulatedTrade",
    "Strategy",
    "TargetPosition",
    "UniverseMember",
    "UniverseSnapshot",
    "calculate_cost",
    "classify_board",
    "execute_buy",
    "execute_sell",
    "is_limit_down",
    "is_limit_up",
    "mark_to_market",
    "parse_market_rules",
    "resolve_state",
    "select_cost_segment",
    "settle_t_plus_one",
    "validate_portfolio_targets",
    "validate_target_transition",
]
