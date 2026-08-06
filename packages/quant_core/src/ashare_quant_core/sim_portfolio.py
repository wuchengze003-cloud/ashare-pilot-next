"""Deterministic simulated-portfolio execution semantics.

This module models a paper portfolio only. It never claims broker
holdings, orders, or fills. All inputs are explicit; no wall-clock time
is read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from math import fsum

from .costs import CostBreakdown, calculate_cost
from .snapshots import SYMBOL_PATTERN


class SimulatedExecutionError(ValueError):
    """Raised for invalid simulated orders or market inputs."""


def classify_board(symbol: str) -> str:
    """Map an A-share symbol to its market-rules board."""
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise SimulatedExecutionError(f"invalid A-share symbol: {symbol}")
    code, exchange = symbol.split(".", 1)
    if exchange == "BJ":
        raise SimulatedExecutionError(f"Beijing exchange is not supported: {symbol}")
    if code.startswith("688"):
        return "star"
    if code.startswith(("300", "301", "302")):
        return "gem"
    return "main"


@dataclass(frozen=True)
class MarketRuleSegment:
    board: str
    lot_size: int
    price_limit_pct: float

    def __post_init__(self) -> None:
        if self.lot_size < 1:
            raise SimulatedExecutionError("lot_size must be positive")
        if not 0 < self.price_limit_pct <= 0.5:
            raise SimulatedExecutionError("price_limit_pct must be in (0, 0.5]")


def parse_market_rules(document: Mapping[str, object]) -> dict[str, MarketRuleSegment]:
    """Extract per-board trading rules from a market-rules contract."""
    raw_segments = document.get("segments")
    if not isinstance(raw_segments, list):
        raise SimulatedExecutionError("market rules segments must be a list")
    rules: dict[str, MarketRuleSegment] = {}
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            raise SimulatedExecutionError("market rules segment must be a mapping")
        board = str(raw["board"])
        if raw.get("security_status") != "normal":
            continue
        if board in rules:
            raise SimulatedExecutionError(f"duplicate market rules board: {board}")
        rules[board] = MarketRuleSegment(
            board=board,
            lot_size=int(raw["lot_size"]),
            price_limit_pct=float(raw["price_limit_pct"]),
        )
    if not rules:
        raise SimulatedExecutionError("market rules contain no normal segments")
    return rules


@dataclass(frozen=True)
class Holding:
    shares: int
    locked_shares: int
    avg_cost: Decimal

    def __post_init__(self) -> None:
        if self.shares < 0 or self.locked_shares < 0:
            raise SimulatedExecutionError("holding quantities must be non-negative")
        if self.locked_shares > self.shares:
            raise SimulatedExecutionError("locked shares exceed held shares")
        if self.shares > 0 and self.avg_cost <= 0:
            raise SimulatedExecutionError("avg_cost must be positive for holdings")


@dataclass(frozen=True)
class SimulatedPortfolioState:
    cash: Decimal
    holdings: Mapping[str, Holding]

    def __post_init__(self) -> None:
        if self.cash < 0:
            raise SimulatedExecutionError("cash must be non-negative")
        for symbol in self.holdings:
            if not SYMBOL_PATTERN.fullmatch(symbol):
                raise SimulatedExecutionError(f"invalid holding symbol: {symbol}")

    @property
    def total_invested_shares(self) -> int:
        return sum(holding.shares for holding in self.holdings.values())


@dataclass(frozen=True)
class SimulatedTrade:
    trade_date: date
    symbol: str
    side: str
    shares: int
    price: float
    gross_amount: Decimal
    cost: CostBreakdown
    reason: str


@dataclass(frozen=True)
class SimulatedSkip:
    symbol: str
    side: str
    reason_code: str


@dataclass(frozen=True)
class DailyBarView:
    open: float
    close: float


@dataclass(frozen=True)
class ExecutionDay:
    """Explicit market inputs for one simulated trading day."""

    trade_date: date
    bars: Mapping[str, DailyBarView | None]
    previous_closes: Mapping[str, float]
    slippage_bps: int

    def __post_init__(self) -> None:
        if self.slippage_bps < 0:
            raise SimulatedExecutionError("slippage_bps must be non-negative")


def _limit_tolerance() -> float:
    return 1e-9


def is_limit_up(*, price: float, previous_close: float, limit_pct: float) -> bool:
    """A one-way limit-up blocks new buys in the simulation."""
    return price >= previous_close * (1 + limit_pct) - _limit_tolerance()


def is_limit_down(*, price: float, previous_close: float, limit_pct: float) -> bool:
    """A one-way limit-down blocks sells in the simulation."""
    return price <= previous_close * (1 - limit_pct) + _limit_tolerance()


def settle_t_plus_one(
    state: SimulatedPortfolioState,
    *,
    trade_date: date,
    buy_dates: Mapping[str, date],
) -> SimulatedPortfolioState:
    """Unlock shares bought strictly before the current trade date."""
    unlocked: dict[str, Holding] = {}
    for symbol, holding in state.holdings.items():
        bought_on = buy_dates.get(symbol)
        if holding.locked_shares and bought_on is not None and bought_on < trade_date:
            unlocked[symbol] = Holding(
                shares=holding.shares,
                locked_shares=0,
                avg_cost=holding.avg_cost,
            )
        else:
            unlocked[symbol] = holding
    return SimulatedPortfolioState(cash=state.cash, holdings=unlocked)


def execute_buy(
    *,
    state: SimulatedPortfolioState,
    day: ExecutionDay,
    symbol: str,
    requested_shares: int,
    rules: Mapping[str, MarketRuleSegment],
    cost_model: Mapping[str, object],
    reason: str,
) -> tuple[SimulatedPortfolioState, SimulatedTrade | None, SimulatedSkip | None, date | None]:
    """Execute one simulated buy at the open with slippage and costs."""
    board = classify_board(symbol)
    segment = rules.get(board)
    if segment is None:
        raise SimulatedExecutionError(f"no market rules for board: {board}")
    bar = day.bars.get(symbol)
    if bar is None:
        return state, None, SimulatedSkip(symbol, "buy", "SUSPENDED_NO_BAR"), None
    previous_close = day.previous_closes.get(symbol)
    if previous_close is None or previous_close <= 0:
        raise SimulatedExecutionError(f"previous close is required for {symbol}")
    limit_up = is_limit_up(
        price=bar.open,
        previous_close=previous_close,
        limit_pct=segment.price_limit_pct,
    )
    if limit_up:
        return state, None, SimulatedSkip(symbol, "buy", "LIMIT_UP_NO_FILL"), None

    slippage = Decimal(1) + Decimal(day.slippage_bps) / Decimal(10_000)
    execution_price = float(Decimal(str(bar.open)) * slippage)
    lot = segment.lot_size
    shares = (requested_shares // lot) * lot
    market = symbol.rsplit(".", 1)[-1]

    while shares >= lot:
        gross = Decimal(shares) * Decimal(str(execution_price))
        cost = calculate_cost(
            side="buy",
            trade_date=day.trade_date,
            market=market,
            gross_amount=gross,
            model=cost_model,
        )
        if cost.total + gross <= state.cash:
            break
        shares -= lot
    if shares < lot:
        return state, None, SimulatedSkip(symbol, "buy", "INSUFFICIENT_CASH"), None

    gross = Decimal(shares) * Decimal(str(execution_price))
    cost = calculate_cost(
        side="buy",
        trade_date=day.trade_date,
        market=market,
        gross_amount=gross,
        model=cost_model,
    )
    holding = state.holdings.get(symbol)
    if holding is None:
        avg_cost = (gross + cost.total) / Decimal(shares)
        new_holding = Holding(shares=shares, locked_shares=shares, avg_cost=avg_cost)
    else:
        total_cost_basis = holding.avg_cost * Decimal(holding.shares) + gross + cost.total
        new_shares = holding.shares + shares
        new_holding = Holding(
            shares=new_shares,
            locked_shares=holding.locked_shares + shares,
            avg_cost=total_cost_basis / Decimal(new_shares),
        )
    new_state = SimulatedPortfolioState(
        cash=state.cash - gross - cost.total,
        holdings={**state.holdings, symbol: new_holding},
    )
    trade = SimulatedTrade(
        trade_date=day.trade_date,
        symbol=symbol,
        side="buy",
        shares=shares,
        price=execution_price,
        gross_amount=gross,
        cost=cost,
        reason=reason,
    )
    return new_state, trade, None, day.trade_date


def execute_sell(
    *,
    state: SimulatedPortfolioState,
    day: ExecutionDay,
    symbol: str,
    rules: Mapping[str, MarketRuleSegment],
    cost_model: Mapping[str, object],
    reason: str,
    max_shares: int | None = None,
) -> tuple[SimulatedPortfolioState, SimulatedTrade | None, SimulatedSkip | None]:
    """Sell unlocked shares (all by default) at the open with slippage."""
    board = classify_board(symbol)
    segment = rules.get(board)
    if segment is None:
        raise SimulatedExecutionError(f"no market rules for board: {board}")
    holding = state.holdings.get(symbol)
    if holding is None or holding.shares == 0:
        return state, None, SimulatedSkip(symbol, "sell", "NO_HOLDING")
    sellable = holding.shares - holding.locked_shares
    if max_shares is not None:
        if max_shares < 0:
            raise SimulatedExecutionError("max_shares must be non-negative")
        sellable = min(sellable, max_shares)
    if sellable <= 0:
        return state, None, SimulatedSkip(symbol, "sell", "T1_LOCKED")
    bar = day.bars.get(symbol)
    if bar is None:
        return state, None, SimulatedSkip(symbol, "sell", "SUSPENDED_NO_BAR")
    previous_close = day.previous_closes.get(symbol)
    if previous_close is None or previous_close <= 0:
        raise SimulatedExecutionError(f"previous close is required for {symbol}")
    limit_down = is_limit_down(
        price=bar.open,
        previous_close=previous_close,
        limit_pct=segment.price_limit_pct,
    )
    if limit_down:
        return state, None, SimulatedSkip(symbol, "sell", "LIMIT_DOWN_NO_FILL")

    slippage = Decimal(1) - Decimal(day.slippage_bps) / Decimal(10_000)
    execution_price = float(Decimal(str(bar.open)) * slippage)
    gross = Decimal(sellable) * Decimal(str(execution_price))
    market = symbol.rsplit(".", 1)[-1]
    cost = calculate_cost(
        side="sell",
        trade_date=day.trade_date,
        market=market,
        gross_amount=gross,
        model=cost_model,
    )
    remaining = holding.shares - sellable
    holdings = dict(state.holdings)
    if remaining == 0:
        holdings.pop(symbol)
    else:
        holdings[symbol] = Holding(
            shares=remaining,
            locked_shares=min(holding.locked_shares, remaining),
            avg_cost=holding.avg_cost,
        )
    new_state = SimulatedPortfolioState(cash=state.cash + gross - cost.total, holdings=holdings)
    trade = SimulatedTrade(
        trade_date=day.trade_date,
        symbol=symbol,
        side="sell",
        shares=sellable,
        price=execution_price,
        gross_amount=gross,
        cost=cost,
        reason=reason,
    )
    return new_state, trade, None


def mark_to_market(
    state: SimulatedPortfolioState,
    *,
    prices: Mapping[str, float],
) -> Decimal:
    """Value holdings at explicit prices; missing prices fail closed."""
    values: list[float] = []
    for symbol, holding in sorted(state.holdings.items()):
        price = prices.get(symbol)
        if price is None or price <= 0:
            raise SimulatedExecutionError(f"missing mark price for {symbol}")
        values.append(holding.shares * price)
    return state.cash + Decimal(str(fsum(values)))
