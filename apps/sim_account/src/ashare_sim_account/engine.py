"""Advance one simulated account from one frozen Production Signal."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from ashare_quant_core import (
    DailyBarView,
    DatasetSnapshot,
    ExecutionDay,
    Holding,
    RuntimeState,
    SimulatedPortfolioState,
    classify_board,
    constrain_execution_targets,
    execute_buy,
    execute_sell,
    mark_to_market,
    parse_market_rules,
    settle_t_plus_one,
)
from ashare_signal_runner import canonical_json_sha256

ACCOUNT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")


@dataclass(frozen=True)
class AccountAdvance:
    """One deterministic account transition and its serialized state."""

    document: Mapping[str, Any]
    portfolio: SimulatedPortfolioState
    buy_dates: Mapping[str, date]


def build_market_day(
    snapshot: DatasetSnapshot,
    *,
    previous_trade_date: date,
    execution_date: date,
) -> dict[str, Any]:
    """Build one complete simulated market day from an immutable snapshot."""
    if snapshot.as_of != execution_date:
        raise ValueError("market-day snapshot as_of must equal execution_date")
    prior_market_dates = sorted(
        {bar.trade_date for bar in snapshot.records if bar.trade_date < execution_date}
    )
    if not prior_market_dates or prior_market_dates[-1] != previous_trade_date:
        raise ValueError("previous_trade_date must be the latest market date before execution_date")
    bars = [
        {
            "symbol": bar.symbol,
            "open": bar.open,
            "close": bar.close,
        }
        for bar in snapshot.records
        if bar.trade_date == execution_date
    ]
    latest_by_symbol = {}
    for bar in snapshot.records:
        if bar.trade_date >= execution_date:
            continue
        current = latest_by_symbol.get(bar.symbol)
        if current is None or bar.trade_date > current.trade_date:
            latest_by_symbol[bar.symbol] = bar
    previous_closes = [
        {
            "symbol": bar.symbol,
            "trade_date": bar.trade_date.isoformat(),
            "close": bar.close,
        }
        for bar in latest_by_symbol.values()
    ]
    if not bars:
        raise ValueError("immutable snapshot contains no execution-date bars")
    if not previous_closes:
        raise ValueError("immutable snapshot contains no previous-trade-date bars")
    return {
        "contract_id": "simulated-market-day",
        "schema_version": "2.0.0",
        "execution_date": execution_date.isoformat(),
        "previous_trade_date": previous_trade_date.isoformat(),
        "dataset_id": snapshot.dataset_id,
        "dataset_snapshot_sha256": snapshot.snapshot_sha256,
        "session_complete": True,
        "bars": sorted(bars, key=lambda item: str(item["symbol"])),
        "previous_closes": sorted(previous_closes, key=lambda item: str(item["symbol"])),
    }


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _restore_portfolio(
    previous_state: Mapping[str, Any] | None,
    *,
    initial_cash: Decimal,
) -> tuple[SimulatedPortfolioState, dict[str, date], dict[str, float], Decimal, int]:
    if previous_state is None:
        return SimulatedPortfolioState(cash=initial_cash, holdings={}), {}, {}, initial_cash, 1
    if previous_state.get("contract_id") != "simulated-account-state":
        raise ValueError("previous state is not a simulated-account-state")
    holdings: dict[str, Holding] = {}
    buy_dates: dict[str, date] = {}
    last_prices: dict[str, float] = {}
    for item in previous_state["holdings"]:
        symbol = str(item["symbol"])
        holdings[symbol] = Holding(
            shares=int(item["shares"]),
            locked_shares=int(item["locked_shares"]),
            avg_cost=_decimal(item["avg_cost"]),
        )
        last_price = float(item["last_price"])
        if last_price <= 0:
            raise ValueError(f"previous state has invalid last price for {symbol}")
        last_prices[symbol] = last_price
        if item.get("last_buy_date") is not None:
            buy_dates[symbol] = date.fromisoformat(str(item["last_buy_date"]))
    recorded_initial_cash = _decimal(previous_state["initial_cash"])
    if recorded_initial_cash != initial_cash:
        raise ValueError("initial cash does not match previous account state")
    return (
        SimulatedPortfolioState(
            cash=_decimal(previous_state["cash"]),
            holdings=holdings,
        ),
        buy_dates,
        last_prices,
        recorded_initial_cash,
        int(previous_state["account_sequence"]) + 1,
    )


def _parse_market_day(
    market_day: Mapping[str, Any],
    *,
    execution_date: date,
) -> tuple[dict[str, DailyBarView], dict[str, float]]:
    if market_day.get("contract_id") != "simulated-market-day":
        raise ValueError("market input is not a simulated-market-day")
    if date.fromisoformat(str(market_day["execution_date"])) != execution_date:
        raise ValueError("market input execution_date mismatch")
    if market_day.get("session_complete") is not True:
        raise ValueError("simulated market day must be complete")
    bars: dict[str, DailyBarView] = {}
    for item in market_day["bars"]:
        symbol = str(item["symbol"])
        if symbol in bars:
            raise ValueError(f"duplicate market bar: {symbol}")
        bars[symbol] = DailyBarView(open=float(item["open"]), close=float(item["close"]))
    previous_closes: dict[str, float] = {}
    for item in market_day["previous_closes"]:
        symbol = str(item["symbol"])
        if symbol in previous_closes:
            raise ValueError(f"duplicate previous close: {symbol}")
        close_date = date.fromisoformat(str(item["trade_date"]))
        if close_date >= execution_date:
            raise ValueError(f"previous close date must precede execution for {symbol}")
        previous_closes[symbol] = float(item["close"])
    return bars, previous_closes


def _require_signal_contract_binding(
    production_signal: Mapping[str, Any],
    *,
    cost_model: Mapping[str, Any],
    market_rules: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
) -> None:
    contract_set = production_signal.get("contract_set")
    if not isinstance(contract_set, Mapping):
        raise ValueError("production signal is missing its contract set")
    bindings = {
        "cost_model_sha256": cost_model,
        "market_rules_sha256": market_rules,
        "execution_policy_sha256": execution_policy,
    }
    for field, document in bindings.items():
        if contract_set.get(field) != canonical_json_sha256(document):
            raise ValueError(f"production signal {field} does not match execution contract")


def _trade_document(trade) -> dict[str, object]:
    return {
        "trade_date": trade.trade_date.isoformat(),
        "symbol": trade.symbol,
        "side": trade.side,
        "shares": trade.shares,
        "price": trade.price,
        "gross_amount": float(trade.gross_amount),
        "commission": float(trade.cost.commission),
        "stamp_duty": float(trade.cost.stamp_duty),
        "transfer_fee": float(trade.cost.transfer_fee),
        "total_cost": float(trade.cost.total),
        "reason": trade.reason,
    }


def advance_account(
    *,
    account_id: str,
    production_signal: Mapping[str, Any],
    market_day: Mapping[str, Any],
    cost_model: Mapping[str, Any],
    market_rules: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    execution_date: date,
    generated_at: datetime,
    initial_cash: Decimal,
    previous_state: Mapping[str, Any] | None = None,
) -> AccountAdvance:
    """Apply one signal exactly once and return the next immutable account state."""
    if not ACCOUNT_ID.fullmatch(account_id):
        raise ValueError("invalid simulated account id")
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must include a timezone")
    if production_signal.get("contract_id") != "production-signal":
        raise ValueError("input is not a production-signal")
    _require_signal_contract_binding(
        production_signal,
        cost_model=cost_model,
        market_rules=market_rules,
        execution_policy=execution_policy,
    )
    try:
        signal_state = RuntimeState(str(production_signal["state"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("unsupported production signal state") from exc
    signal_as_of = date.fromisoformat(str(production_signal["as_of"]))
    if execution_date <= signal_as_of:
        raise ValueError("execution date must be after the signal as_of")
    if str(execution_policy.get("decision_timing")) != "daily_close":
        raise ValueError("simulated account requires daily_close decisions")
    if str(execution_policy.get("execution_timing")) != "next_open":
        raise ValueError("simulated account requires next_open execution")
    if date.fromisoformat(str(market_day["previous_trade_date"])) != signal_as_of:
        raise ValueError("market previous_trade_date must match the signal as_of")

    portfolio, buy_dates, previous_last_prices, initial_cash, account_sequence = _restore_portfolio(
        previous_state,
        initial_cash=initial_cash,
    )
    signal_sequence = int(production_signal["sequence"])
    if previous_state is not None:
        previous_signal_sequence = int(previous_state["source_signal"]["sequence"])
        if signal_sequence != previous_signal_sequence + 1:
            raise ValueError("production signal sequence must advance by exactly one")
        previous_as_of = date.fromisoformat(str(previous_state["as_of"]))
        if execution_date <= previous_as_of:
            raise ValueError("account execution date must advance monotonically")

    bars, previous_closes = _parse_market_day(market_day, execution_date=execution_date)
    portfolio = settle_t_plus_one(
        portfolio,
        trade_date=execution_date,
        buy_dates=buy_dates,
    )
    rules = parse_market_rules(market_rules)
    day = ExecutionDay(
        trade_date=execution_date,
        bars=bars,
        previous_closes=previous_closes,
        slippage_bps=int(execution_policy["slippage_bps"]),
    )

    held_reference_prices = {
        symbol: previous_closes.get(symbol, previous_last_prices.get(symbol))
        for symbol in portfolio.holdings
    }
    held_reference_prices = {
        symbol: price for symbol, price in held_reference_prices.items() if price is not None
    }
    if set(held_reference_prices) != set(portfolio.holdings):
        missing = sorted(set(portfolio.holdings) - set(held_reference_prices))
        raise ValueError("missing previous close for holdings: " + ", ".join(missing))
    total_assets_before = (
        mark_to_market(portfolio, prices=held_reference_prices)
        if portfolio.holdings
        else portfolio.cash
    )

    targets: dict[str, float] = {}
    for item in production_signal["target_positions"]:
        symbol = str(item["symbol"])
        if symbol in targets:
            raise ValueError(f"duplicate target position: {symbol}")
        targets[symbol] = float(item["target_weight"])
    if sum(targets.values()) > 1.0 + 1e-9:
        raise ValueError("target weights exceed total account exposure")
    raw_desired_shares: dict[str, int] = {}
    skips: list[dict[str, str]] = []
    for symbol, weight in sorted(targets.items()):
        reference = previous_closes.get(symbol)
        if reference is None or reference <= 0:
            skips.append(
                {"symbol": symbol, "side": "buy", "reason_code": "MISSING_REFERENCE_PRICE"}
            )
            continue
        lot = rules[classify_board(symbol)].lot_size
        raw_desired_shares[symbol] = (
            int(float(total_assets_before) * weight / reference / lot) * lot
        )

    current_shares = {symbol: holding.shares for symbol, holding in portfolio.holdings.items()}
    desired_shares = constrain_execution_targets(
        state=signal_state,
        desired_shares=raw_desired_shares,
        current_shares=current_shares,
    )
    for symbol in sorted(set(raw_desired_shares) | set(current_shares)):
        raw = raw_desired_shares.get(symbol, 0)
        constrained = desired_shares.get(symbol, 0)
        if constrained == raw:
            continue
        side = "buy" if raw > constrained else "sell"
        reason = "HOLD_NO_TRADE" if signal_state is RuntimeState.HOLD else "REDUCE_ONLY_NO_BUY"
        skips.append({"symbol": symbol, "side": side, "reason_code": reason})

    trades: list[dict[str, object]] = []
    cash_before_trades = portfolio.cash
    cash_delta = Decimal("0")
    for symbol in sorted(portfolio.holdings):
        held = portfolio.holdings[symbol].shares
        desired = desired_shares.get(symbol, 0)
        if held <= desired:
            continue
        portfolio, trade, skip = execute_sell(
            state=portfolio,
            day=day,
            symbol=symbol,
            rules=rules,
            cost_model=cost_model,
            reason="TARGET_REBALANCE",
            max_shares=held - desired,
        )
        if trade is not None:
            trades.append(_trade_document(trade))
            cash_delta += trade.gross_amount - trade.cost.total
            if symbol not in portfolio.holdings:
                buy_dates.pop(symbol, None)
        elif skip is not None:
            skips.append(
                {"symbol": skip.symbol, "side": skip.side, "reason_code": skip.reason_code}
            )

    for symbol, desired in sorted(desired_shares.items()):
        held = portfolio.holdings.get(symbol)
        current = held.shares if held is not None else 0
        if desired <= current:
            continue
        portfolio, trade, skip, bought_on = execute_buy(
            state=portfolio,
            day=day,
            symbol=symbol,
            requested_shares=desired - current,
            rules=rules,
            cost_model=cost_model,
            reason="TARGET_REBALANCE",
        )
        if trade is not None and bought_on is not None:
            trades.append(_trade_document(trade))
            cash_delta -= trade.gross_amount + trade.cost.total
            buy_dates[symbol] = bought_on
        elif skip is not None:
            skips.append(
                {"symbol": skip.symbol, "side": skip.side, "reason_code": skip.reason_code}
            )

    if portfolio.cash != cash_before_trades + cash_delta:
        raise ValueError("simulated cash ledger does not reconcile")

    mark_prices: dict[str, float] = {}
    frozen_symbols: set[str] = set()
    for symbol in portfolio.holdings:
        bar = bars.get(symbol)
        if bar is not None:
            mark_prices[symbol] = bar.close
        elif symbol in previous_closes:
            mark_prices[symbol] = previous_closes[symbol]
            frozen_symbols.add(symbol)
        elif symbol in previous_last_prices:
            mark_prices[symbol] = previous_last_prices[symbol]
            frozen_symbols.add(symbol)
        else:
            raise ValueError(f"missing mark price for {symbol}")
    total_assets = (
        mark_to_market(portfolio, prices=mark_prices) if portfolio.holdings else portfolio.cash
    )
    market_value = total_assets - portfolio.cash
    expected_market_value = sum(
        (
            Decimal(holding.shares) * Decimal(str(mark_prices[symbol]))
            for symbol, holding in portfolio.holdings.items()
        ),
        Decimal("0"),
    )
    if market_value != expected_market_value:
        raise ValueError("simulated holdings do not reconcile to market value")
    if total_assets != portfolio.cash + market_value:
        raise ValueError("simulated account assets do not reconcile")
    signal_sha256 = canonical_json_sha256(production_signal)
    previous_state_sha256 = (
        canonical_json_sha256(previous_state) if previous_state is not None else None
    )
    holdings_document = []
    for symbol, holding in sorted(portfolio.holdings.items()):
        last_price = mark_prices[symbol]
        holdings_document.append(
            {
                "symbol": symbol,
                "shares": holding.shares,
                "locked_shares": holding.locked_shares,
                "avg_cost": float(holding.avg_cost),
                "last_price": last_price,
                "market_value": holding.shares * last_price,
                "target_weight": targets.get(symbol, 0.0),
                "last_buy_date": (buy_dates[symbol].isoformat() if symbol in buy_dates else None),
                "valuation_frozen": symbol in frozen_symbols,
            }
        )

    document = {
        "contract_id": "simulated-account-state",
        "schema_version": "2.0.0",
        "state_id": f"{account_id}-{execution_date.isoformat()}-seq{account_sequence}",
        "account_id": account_id,
        "account_sequence": account_sequence,
        "as_of": execution_date.isoformat(),
        "generated_at": generated_at.isoformat(),
        "previous_state_sha256": previous_state_sha256,
        "source_signal": {
            "signal_id": str(production_signal["signal_id"]),
            "sequence": signal_sequence,
            "state": signal_state.value,
            "as_of": signal_as_of.isoformat(),
            "sha256": signal_sha256,
        },
        "initial_cash": float(initial_cash),
        "cash": float(portfolio.cash),
        "market_value": float(market_value),
        "total_assets": float(total_assets),
        "nav": float(total_assets / initial_cash),
        "target_positions": [
            {"symbol": symbol, "target_weight": weight}
            for symbol, weight in sorted(targets.items())
        ],
        "holdings": holdings_document,
        "trades": trades,
        "skips": skips,
        "contract_set": {
            "market_day_sha256": canonical_json_sha256(market_day),
            "dataset_snapshot_sha256": str(market_day["dataset_snapshot_sha256"]),
            "cost_model_sha256": canonical_json_sha256(cost_model),
            "market_rules_sha256": canonical_json_sha256(market_rules),
            "execution_policy_sha256": canonical_json_sha256(execution_policy),
        },
    }
    return AccountAdvance(document=document, portfolio=portfolio, buy_dates=buy_dates)
