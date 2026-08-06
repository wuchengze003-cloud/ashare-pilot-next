"""Focused tests for simulated-portfolio execution semantics."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from ashare_quant_core import (
    DailyBarView,
    ExecutionDay,
    SimulatedExecutionError,
    SimulatedPortfolioState,
    classify_board,
    execute_buy,
    execute_sell,
    is_limit_down,
    is_limit_up,
    mark_to_market,
    parse_market_rules,
    settle_t_plus_one,
)

ROOT = Path(__file__).resolve().parents[3]
MARKET_RULES = {
    "contract_id": "market-rules",
    "schema_version": "1.0.0",
    "rules_id": "test-a-share-market/v1",
    "segments": [
        {
            "segment_id": "main-normal",
            "board": "main",
            "security_status": "normal",
            "valid_from": "2020-01-01",
            "valid_to": None,
            "lot_size": 100,
            "sell_t_plus": 1,
            "price_limit_pct": 0.1,
        },
        {
            "segment_id": "gem-normal",
            "board": "gem",
            "security_status": "normal",
            "valid_from": "2020-01-01",
            "valid_to": None,
            "lot_size": 100,
            "sell_t_plus": 1,
            "price_limit_pct": 0.2,
        },
        {
            "segment_id": "star-normal",
            "board": "star",
            "security_status": "normal",
            "valid_from": "2020-01-01",
            "valid_to": None,
            "lot_size": 100,
            "sell_t_plus": 1,
            "price_limit_pct": 0.2,
        },
    ],
}


def cost_model() -> dict[str, object]:
    return json.loads(
        (ROOT / "contracts/examples/cost-model.example.json").read_text(encoding="utf-8")
    )


def rules() -> dict[str, object]:
    return parse_market_rules(MARKET_RULES)


def day(
    *,
    trade_date: date = date(2023, 9, 1),
    open_price: float = 10.0,
    close_price: float = 10.2,
    previous_close: float = 9.9,
    slippage_bps: int = 10,
    has_bar: bool = True,
    symbol: str = "000001.SZ",
) -> ExecutionDay:
    bar = DailyBarView(open=open_price, close=close_price) if has_bar else None
    return ExecutionDay(
        trade_date=trade_date,
        bars={symbol: bar},
        previous_closes={symbol: previous_close},
        slippage_bps=slippage_bps,
    )


def start_state(cash: Decimal = Decimal("500000")) -> SimulatedPortfolioState:
    return SimulatedPortfolioState(cash=cash, holdings={})


def test_classify_board_covers_all_supported_segments() -> None:
    assert classify_board("000001.SZ") == "main"
    assert classify_board("600519.SH") == "main"
    assert classify_board("300750.SZ") == "gem"
    assert classify_board("688981.SH") == "star"


def test_classify_board_rejects_beijing_exchange() -> None:
    with pytest.raises(SimulatedExecutionError, match="Beijing"):
        classify_board("830799.BJ")


def test_buy_applies_slippage_costs_and_t1_lock() -> None:
    state = start_state()
    market = rules()

    state, trade, skip, buy_date = execute_buy(
        state=state,
        day=day(),
        symbol="000001.SZ",
        requested_shares=1000,
        rules=market,
        cost_model=cost_model(),
        reason="MODEL_TOP_SCORE",
    )

    assert skip is None and trade is not None
    assert buy_date == date(2023, 9, 1)
    assert trade.shares == 1000
    assert trade.price == pytest.approx(10.0 * 1.001)
    gross = Decimal("1000") * Decimal(str(trade.price))
    assert trade.gross_amount == gross
    assert trade.cost.commission >= Decimal("5")
    assert trade.cost.stamp_duty == Decimal("0.00")
    holding = state.holdings["000001.SZ"]
    assert holding.shares == 1000
    assert holding.locked_shares == 1000
    assert state.cash == Decimal("500000") - gross - trade.cost.total


def test_buy_rounds_requested_shares_to_lot_size() -> None:
    state, trade, skip, _ = execute_buy(
        state=start_state(),
        day=day(),
        symbol="000001.SZ",
        requested_shares=150,
        rules=rules(),
        cost_model=cost_model(),
        reason="TEST",
    )

    assert skip is None and trade is not None
    assert trade.shares == 100


def test_buy_fails_closed_on_insufficient_cash() -> None:
    state, trade, skip, _ = execute_buy(
        state=start_state(cash=Decimal("100")),
        day=day(),
        symbol="000001.SZ",
        requested_shares=1000,
        rules=rules(),
        cost_model=cost_model(),
        reason="TEST",
    )

    assert trade is None
    assert skip is not None and skip.reason_code == "INSUFFICIENT_CASH"
    assert state.cash == Decimal("100")


def test_same_day_sell_is_blocked_by_t1() -> None:
    state, _, _, buy_dates_lock = execute_buy(
        state=start_state(),
        day=day(),
        symbol="000001.SZ",
        requested_shares=1000,
        rules=rules(),
        cost_model=cost_model(),
        reason="TEST",
    )
    assert buy_dates_lock is not None

    state, trade, skip = execute_sell(
        state=state,
        day=day(),
        symbol="000001.SZ",
        rules=rules(),
        cost_model=cost_model(),
        reason="MODEL_EXIT",
    )

    assert trade is None
    assert skip is not None and skip.reason_code == "T1_LOCKED"


def test_next_day_settlement_unlocks_and_sell_pays_stamp_duty() -> None:
    execution_day = day()
    state, _, _, _ = execute_buy(
        state=start_state(),
        day=execution_day,
        symbol="000001.SZ",
        requested_shares=1000,
        rules=rules(),
        cost_model=cost_model(),
        reason="TEST",
    )
    next_day = day(trade_date=date(2023, 9, 4), open_price=10.5, previous_close=10.2)
    state = settle_t_plus_one(
        state,
        trade_date=next_day.trade_date,
        buy_dates={"000001.SZ": execution_day.trade_date},
    )
    assert state.holdings["000001.SZ"].locked_shares == 0

    cash_before = state.cash
    state, trade, skip = execute_sell(
        state=state,
        day=next_day,
        symbol="000001.SZ",
        rules=rules(),
        cost_model=cost_model(),
        reason="MODEL_EXIT",
    )

    assert skip is None and trade is not None
    assert trade.shares == 1000
    assert trade.cost.stamp_duty > Decimal("0")
    assert "000001.SZ" not in state.holdings
    assert state.cash == cash_before + trade.gross_amount - trade.cost.total


def test_limit_up_open_blocks_buy() -> None:
    state, trade, skip, _ = execute_buy(
        state=start_state(),
        day=day(open_price=10.89, previous_close=9.9),
        symbol="000001.SZ",
        requested_shares=1000,
        rules=rules(),
        cost_model=cost_model(),
        reason="TEST",
    )

    assert trade is None
    assert skip is not None and skip.reason_code == "LIMIT_UP_NO_FILL"
    assert is_limit_up(price=10.89, previous_close=9.9, limit_pct=0.1)


def test_limit_down_open_blocks_sell() -> None:
    execution_day = day()
    state, _, _, _ = execute_buy(
        state=start_state(),
        day=execution_day,
        symbol="000001.SZ",
        requested_shares=1000,
        rules=rules(),
        cost_model=cost_model(),
        reason="TEST",
    )
    state = settle_t_plus_one(
        state,
        trade_date=date(2023, 9, 4),
        buy_dates={"000001.SZ": execution_day.trade_date},
    )
    crash_day = day(trade_date=date(2023, 9, 4), open_price=8.91, previous_close=9.9)

    state, trade, skip = execute_sell(
        state=state,
        day=crash_day,
        symbol="000001.SZ",
        rules=rules(),
        cost_model=cost_model(),
        reason="MODEL_EXIT",
    )

    assert trade is None
    assert skip is not None and skip.reason_code == "LIMIT_DOWN_NO_FILL"
    assert is_limit_down(price=8.91, previous_close=9.9, limit_pct=0.1)


def test_gem_and_star_allow_twenty_percent_moves() -> None:
    assert not is_limit_up(price=11.9, previous_close=10.0, limit_pct=0.2)
    assert is_limit_up(price=12.0, previous_close=10.0, limit_pct=0.2)
    assert not is_limit_down(price=8.1, previous_close=10.0, limit_pct=0.2)
    assert is_limit_down(price=8.0, previous_close=10.0, limit_pct=0.2)


def test_suspended_symbol_skips_both_sides() -> None:
    bought, _, _, _ = execute_buy(
        state=start_state(),
        day=day(),
        symbol="000001.SZ",
        requested_shares=1000,
        rules=rules(),
        cost_model=cost_model(),
        reason="TEST",
    )
    suspended_day = day(trade_date=date(2023, 9, 4), has_bar=False)

    _, trade, skip, _ = execute_buy(
        state=start_state(),
        day=suspended_day,
        symbol="000001.SZ",
        requested_shares=1000,
        rules=rules(),
        cost_model=cost_model(),
        reason="TEST",
    )
    assert trade is None and skip is not None
    assert skip.reason_code == "SUSPENDED_NO_BAR"

    unlocked = settle_t_plus_one(
        bought,
        trade_date=date(2023, 9, 4),
        buy_dates={"000001.SZ": date(2023, 9, 1)},
    )
    _, sell_trade, sell_skip = execute_sell(
        state=unlocked,
        day=suspended_day,
        symbol="000001.SZ",
        rules=rules(),
        cost_model=cost_model(),
        reason="TEST",
    )
    assert sell_trade is None and sell_skip is not None
    assert sell_skip.reason_code == "SUSPENDED_NO_BAR"


def test_mark_to_market_requires_every_price() -> None:
    state, _, _, _ = execute_buy(
        state=start_state(),
        day=day(),
        symbol="000001.SZ",
        requested_shares=1000,
        rules=rules(),
        cost_model=cost_model(),
        reason="TEST",
    )

    total = mark_to_market(state, prices={"000001.SZ": 10.2})
    assert total == state.cash + Decimal(str(1000 * 10.2))

    with pytest.raises(SimulatedExecutionError, match="missing mark price"):
        mark_to_market(state, prices={})


def test_simulation_is_deterministic_for_identical_inputs() -> None:
    def run_once() -> tuple[Decimal, Decimal]:
        state = start_state()
        state, trade, _, _ = execute_buy(
            state=state,
            day=day(),
            symbol="000001.SZ",
            requested_shares=1000,
            rules=rules(),
            cost_model=cost_model(),
            reason="TEST",
        )
        assert trade is not None
        return state.cash, trade.cost.total

    assert run_once() == run_once()
