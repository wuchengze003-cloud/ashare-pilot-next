from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from ashare_quant_core import (
    DAILY_BAR_SCHEMA_ID,
    DAILY_BAR_SCHEMA_SHA256,
    DailyBar,
    DatasetSnapshot,
)
from ashare_signal_runner import canonical_json_sha256
from ashare_sim_account import advance_account, build_market_day

ROOT = Path(__file__).resolve().parents[3]


def _contract(name: str) -> dict:
    return json.loads(
        (ROOT / "contracts" / "examples" / f"{name}.example.json").read_text(
            encoding="utf-8"
        )
    )


def _signal(*, sequence: int, as_of: str, weight: float) -> dict:
    return {
        "contract_id": "production-signal",
        "schema_version": "4.0.0",
        "signal_id": f"signal-{sequence}",
        "sequence": sequence,
        "state": "ACTIVE" if weight else "FLAT",
        "as_of": as_of,
        "target_positions": (
            [{"symbol": "000001.SZ", "target_weight": weight}] if weight else []
        ),
    }


def _market_day(*, execution_date: str, open_price: float | None, close: float = 10.0) -> dict:
    bars = (
        [{"symbol": "000001.SZ", "open": open_price, "close": close}]
        if open_price is not None
        else []
    )
    return {
        "contract_id": "simulated-market-day",
        "schema_version": "1.0.0",
        "execution_date": execution_date,
        "previous_trade_date": "2026-08-17" if execution_date == "2026-08-18" else "2026-08-18",
        "dataset_id": f"dataset-{execution_date}",
        "dataset_snapshot_sha256": "e" * 64,
        "session_complete": True,
        "bars": bars,
        "previous_closes": [{"symbol": "000001.SZ", "close": 10.0}],
    }


def _advance(*, signal: dict, market_day: dict, previous_state: dict | None = None):
    return advance_account(
        account_id="paper-main",
        production_signal=signal,
        market_day=market_day,
        cost_model=_contract("cost-model"),
        market_rules=_contract("market-rules"),
        execution_policy=_contract("execution-policy"),
        execution_date=date.fromisoformat(market_day["execution_date"]),
        generated_at=datetime(2026, 8, 18, 9, 31, tzinfo=UTC),
        initial_cash=Decimal("100000"),
        previous_state=previous_state,
    )


def test_market_day_is_built_from_one_exact_immutable_snapshot() -> None:
    snapshot = DatasetSnapshot(
        dataset_id="dataset-sim-2026-08-18",
        dataset_family_id="dataset-family-sim",
        manifest_sha256="a" * 64,
        as_of=date(2026, 8, 18),
        data_schema_id=DAILY_BAR_SCHEMA_ID,
        data_schema_sha256=DAILY_BAR_SCHEMA_SHA256,
        normalization_version="v1",
        records=tuple(
            DailyBar(
                symbol="000001.SZ",
                trade_date=trade_date,
                open=open_price,
                high=max(open_price, close),
                low=min(open_price, close),
                close=close,
                volume=1000000,
                amount=10000000,
            )
            for trade_date, open_price, close in (
                (date(2026, 8, 17), 10.0, 10.0),
                (date(2026, 8, 18), 10.1, 10.2),
            )
        ),
    )

    document = build_market_day(
        snapshot,
        previous_trade_date=date(2026, 8, 17),
        execution_date=date(2026, 8, 18),
    )

    assert document["dataset_snapshot_sha256"] == snapshot.snapshot_sha256
    assert document["bars"] == [{"symbol": "000001.SZ", "open": 10.1, "close": 10.2}]


def test_account_buys_then_sells_forward_without_rewriting_history() -> None:
    first = _advance(
        signal=_signal(sequence=1, as_of="2026-08-17", weight=0.5),
        market_day=_market_day(execution_date="2026-08-18", open_price=10.0, close=10.2),
    )
    assert first.document["account_sequence"] == 1
    assert first.document["holdings"][0]["locked_shares"] > 0
    assert first.document["trades"][0]["side"] == "buy"

    second = _advance(
        signal=_signal(sequence=2, as_of="2026-08-18", weight=0.0),
        market_day=_market_day(execution_date="2026-08-19", open_price=10.1),
        previous_state=dict(first.document),
    )
    assert second.document["account_sequence"] == 2
    assert second.document["previous_state_sha256"] == canonical_json_sha256(first.document)
    assert second.document["holdings"] == []
    assert second.document["trades"][0]["side"] == "sell"
    assert first.document["holdings"]


def test_limit_up_buy_is_recorded_as_a_skip() -> None:
    result = _advance(
        signal=_signal(sequence=1, as_of="2026-08-17", weight=0.5),
        market_day=_market_day(execution_date="2026-08-18", open_price=11.0),
    )

    assert result.document["holdings"] == []
    assert result.document["trades"] == []
    assert result.document["skips"] == [
        {"symbol": "000001.SZ", "side": "buy", "reason_code": "LIMIT_UP_NO_FILL"}
    ]


def test_suspended_holding_uses_an_explicit_frozen_valuation() -> None:
    first = _advance(
        signal=_signal(sequence=1, as_of="2026-08-17", weight=0.5),
        market_day=_market_day(execution_date="2026-08-18", open_price=10.0),
    )
    second = _advance(
        signal=_signal(sequence=2, as_of="2026-08-18", weight=0.5),
        market_day=_market_day(execution_date="2026-08-19", open_price=None),
        previous_state=dict(first.document),
    )

    assert second.document["holdings"][0]["valuation_frozen"] is True
    assert second.document["holdings"][0]["market_value"] > 0
    assert second.document["total_assets"] > second.document["cash"]


def test_signal_replay_or_gap_is_rejected() -> None:
    first = _advance(
        signal=_signal(sequence=1, as_of="2026-08-17", weight=0.5),
        market_day=_market_day(execution_date="2026-08-18", open_price=10.0),
    )
    for sequence in (1, 3):
        with pytest.raises(ValueError, match="advance by exactly one"):
            _advance(
                signal=_signal(sequence=sequence, as_of="2026-08-18", weight=0.5),
                market_day=_market_day(execution_date="2026-08-19", open_price=10.0),
                previous_state=dict(first.document),
            )
