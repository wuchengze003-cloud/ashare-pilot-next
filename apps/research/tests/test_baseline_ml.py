"""Synthetic-data tests for PIT features, labels, and the walk-forward loop."""

import json
import random
from datetime import date, timedelta
from pathlib import Path

import pytest
from ashare_quant_core import (
    DAILY_BAR_SCHEMA_ID,
    DAILY_BAR_SCHEMA_SHA256,
    DailyBar,
    DatasetSnapshot,
)
from ashare_research_app.backtest import PilotConfig, run_walk_forward
from ashare_research_app.baseline_model import MultiHorizonModel
from ashare_research_app.features import (
    FEATURE_NAMES,
    build_feature_panel,
    compute_feature_row,
    forward_return_label,
)

ROOT = Path(__file__).resolve().parents[3]
SYMBOLS = (
    "000001.SZ",
    "000333.SZ",
    "002594.SZ",
    "300750.SZ",
    "600036.SH",
    "600519.SH",
    "601318.SH",
    "688981.SH",
)


def trading_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def synthetic_snapshot(
    *,
    days: int = 80,
    symbols: tuple[str, ...] = SYMBOLS,
    seed: int = 7,
    as_of: date | None = None,
    extra_tail_days: int = 0,
) -> DatasetSnapshot:
    calendar = trading_days(date(2023, 6, 1), days + extra_tail_days)
    records: list[DailyBar] = []
    for symbol in symbols:
        generator = random.Random(f"{seed}:{symbol}")
        price = 10.0 + generator.random() * 40
        for day in calendar:
            drift = generator.uniform(-0.03, 0.032)
            open_price = price
            close_price = max(1.0, price * (1 + drift))
            high = max(open_price, close_price) * (1 + generator.uniform(0, 0.01))
            low = min(open_price, close_price) * (1 - generator.uniform(0, 0.01))
            volume = generator.uniform(5e6, 5e7)
            amount = volume * (open_price + close_price) / 2
            records.append(
                DailyBar(
                    symbol=symbol,
                    trade_date=day,
                    open=round(open_price, 4),
                    high=round(high, 4),
                    low=round(low, 4),
                    close=round(close_price, 4),
                    volume=round(volume, 0),
                    amount=round(amount, 0),
                )
            )
            price = close_price
    cutoff = as_of or calendar[days - 1]
    return DatasetSnapshot(
        dataset_id="synthetic-pilot/v1",
        dataset_family_id="synthetic-pilot/v1",
        manifest_sha256="a" * 64,
        as_of=cutoff,
        data_schema_id=DAILY_BAR_SCHEMA_ID,
        data_schema_sha256=DAILY_BAR_SCHEMA_SHA256,
        normalization_version="synthetic-normalizer/v1",
        records=tuple(
            record for record in records if record.trade_date <= cutoff
        ),
    )


def contract_documents() -> dict[str, dict[str, object]]:
    cost_model = json.loads(
        (ROOT / "contracts/examples/cost-model.example.json").read_text(encoding="utf-8")
    )
    market_rules = {
        "contract_id": "market-rules",
        "schema_version": "1.0.0",
        "rules_id": "test-a-share-market/v1",
        "segments": [
            {
                "segment_id": f"{board}-normal",
                "board": board,
                "security_status": "normal",
                "valid_from": "2020-01-01",
                "valid_to": None,
                "lot_size": 100,
                "sell_t_plus": 1,
                "price_limit_pct": 0.2 if board in {"gem", "star"} else 0.1,
            }
            for board in ("main", "gem", "star")
        ],
    }
    execution_policy = json.loads(
        (ROOT / "contracts/examples/execution-policy.example.json").read_text(encoding="utf-8")
    )
    portfolio_risk = json.loads(
        (ROOT / "contracts/examples/portfolio-risk.example.json").read_text(encoding="utf-8")
    )
    portfolio_risk["max_positions"] = 5
    portfolio_risk["max_single_weight"] = 0.25
    return {
        "cost-model": cost_model,
        "market-rules": market_rules,
        "execution-policy": execution_policy,
        "portfolio-risk": portfolio_risk,
    }


def test_feature_panel_respects_as_of_cutoff() -> None:
    snapshot = synthetic_snapshot()
    calendar = sorted({bar.trade_date for bar in snapshot.records})
    midpoint = calendar[len(calendar) // 2]

    truncated = build_feature_panel(snapshot, as_of=midpoint)

    assert truncated
    assert all(row.trade_date <= midpoint for row in truncated)
    full_visible = tuple(row for row in build_feature_panel(snapshot) if row.trade_date <= midpoint)
    assert truncated == full_visible


def test_future_rows_do_not_change_historical_features() -> None:
    baseline = synthetic_snapshot(days=70)
    extended = synthetic_snapshot(days=70, extra_tail_days=10)
    panel_baseline = build_feature_panel(baseline)
    panel_extended = build_feature_panel(extended, as_of=baseline.as_of)

    assert panel_extended == panel_baseline


def test_feature_row_width_and_label_horizon_bounds() -> None:
    snapshot = synthetic_snapshot(days=40, symbols=("000001.SZ",))
    bars = sorted(snapshot.records, key=lambda bar: bar.trade_date)
    values = compute_feature_row(bars, len(bars) - 1)
    assert values is not None and len(values) == len(FEATURE_NAMES)
    assert compute_feature_row(bars, 10) is None
    assert forward_return_label(bars, len(bars) - 1, 1) is None
    assert forward_return_label(bars, 0, 5) is not None


def test_walk_forward_is_deterministic_and_leak_checked() -> None:
    snapshot = synthetic_snapshot()
    documents = contract_documents()

    def run_once() -> dict[str, object]:
        model, _production, report = run_walk_forward(
            snapshot,
            cost_model_doc=documents["cost-model"],
            market_rules_doc=documents["market-rules"],
            execution_policy_doc=documents["execution-policy"],
            portfolio_risk_doc=documents["portfolio-risk"],
            config=PilotConfig(top_k=4, per_weight=0.24),
        )
        assert all(check.status == "pass" for check in report.leak_checks)
        assert report.train_end < report.validation_start
        assert report.validation_end < report.test_start
        assert report.metrics["max_drawdown"] >= 0
        assert report.recommendations
        assert report.nav_curve
        rebalance = report.metrics
        assert all(value == value for value in rebalance.values())
        return {
            "bundle": report.model_bundle_sha256,
            "metrics": dict(report.metrics),
            "nav": [dict(point) for point in report.nav_curve],
            "recs": [dict(item) for item in report.recommendations],
        }

    assert run_once() == run_once()


def test_recommendation_labels_cover_buy_hold_sell_watch() -> None:
    snapshot = synthetic_snapshot()
    documents = contract_documents()
    _, _, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )

    labels = {item["recommendation"] for item in report.recommendations}
    assert labels <= {"BUY", "HOLD", "SELL", "WATCH"}
    assert "BUY" in labels or "HOLD" in labels
    ranks = [item["rank"] for item in report.recommendations]
    assert sorted(ranks) == list(range(1, len(ranks) + 1))


def test_model_bundle_roundtrip_preserves_scores() -> None:
    snapshot = synthetic_snapshot()
    documents = contract_documents()
    model, _production, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )

    restored = MultiHorizonModel.from_bundle_bytes(model.bundle_bytes())
    matrix = __import__("numpy").asarray(
        [[0.01] * len(FEATURE_NAMES), [-0.01] * len(FEATURE_NAMES)], dtype=float
    )
    assert restored.score(matrix).tolist() == model.score(matrix).tolist()
    assert report.training_cutoff == model.training_cutoff


def test_walk_forward_rejects_tiny_history() -> None:
    snapshot = synthetic_snapshot(days=30)
    documents = contract_documents()

    with pytest.raises(ValueError, match="at least 40 trade dates"):
        run_walk_forward(
            snapshot,
            cost_model_doc=documents["cost-model"],
            market_rules_doc=documents["market-rules"],
            execution_policy_doc=documents["execution-policy"],
            portfolio_risk_doc=documents["portfolio-risk"],
        )


def test_within_window_mutation_does_not_change_historical_features() -> None:
    snapshot = synthetic_snapshot(days=60)
    calendar = sorted({bar.trade_date for bar in snapshot.records})
    mutation_day = calendar[45]
    target_day = calendar[30]

    mutated_records = tuple(
        DailyBar(
            symbol=bar.symbol,
            trade_date=bar.trade_date,
            open=bar.open * 1.07,
            high=bar.high * 1.07,
            low=bar.low * 1.07,
            close=bar.close * 1.07,
            volume=bar.volume * 2,
            amount=bar.amount * 2,
        )
        if bar.symbol == "000001.SZ" and bar.trade_date == mutation_day
        else bar
        for bar in snapshot.records
    )
    mutated = DatasetSnapshot(
        dataset_id=snapshot.dataset_id,
        dataset_family_id=snapshot.dataset_family_id,
        manifest_sha256=snapshot.manifest_sha256,
        as_of=snapshot.as_of,
        data_schema_id=snapshot.data_schema_id,
        data_schema_sha256=snapshot.data_schema_sha256,
        normalization_version=snapshot.normalization_version,
        records=mutated_records,
    )

    base_rows = tuple(
        row
        for row in build_feature_panel(snapshot)
        if row.symbol == "000001.SZ" and row.trade_date <= target_day
    )
    mutated_rows = tuple(
        row
        for row in build_feature_panel(mutated)
        if row.symbol == "000001.SZ" and row.trade_date <= target_day
    )

    assert base_rows
    assert mutated_rows == base_rows


def test_blocked_sells_cannot_push_portfolio_past_position_limit() -> None:
    snapshot = synthetic_snapshot(days=90)
    calendar = sorted({bar.trade_date for bar in snapshot.records})
    stop_day = calendar[70]
    records = tuple(
        bar
        for bar in snapshot.records
        if not (bar.symbol == "600519.SH" and bar.trade_date > stop_day)
    )
    gapped = DatasetSnapshot(
        dataset_id=snapshot.dataset_id,
        dataset_family_id=snapshot.dataset_family_id,
        manifest_sha256=snapshot.manifest_sha256,
        as_of=snapshot.as_of,
        data_schema_id=snapshot.data_schema_id,
        data_schema_sha256=snapshot.data_schema_sha256,
        normalization_version=snapshot.normalization_version,
        records=records,
    )
    documents = contract_documents()
    documents["portfolio-risk"]["max_positions"] = 4

    _model, _production, report = run_walk_forward(
        gapped,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )

    assert len(report.final_state.holdings) <= 4
    market_value = sum(
        holding.shares * report.final_prices.get(symbol, 0.0)
        for symbol, holding in report.final_state.holdings.items()
    )
    total_assets = float(report.final_state.cash) + market_value
    assert market_value / total_assets <= 1.0 + 1e-6


def test_nav_curve_has_no_duplicate_trade_dates() -> None:
    snapshot = synthetic_snapshot()
    documents = contract_documents()
    _m, _p, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )
    dates_seen = [str(point["trade_date"]) for point in report.nav_curve]
    assert len(dates_seen) == len(set(dates_seen))
    assert report.nav_curve[0]["nav"] == 1.0
    assert dates_seen[0] < dates_seen[1]


def test_bundle_load_rejects_library_version_mismatch() -> None:
    import pickle as _pickle

    snapshot = synthetic_snapshot()
    documents = contract_documents()
    _m, production, _r = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )
    bundle = _pickle.loads(production.bundle_bytes())
    bundle["sklearn_version"] = "0.0.1-tampered"
    with pytest.raises(ValueError, match="sklearn version"):
        MultiHorizonModel.from_bundle_bytes(_pickle.dumps(bundle))


def test_validation_ic_window_ends_before_test_period() -> None:
    snapshot = synthetic_snapshot()
    documents = contract_documents()
    _m, _p, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )
    checks = {check.check_id: check for check in report.leak_checks}
    assert "validation_labels_outside_test" in checks
    assert checks["validation_labels_outside_test"].status == "pass"


def test_suspended_symbol_labels_cannot_cross_test_period() -> None:
    snapshot = synthetic_snapshot(days=90)
    calendar = sorted({bar.trade_date for bar in snapshot.records})
    validation_end_day = calendar[int(len(calendar) * 0.78) - 1]
    gap_start = calendar[int(len(calendar) * 0.62)]
    records = tuple(
        bar
        for bar in snapshot.records
        if not (
            bar.symbol == "000732.SZ"
            and gap_start <= bar.trade_date <= validation_end_day
        )
    )
    gapped = DatasetSnapshot(
        dataset_id=snapshot.dataset_id,
        dataset_family_id=snapshot.dataset_family_id,
        manifest_sha256=snapshot.manifest_sha256,
        as_of=snapshot.as_of,
        data_schema_id=snapshot.data_schema_id,
        data_schema_sha256=snapshot.data_schema_sha256,
        normalization_version=snapshot.normalization_version,
        records=records,
    )
    documents = contract_documents()
    _m, _p, report = run_walk_forward(
        gapped,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )
    checks = {check.check_id: check for check in report.leak_checks}
    check = checks["validation_labels_outside_test"]
    assert check.status == "pass"
    from datetime import date as _date

    detail_max = _date.fromisoformat(check.detail.split("actual max label end ")[1].split(" ")[0])
    assert detail_max < report.test_start
