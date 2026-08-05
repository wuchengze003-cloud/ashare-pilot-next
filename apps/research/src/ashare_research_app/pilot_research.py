"""Public command: run the pilot research cycle on one immutable dataset.

Loads the dataset snapshot, runs the walk-forward backtest, promotes the
trained baseline model, and writes a research report for the web-state
producer. All outputs live under an explicit runtime root.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from ashare_quant_core import mark_to_market

from .backtest import PilotConfig, run_walk_forward
from .datasets import load_manifest, load_snapshot
from .features import FEATURE_NAMES
from .promotion import promote_baseline_model


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def build_research_report(report, *, initial_capital) -> dict:
    final_state = report.final_state
    prices = report.final_prices
    market_value = float(
        mark_to_market(
            final_state,
            prices={symbol: prices[symbol] for symbol in final_state.holdings if symbol in prices},
        )
        - final_state.cash
    ) if final_state.holdings else 0.0
    total_assets = float(
        mark_to_market(
            final_state,
            prices={symbol: prices[symbol] for symbol in final_state.holdings if symbol in prices},
        )
    ) if final_state.holdings else float(final_state.cash)
    positions = [
        {
            "symbol": symbol,
            "shares": holding.shares,
            "locked_shares": holding.locked_shares,
            "avg_cost": float(holding.avg_cost),
            "last_price": float(prices[symbol]),
            "market_value": float(holding.shares * prices[symbol]),
            "weight": round(float(holding.shares * prices[symbol]) / total_assets, 6)
            if total_assets > 0
            else 0.0,
        }
        for symbol, holding in sorted(final_state.holdings.items())
        if symbol in prices
    ]
    return {
        "report_id": "pilot-research-report/v1",
        "dataset_id": report.dataset_id,
        "snapshot_sha256": report.snapshot_sha256,
        "model_bundle_sha256": report.model_bundle_sha256,
        "training_cutoff": report.training_cutoff.isoformat(),
        "validation_cutoff": report.validation_end.isoformat(),
        "validation_ic_mean": report.validation_ic_mean,
        "train_end": report.train_end.isoformat(),
        "test_start": report.test_start.isoformat(),
        "test_end": report.test_end.isoformat(),
        "splits": {
            "in_sample": f"{report.train_end.isoformat()}",
            "validation": (
                f"{report.validation_start.isoformat()}/{report.validation_end.isoformat()}"
            ),
            "out_of_sample": (
                f"{report.first_nav_date.isoformat()}/{report.test_end.isoformat()}"
            ),
        },
        "production_training_cutoff": report.production_training_cutoff.isoformat(),
        "first_nav_date": report.first_nav_date.isoformat(),
        "frozen_valuations": list(report.frozen_valuations),
        "metrics": dict(report.metrics),
        "nav_curve": [dict(point) for point in report.nav_curve],
        "trades": [dict(trade) for trade in report.trades],
        "recommendations": [dict(item) for item in report.recommendations],
        "feature_weights": [dict(item) for item in report.feature_weights],
        "leak_checks": [
            {"check_id": check.check_id, "status": check.status, "detail": check.detail}
            for check in report.leak_checks
        ],
        "latest_signal_date": report.latest_signal_date.isoformat(),
        "portfolio": {
            "initial_capital": float(initial_capital),
            "cash": float(final_state.cash),
            "market_value": market_value,
            "total_assets": total_assets,
            "positions": positions,
        },
        "buy_dates": _jsonable(report.buy_dates),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pilot research cycle")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--per-weight", type=float, default=0.2)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    args = parser.parse_args(argv)

    repository_root = Path(args.repository_root)
    runtime_root = Path(args.runtime_root)
    manifest = load_manifest(Path(args.dataset_manifest))
    as_of_text = str(manifest["as_of"])
    snapshot = load_snapshot(
        manifest=manifest,
        dataset_root=Path(args.dataset_root),
        as_of=datetime.strptime(as_of_text, "%Y-%m-%d").date(),
    )

    cost_model_path = repository_root / "contracts/examples/cost-model.example.json"
    cost_model_doc = json.loads(cost_model_path.read_text(encoding="utf-8"))
    market_rules_doc = {
        "contract_id": "market-rules",
        "schema_version": "1.0.0",
        "rules_id": "pilot-a-share-market/v1",
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
    execution_policy_doc = {
        "contract_id": "execution-policy",
        "schema_version": "1.0.0",
        "policy_id": "pilot-next-open/v1",
        "decision_timing": "daily_close",
        "execution_timing": "next_open",
        "slippage_bps": 10,
        "impact_model": "linear_participation",
        "maximum_participation_rate": 0.05,
        "enforce_t_plus_one": True,
    }
    portfolio_risk_doc = {
        "contract_id": "portfolio-risk",
        "schema_version": "1.0.0",
        "risk_id": "pilot-aggressive-long-only/v1",
        "max_positions": 6,
        "max_single_weight": 0.25,
        "max_theme_weight": 1.0,
        "max_gross_exposure": 1.0,
        "rebalance_threshold": 0.02,
    }

    config = PilotConfig(
        initial_capital=Decimal(str(args.initial_capital)),
        top_k=args.top_k,
        per_weight=args.per_weight,
    )
    model, production_model, report = run_walk_forward(
        snapshot,
        cost_model_doc=cost_model_doc,
        market_rules_doc=market_rules_doc,
        execution_policy_doc=execution_policy_doc,
        portfolio_risk_doc=portfolio_risk_doc,
        config=config,
    )
    generated_at = datetime.fromisoformat(args.generated_at.replace("Z", "+00:00"))
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)
    paths = promote_baseline_model(
        repository_root=repository_root,
        runtime_root=runtime_root,
        model_bundle_bytes=production_model.bundle_bytes(),
        report=report,
        dataset_manifest=manifest,
        snapshot_symbols=tuple(sorted({bar.symbol for bar in snapshot.records})),
        as_of=as_of_text,
        generated_at=generated_at,
        top_k=args.top_k,
        per_weight=args.per_weight,
        feature_names=FEATURE_NAMES,
    )

    runtime_root.mkdir(parents=True, exist_ok=True)
    research_report = build_research_report(report, initial_capital=config.initial_capital)
    research_report["as_of"] = as_of_text
    research_report["latest_prices"] = {
        symbol: float(price) for symbol, price in report.final_prices.items()
    }
    research_report["scores_latest"] = {
        symbol: float(score) for symbol, score in report.scores_latest.items()
    }
    report_path = runtime_root / "research-report.json"
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(json.dumps(research_report, ensure_ascii=True, indent=2) + "\n")
    temporary.replace(report_path)

    json.dump(
        {
            "as_of": as_of_text,
            "dataset_id": snapshot.dataset_id,
            "contracts_dir": str(paths.contracts_dir),
            "adapter_root": str(paths.adapter_root),
            "champion_path": str(paths.champion_path),
            "research_report": str(report_path),
            "validation_ic_mean": report.validation_ic_mean,
        },
        sys.stdout,
        ensure_ascii=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
