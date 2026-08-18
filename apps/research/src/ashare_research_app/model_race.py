"""Pre-registered model race on one immutable dataset.

The race uses the existing walk-forward engine and quant_core simulation. Six
candidate configurations are evaluated on a snapshot truncated before the
final window. Only the best validation Sharpe is then evaluated once on the
full final window. The command never promotes or activates a model.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch.bootstrap import SPA
from ashare_quant_core import DatasetSnapshot

from .backtest import BacktestReport, PilotConfig, _split_dates, run_walk_forward
from .baseline_model import TrainingWindow
from .datasets import load_manifest, load_snapshot
from .feature_datasets import FeatureDataset
from .performance import annualized_sharpe, returns_from_nav_curve
from .promotion import (
    PILOT_EXECUTION_POLICY,
    PILOT_MARKET_RULES,
    evaluate_promotion_gate,
    load_pilot_cost_model,
)

SHARPE_THRESHOLD = 1.5
INITIAL_CAPITAL = Decimal("1000000")
RACE_ID = "daily-sklearn-race/v1"


@dataclass(frozen=True, order=True)
class CandidateSpec:
    candidate_id: str
    model_kind: str
    top_k: int
    rebalance_interval: int = 5
    model_refit_interval: int = 20
    feature_transform: str = "raw"

    @property
    def per_weight(self) -> float:
        return 1.0 / self.top_k


CANDIDATES: tuple[CandidateSpec, ...] = tuple(
    CandidateSpec(
        candidate_id=f"{model_kind}-top{top_k}",
        model_kind=model_kind,
        top_k=top_k,
    )
    for model_kind in ("ridge", "hist_gradient_boosting", "extra_trees")
    for top_k in (5, 20)
)


def _portfolio_risk() -> dict[str, object]:
    return {
        "contract_id": "portfolio-risk",
        "schema_version": "1.0.0",
        "risk_id": "model-race-unconstrained-count/v1",
        "max_positions": 500,
        "max_single_weight": 1.0,
        "max_theme_weight": 1.0,
        "max_gross_exposure": 1.0,
        "rebalance_threshold": 0.02,
    }


def _truncate_snapshot(snapshot: DatasetSnapshot, *, as_of: date) -> DatasetSnapshot:
    if as_of >= snapshot.as_of:
        raise ValueError("validation snapshot must end before the full snapshot")
    return DatasetSnapshot(
        dataset_id=snapshot.dataset_id,
        dataset_family_id=snapshot.dataset_family_id,
        manifest_sha256=snapshot.manifest_sha256,
        as_of=as_of,
        data_schema_id=snapshot.data_schema_id,
        data_schema_sha256=snapshot.data_schema_sha256,
        normalization_version=snapshot.normalization_version,
        records=tuple(record for record in snapshot.records if record.trade_date <= as_of),
    )


def _run_candidate(
    *,
    snapshot: DatasetSnapshot,
    spec: CandidateSpec,
    repository_root: Path,
    training_window: TrainingWindow | None = None,
    feature_dataset: FeatureDataset | None = None,
) -> BacktestReport:
    _model, _production, report = run_walk_forward(
        snapshot,
        cost_model_doc=load_pilot_cost_model(repository_root),
        market_rules_doc=PILOT_MARKET_RULES,
        execution_policy_doc=PILOT_EXECUTION_POLICY,
        portfolio_risk_doc=_portfolio_risk(),
        config=PilotConfig(
            initial_capital=INITIAL_CAPITAL,
            top_k=spec.top_k,
            per_weight=spec.per_weight,
            rebalance_interval=spec.rebalance_interval,
            model_refit_interval=spec.model_refit_interval,
            model_kind=spec.model_kind,
            feature_transform=spec.feature_transform,
        ),
        training_window=training_window,
        feature_dataset=feature_dataset,
    )
    return report


def _summary(spec: CandidateSpec, report: BacktestReport) -> dict[str, object]:
    returns = returns_from_nav_curve(report.nav_curve)
    executed = sum(1 for trade in report.trades if int(trade["shares"]) > 0)
    return {
        **asdict(spec),
        "per_weight": spec.per_weight,
        "evaluation_start": report.first_nav_date.isoformat(),
        "evaluation_end": report.test_end.isoformat(),
        "sharpe": round(annualized_sharpe(returns), 6),
        "total_return": report.metrics["total_return"],
        "benchmark_total_return": report.metrics["benchmark_total_return"],
        "max_drawdown": report.metrics["max_drawdown"],
        "turnover": report.metrics["turnover"],
        "executed_trades": executed,
        "validation_ic_mean": round(report.validation_ic_mean, 6),
        "leak_checks_passed": all(check.status == "pass" for check in report.leak_checks),
    }


def _spa_result(reports: dict[str, BacktestReport]) -> dict[str, object]:
    ordered_ids = tuple(reports)
    first = reports[ordered_ids[0]]
    dates = tuple(str(point["trade_date"]) for point in first.nav_curve[1:])
    benchmark_returns = returns_from_nav_curve(first.nav_curve, field="benchmark_nav")
    model_returns: dict[str, tuple[float, ...]] = {}
    for candidate_id in ordered_ids:
        report = reports[candidate_id]
        candidate_dates = tuple(str(point["trade_date"]) for point in report.nav_curve[1:])
        if candidate_dates != dates:
            raise ValueError("candidate validation curves do not share dates")
        if returns_from_nav_curve(report.nav_curve, field="benchmark_nav") != benchmark_returns:
            raise ValueError("candidate validation curves do not share a benchmark")
        model_returns[candidate_id] = returns_from_nav_curve(report.nav_curve)
    model_frame = pd.DataFrame(model_returns, index=dates)
    comparison = SPA(
        -np.asarray(benchmark_returns),
        -model_frame,
        reps=2000,
        bootstrap="stationary",
        seed=20260818,
    )
    comparison.compute()
    pvalues = {str(key): float(value) for key, value in comparison.pvalues.items()}
    superior = [
        str(model_frame.columns[int(index)])
        for index in comparison.better_models(pvalue=0.05)
    ]
    return {
        "method": "arch.bootstrap.SPA",
        "repetitions": 2000,
        "seed": 20260818,
        "pvalues": pvalues,
        "superior_to_benchmark_at_5pct": superior,
    }


def run_race(
    *,
    snapshot: DatasetSnapshot,
    repository_root: Path,
    generated_at: datetime,
) -> dict[str, Any]:
    dates = tuple(sorted({record.trade_date for record in snapshot.records}))
    _train_end, _validation_start, validation_end, final_start = _split_dates(dates)
    validation_snapshot = _truncate_snapshot(snapshot, as_of=validation_end)
    validation_reports: dict[str, BacktestReport] = {}
    validation_summaries: list[dict[str, object]] = []
    for spec in CANDIDATES:
        print(f"validation candidate {spec.candidate_id}", flush=True)
        report = _run_candidate(
            snapshot=validation_snapshot,
            spec=spec,
            repository_root=repository_root,
        )
        validation_reports[spec.candidate_id] = report
        validation_summaries.append(_summary(spec, report))
    validation_summaries.sort(
        key=lambda item: (-float(item["sharpe"]), str(item["candidate_id"]))
    )
    selected_id = str(validation_summaries[0]["candidate_id"])
    selected_spec = next(spec for spec in CANDIDATES if spec.candidate_id == selected_id)
    print(f"final candidate {selected_id}", flush=True)
    final_report = _run_candidate(
        snapshot=snapshot,
        spec=selected_spec,
        repository_root=repository_root,
    )
    gate = {
        "gate_id": "sharpe-first-promotion/v1",
        "maximum_drawdown": 1.0,
        "minimum_trades": 20,
        "minimum_sharpe": SHARPE_THRESHOLD,
        "minimum_validation_ic": 0.0,
        "maximum_top_trade_profit_share": 0.35,
        "require_untouched_final_window": True,
    }
    gate_evaluation = evaluate_promotion_gate(final_report, gate)
    return {
        "race_id": RACE_ID,
        "dataset_id": snapshot.dataset_id,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "initial_capital": float(INITIAL_CAPITAL),
        "sharpe_threshold": SHARPE_THRESHOLD,
        "trial_count": len(CANDIDATES),
        "selection_data_end": validation_end.isoformat(),
        "final_data_start": final_start.isoformat(),
        "candidate_registry": [asdict(spec) for spec in CANDIDATES],
        "validation_results": validation_summaries,
        "spa": _spa_result(validation_reports),
        "selected_candidate": selected_id,
        "final_result": _summary(selected_spec, final_report),
        "promotion_gate_evaluation": gate_evaluation,
        "candidate_status": (
            "eligible_for_human_review"
            if gate_evaluation["status"] == "pass"
            else "rejected"
        ),
        "production_state_changed": False,
    }


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings(
        "ignore",
        message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
        category=UserWarning,
    )
    parser = argparse.ArgumentParser(description="Run the pre-registered daily model race")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"race output already exists: {args.output}")
    generated_at = datetime.fromisoformat(args.generated_at.replace("Z", "+00:00"))
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated-at must be timezone-aware")
    manifest = load_manifest(args.dataset_manifest)
    as_of = date.fromisoformat(str(manifest["as_of"]))
    snapshot = load_snapshot(manifest=manifest, dataset_root=args.dataset_root, as_of=as_of)
    result = run_race(
        snapshot=snapshot,
        repository_root=args.repository_root,
        generated_at=generated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
