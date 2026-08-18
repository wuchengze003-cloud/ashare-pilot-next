"""Multi-period robustness race after the first holdout was opened.

This command treats all available history as development evidence. It uses
four non-overlapping out-of-sample periods, corrects for the pre-registered
candidate count with Hansen's SPA test, and never promotes or activates a
model. A historical winner still requires prospective shadow evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch.bootstrap import SPA
from ashare_quant_core import DatasetSnapshot

from .baseline_model import TrainingWindow
from .datasets import load_manifest, load_snapshot
from .model_race import CandidateSpec, _run_candidate, _summary
from .performance import annualized_sharpe, returns_from_nav_curve

RACE_ID = "daily-cross-sectional-robustness/v1"
SHARPE_THRESHOLD = 1.5
EXPECTED_SOURCE_UNIVERSE = "fixed-top800-market-cap"


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date

    @property
    def training_window(self) -> TrainingWindow:
        return TrainingWindow(
            train_end=self.train_end,
            validation_start=self.validation_start,
            validation_end=self.validation_end,
            test_start=self.test_start,
        )


FOLDS: tuple[FoldSpec, ...] = (
    FoldSpec(
        "fold-1",
        date(2024, 7, 23),
        date(2024, 7, 24),
        date(2024, 11, 21),
        date(2024, 11, 22),
        date(2025, 4, 22),
    ),
    FoldSpec(
        "fold-2",
        date(2024, 12, 19),
        date(2024, 12, 20),
        date(2025, 4, 22),
        date(2025, 4, 23),
        date(2025, 9, 15),
    ),
    FoldSpec(
        "fold-3",
        date(2025, 5, 23),
        date(2025, 5, 26),
        date(2025, 9, 15),
        date(2025, 9, 16),
        date(2026, 2, 12),
    ),
    FoldSpec(
        "fold-4",
        date(2025, 10, 21),
        date(2025, 10, 22),
        date(2026, 2, 12),
        date(2026, 2, 13),
        date(2026, 8, 17),
    ),
)


@dataclass(frozen=True, order=True)
class RobustCandidate:
    candidate_id: str
    universe_size: int
    top_k: int
    rebalance_interval: int
    feature_transform: str
    model_kind: str = "ridge"
    model_refit_interval: int = 20

    @property
    def model_spec(self) -> CandidateSpec:
        return CandidateSpec(
            candidate_id=self.candidate_id,
            model_kind=self.model_kind,
            top_k=self.top_k,
            rebalance_interval=self.rebalance_interval,
            model_refit_interval=self.model_refit_interval,
            feature_transform=self.feature_transform,
        )


CANDIDATES: tuple[RobustCandidate, ...] = (
    RobustCandidate("raw-u800-top5-r5", 800, 5, 5, "raw"),
    *(
        RobustCandidate(
            candidate_id=f"rank-u{universe_size}-top{top_k}-r{rebalance}",
            universe_size=universe_size,
            top_k=top_k,
            rebalance_interval=rebalance,
            feature_transform="cross_sectional_rank",
        )
        for universe_size in (500, 800)
        for top_k in (5, 20)
        for rebalance in (5, 10)
    ),
)


def _load_ranked_universe(path: Path) -> tuple[tuple[str, ...], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("source universe must be a regular file")
    content = path.read_bytes()
    document = json.loads(content)
    if not isinstance(document, dict):
        raise ValueError("source universe must be an object")
    if document.get("contract_id") != "universe":
        raise ValueError("source universe contract_id is invalid")
    if document.get("universe_policy_id") != EXPECTED_SOURCE_UNIVERSE:
        raise ValueError("source universe is not the audited top-800 ranking")
    members = document.get("members")
    if not isinstance(members, list) or len(members) != 800:
        raise ValueError("source universe must contain 800 ranked members")
    ranked = tuple(str(member["symbol"]) for member in members)
    if len(set(ranked)) != len(ranked):
        raise ValueError("source universe contains duplicate symbols")
    return ranked, hashlib.sha256(content).hexdigest()


def _research_view(
    snapshot: DatasetSnapshot,
    *,
    ranked_symbols: tuple[str, ...],
    universe_size: int,
    as_of: date,
) -> DatasetSnapshot:
    if not 20 <= universe_size <= len(ranked_symbols):
        raise ValueError("research universe size is outside the ranked source")
    if as_of > snapshot.as_of:
        raise ValueError("research view cannot exceed the immutable snapshot")
    allowed = frozenset(ranked_symbols[:universe_size])
    records = tuple(
        record
        for record in snapshot.records
        if record.symbol in allowed and record.trade_date <= as_of
    )
    if {record.symbol for record in records} != set(allowed):
        raise ValueError("research view lost one or more ranked members")
    return DatasetSnapshot(
        dataset_id=snapshot.dataset_id,
        dataset_family_id=snapshot.dataset_family_id,
        manifest_sha256=snapshot.manifest_sha256,
        as_of=as_of,
        data_schema_id=snapshot.data_schema_id,
        data_schema_sha256=snapshot.data_schema_sha256,
        normalization_version=snapshot.normalization_version,
        records=records,
    )


def _aggregate(reports: list[Any]) -> dict[str, object]:
    returns: list[float] = []
    dates: list[str] = []
    executed_trades = 0
    total_cost = 0.0
    weighted_turnover = 0.0
    weighted_days = 0
    validation_ics: list[float] = []
    leak_checks_passed = True
    for report in reports:
        fold_returns = returns_from_nav_curve(report.nav_curve)
        fold_dates = [str(point["trade_date"]) for point in report.nav_curve[1:]]
        if len(fold_dates) != len(fold_returns):
            raise ValueError("fold NAV dates and returns do not align")
        if dates and fold_dates[0] <= dates[-1]:
            raise ValueError("fold out-of-sample dates overlap")
        dates.extend(fold_dates)
        returns.extend(fold_returns)
        executed = [trade for trade in report.trades if int(trade["shares"]) > 0]
        executed_trades += len(executed)
        total_cost += sum(float(trade["total_cost"]) for trade in executed)
        weighted_turnover += float(report.metrics["turnover"]) * len(fold_returns)
        weighted_days += len(fold_returns)
        validation_ics.append(report.validation_ic_mean)
        leak_checks_passed = leak_checks_passed and all(
            check.status == "pass" for check in report.leak_checks
        )
    if not returns:
        raise ValueError("robustness aggregation has no returns")
    equity = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak)
    return {
        "dates": tuple(dates),
        "returns": tuple(returns),
        "sharpe": round(annualized_sharpe(returns), 6),
        "total_return": round(equity - 1.0, 6),
        "maximum_drawdown": round(maximum_drawdown, 6),
        "average_daily_turnover": round(weighted_turnover / weighted_days, 6),
        "executed_trades": executed_trades,
        "total_cost": round(total_cost, 2),
        "mean_validation_ic": round(sum(validation_ics) / len(validation_ics), 6),
        "leak_checks_passed": leak_checks_passed,
    }


def _spa(
    *,
    benchmark_returns: tuple[float, ...],
    model_returns: dict[str, tuple[float, ...]],
) -> dict[str, object]:
    if any(len(values) != len(benchmark_returns) for values in model_returns.values()):
        raise ValueError("SPA inputs do not have a common length")
    model_frame = pd.DataFrame(model_returns)
    comparison = SPA(
        -np.asarray(benchmark_returns),
        -model_frame,
        reps=2000,
        bootstrap="stationary",
        seed=20260818,
    )
    comparison.compute()
    return {
        "method": "arch.bootstrap.SPA",
        "repetitions": 2000,
        "seed": 20260818,
        "pvalues": {
            str(key): float(value) for key, value in comparison.pvalues.items()
        },
        "superior_to_benchmark_at_5pct": [
            str(model_frame.columns[int(index)])
            for index in comparison.better_models(pvalue=0.05)
        ],
    }


def run_robustness_race(
    *,
    snapshot: DatasetSnapshot,
    ranked_symbols: tuple[str, ...],
    universe_sha256: str,
    repository_root: Path,
    generated_at: datetime,
) -> dict[str, Any]:
    available_dates = {record.trade_date for record in snapshot.records}
    required_dates = {
        boundary
        for fold in FOLDS
        for boundary in (
            fold.train_end,
            fold.validation_start,
            fold.validation_end,
            fold.test_start,
            fold.test_end,
        )
    }
    if not required_dates <= available_dates:
        raise ValueError("immutable dataset does not cover the pre-registered folds")

    reports_by_candidate: dict[str, list[Any]] = {
        candidate.candidate_id: [] for candidate in CANDIDATES
    }
    fold_results: dict[str, list[dict[str, object]]] = {
        candidate.candidate_id: [] for candidate in CANDIDATES
    }
    view_hashes: dict[str, dict[str, str]] = {}
    for fold in FOLDS:
        view_hashes[fold.fold_id] = {}
        views: dict[int, DatasetSnapshot] = {}
        for universe_size in sorted({candidate.universe_size for candidate in CANDIDATES}):
            views[universe_size] = _research_view(
                snapshot,
                ranked_symbols=ranked_symbols,
                universe_size=universe_size,
                as_of=fold.test_end,
            )
            view_hashes[fold.fold_id][str(universe_size)] = views[
                universe_size
            ].snapshot_sha256
        for candidate in CANDIDATES:
            print(f"{fold.fold_id} {candidate.candidate_id}", flush=True)
            report = _run_candidate(
                snapshot=views[candidate.universe_size],
                spec=candidate.model_spec,
                repository_root=repository_root,
                training_window=fold.training_window,
            )
            reports_by_candidate[candidate.candidate_id].append(report)
            fold_summary = _summary(candidate.model_spec, report)
            fold_summary["fold_id"] = fold.fold_id
            fold_summary["universe_size"] = candidate.universe_size
            fold_results[candidate.candidate_id].append(fold_summary)

    aggregate_internal = {
        candidate_id: _aggregate(reports)
        for candidate_id, reports in reports_by_candidate.items()
    }
    common_dates = aggregate_internal[CANDIDATES[0].candidate_id]["dates"]
    if any(result["dates"] != common_dates for result in aggregate_internal.values()):
        raise ValueError("candidate robustness periods do not share dates")
    benchmark_returns: list[float] = []
    for report in reports_by_candidate[CANDIDATES[0].candidate_id]:
        benchmark_returns.extend(
            returns_from_nav_curve(report.nav_curve, field="benchmark_nav")
        )
    model_returns = {
        candidate_id: result["returns"]
        for candidate_id, result in aggregate_internal.items()
    }
    spa = _spa(
        benchmark_returns=tuple(benchmark_returns),
        model_returns=model_returns,
    )

    public_aggregates: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        internal = aggregate_internal[candidate.candidate_id]
        public_aggregates.append(
            {
                **asdict(candidate),
                **{
                    key: value
                    for key, value in internal.items()
                    if key not in {"dates", "returns"}
                },
            }
        )
    public_aggregates.sort(
        key=lambda item: (-float(item["sharpe"]), str(item["candidate_id"]))
    )
    selected = public_aggregates[0]
    selected_id = str(selected["candidate_id"])
    statistically_superior = selected_id in set(spa["superior_to_benchmark_at_5pct"])
    historical_gate_passed = (
        float(selected["sharpe"]) >= SHARPE_THRESHOLD
        and bool(selected["leak_checks_passed"])
        and statistically_superior
    )
    return {
        "race_id": RACE_ID,
        "dataset_id": snapshot.dataset_id,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "source_universe_sha256": universe_sha256,
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "data_frequency": "daily",
        "sharpe_threshold": SHARPE_THRESHOLD,
        "candidate_count": len(CANDIDATES),
        "fold_count": len(FOLDS),
        "holdout_status": "historical_window_previously_observed",
        "candidate_registry": [asdict(candidate) for candidate in CANDIDATES],
        "fold_registry": [
            {
                "fold_id": fold.fold_id,
                "train_end": fold.train_end.isoformat(),
                "validation_start": fold.validation_start.isoformat(),
                "validation_end": fold.validation_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
            }
            for fold in FOLDS
        ],
        "derived_view_hashes": view_hashes,
        "fold_results": fold_results,
        "aggregate_results": public_aggregates,
        "spa": spa,
        "selected_candidate": selected_id,
        "historical_gate_passed": historical_gate_passed,
        "candidate_status": (
            "requires_prospective_shadow"
            if historical_gate_passed
            else "rejected"
        ),
        "promotion_eligible": False,
        "production_state_changed": False,
    }


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings(
        "ignore",
        message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
        category=UserWarning,
    )
    parser = argparse.ArgumentParser(description="Run the multi-period robustness race")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--source-universe", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"robustness output already exists: {args.output}")
    generated_at = datetime.fromisoformat(args.generated_at.replace("Z", "+00:00"))
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated-at must be timezone-aware")
    manifest = load_manifest(args.dataset_manifest)
    as_of = date.fromisoformat(str(manifest["as_of"]))
    snapshot = load_snapshot(
        manifest=manifest,
        dataset_root=args.dataset_root,
        as_of=as_of,
    )
    ranked_symbols, universe_sha256 = _load_ranked_universe(args.source_universe)
    result = run_robustness_race(
        snapshot=snapshot,
        ranked_symbols=ranked_symbols,
        universe_sha256=universe_sha256,
        repository_root=args.repository_root,
        generated_at=generated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
