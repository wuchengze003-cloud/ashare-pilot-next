"""Pre-registered robustness race on immutable point-in-time auxiliary data.

All available history has already been observed, so this command can reject a
candidate but cannot promote one. A historical pass still requires prospective
shadow evidence, production feature parity, and explicit human approval.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ashare_quant_core import DatasetSnapshot

from .datasets import load_manifest, load_snapshot
from .feature_datasets import FeatureDataset, load_feature_dataset
from .model_race import CandidateSpec, _run_candidate, _summary
from .robustness_race import (
    FOLDS,
    _aggregate,
    _load_ranked_universe,
    _research_view,
    _spa,
)

RACE_ID = "daily-market-auxiliary-robustness/v1"
SHARPE_THRESHOLD = 1.5
MINIMUM_EXECUTED_TRADES = 20


@dataclass(frozen=True, order=True)
class AuxiliaryCandidate:
    candidate_id: str
    universe_size: int
    top_k: int
    model_kind: str
    rebalance_interval: int = 5
    model_refit_interval: int = 20
    feature_transform: str = "market_auxiliary_rank"

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


CANDIDATES: tuple[AuxiliaryCandidate, ...] = tuple(
    AuxiliaryCandidate(
        candidate_id=f"aux-{model_kind}-u{universe_size}-top{top_k}-r5",
        universe_size=universe_size,
        top_k=top_k,
        model_kind=model_kind,
    )
    for model_kind in ("ridge", "hist_gradient_boosting")
    for universe_size in (800, 500)
    for top_k in (5, 20)
)


def run_auxiliary_race(
    *,
    snapshot: DatasetSnapshot,
    feature_dataset: FeatureDataset,
    ranked_symbols: tuple[str, ...],
    universe_sha256: str,
    repository_root: Path,
    generated_at: datetime,
) -> dict[str, Any]:
    feature_dataset.assert_matches_snapshot(snapshot)
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
                feature_dataset=feature_dataset,
            )
            reports_by_candidate[candidate.candidate_id].append(report)
            fold_summary = _summary(candidate.model_spec, report)
            fold_summary["fold_id"] = fold.fold_id
            fold_summary["universe_size"] = candidate.universe_size
            fold_summary["feature_dataset_id"] = report.feature_dataset_id
            fold_results[candidate.candidate_id].append(fold_summary)

    aggregate_internal = {
        candidate_id: _aggregate(reports)
        for candidate_id, reports in reports_by_candidate.items()
    }
    common_dates = aggregate_internal[CANDIDATES[0].candidate_id]["dates"]
    if any(result["dates"] != common_dates for result in aggregate_internal.values()):
        raise ValueError("candidate robustness periods do not share dates")
    model_returns = {
        candidate_id: result["returns"]
        for candidate_id, result in aggregate_internal.items()
    }
    spa = _spa(
        benchmark_returns=tuple(0.0 for _ in common_dates),
        model_returns=model_returns,
    )
    spa["benchmark"] = "cash-zero-return"

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
        and int(selected["executed_trades"]) >= MINIMUM_EXECUTED_TRADES
        and bool(selected["leak_checks_passed"])
        and statistically_superior
    )
    return {
        "race_id": RACE_ID,
        "dataset_id": snapshot.dataset_id,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "feature_dataset_id": feature_dataset.feature_dataset_id,
        "feature_dataset_manifest_sha256": feature_dataset.manifest_sha256,
        "source_universe_sha256": universe_sha256,
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "data_frequency": "daily",
        "sharpe_threshold": SHARPE_THRESHOLD,
        "minimum_executed_trades": MINIMUM_EXECUTED_TRADES,
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
            "requires_prospective_shadow_and_production_parity"
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
    parser = argparse.ArgumentParser(
        description="Run the immutable point-in-time auxiliary feature race"
    )
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--feature-dataset-root", required=True, type=Path)
    parser.add_argument("--source-universe", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"auxiliary race output already exists: {args.output}")
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
    feature_dataset = load_feature_dataset(
        manifest_path=args.feature_manifest,
        dataset_root=args.feature_dataset_root,
        expected_base_dataset_id=snapshot.dataset_id,
        expected_base_manifest_sha256=snapshot.manifest_sha256,
        expected_universe_sha256=universe_sha256,
        as_of=as_of,
    )
    result = run_auxiliary_race(
        snapshot=snapshot,
        feature_dataset=feature_dataset,
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
