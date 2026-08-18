"""Leakage-safe reconstruction of the former 60-day slow-alpha design."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .auxiliary_race import run_feature_race
from .datasets import load_manifest, load_snapshot
from .feature_datasets import load_feature_dataset
from .model_race import CandidateSpec
from .robustness_race import _load_ranked_universe

RACE_ID = "daily-slow-alpha-reconstruction/v1"


@dataclass(frozen=True, order=True)
class SlowAlphaCandidate:
    candidate_id: str
    universe_size: int
    use_market_timing: bool
    top_k: int = 10
    model_kind: str = "hist_gradient_boosting_slow"
    rebalance_interval: int = 5
    model_refit_interval: int = 20
    feature_transform: str = "slow_market_auxiliary_rank"
    horizons: tuple[int, ...] = (60,)
    label_transform: str = "cross_sectional_demean"
    training_lookback_days: int = 120

    @property
    def model_spec(self) -> CandidateSpec:
        return CandidateSpec(
            candidate_id=self.candidate_id,
            model_kind=self.model_kind,
            top_k=self.top_k,
            rebalance_interval=self.rebalance_interval,
            model_refit_interval=self.model_refit_interval,
            feature_transform=self.feature_transform,
            horizons=self.horizons,
            label_transform=self.label_transform,
            training_lookback_days=self.training_lookback_days,
            use_market_timing=self.use_market_timing,
        )


CANDIDATES: tuple[SlowAlphaCandidate, ...] = tuple(
    SlowAlphaCandidate(
        candidate_id=(
            f"slow-hgb-u{universe_size}-top10-"
            f"{'timing' if use_market_timing else 'always'}"
        ),
        universe_size=universe_size,
        use_market_timing=use_market_timing,
    )
    for universe_size in (800, 500)
    for use_market_timing in (True, False)
)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings(
        "ignore",
        message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
        category=UserWarning,
    )
    parser = argparse.ArgumentParser(
        description="Run the leakage-safe 60-day slow-alpha reconstruction"
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
        raise FileExistsError(f"slow-alpha race output already exists: {args.output}")
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
    result = run_feature_race(
        snapshot=snapshot,
        feature_dataset=feature_dataset,
        ranked_symbols=ranked_symbols,
        universe_sha256=universe_sha256,
        repository_root=args.repository_root,
        generated_at=generated_at,
        candidates=CANDIDATES,
        race_id=RACE_ID,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
