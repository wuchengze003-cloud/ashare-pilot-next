"""Consumer and causality tests for immutable auxiliary feature data."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from types import MappingProxyType

import pytest
from ashare_research_app.backtest import PilotConfig, run_walk_forward
from ashare_research_app.canonical import canonical_json_bytes
from ashare_research_app.feature_datasets import (
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
    NORMALIZATION_VERSION,
    SOURCE_INDEX_FORMAT,
    FeatureDataset,
    HolderSnapshot,
    load_feature_dataset,
)
from ashare_research_app.features import (
    MARKET_AUXILIARY_FEATURE_NAMES,
    SLOW_MARKET_AUXILIARY_FEATURE_NAMES,
    build_feature_panel,
)
from test_baseline_ml import contract_documents, synthetic_snapshot


def _file_entry(*, role: str, path: str, content: bytes, rows: int) -> dict[str, object]:
    return {
        "role": role,
        "path": path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "row_count": rows,
        "file_size_bytes": len(content),
        "min_effective_date": None,
        "max_effective_date": None,
    }


def _write_feature_dataset(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    base_dataset_id = f"dataset-sha256-{'b' * 64}"
    base_manifest_sha256 = "c" * 64
    universe_sha256 = "d" * 64
    daily_rows = [
        {
            "symbol": symbol,
            "trade_date": trade_date,
            "turnover_rate_f": 2.0,
            "volume_ratio": 1.1,
            "pe_ttm": 10.0,
            "pb": 1.0,
            "ps_ttm": 2.0,
            "dv_ttm": 0.5,
            "total_mv": 1000.0,
            "circ_mv": 800.0,
            "net_mf_amount": 12.0,
        }
        for trade_date in ("2023-06-01", "2023-06-02")
        for symbol in ("000001.SZ", "600000.SH")
    ]
    holder_rows = [
        {
            "symbol": "000001.SZ",
            "ann_date": "2023-06-02",
            "end_date": "2023-05-31",
            "holder_num": 10000,
        }
    ]
    source_index = {
        "format": SOURCE_INDEX_FORMAT,
        "base_dataset_id": base_dataset_id,
        "universe_sha256": universe_sha256,
        "missing_holder_symbols": ["600000.SH"],
        "quarantined_holder_rows": [],
        "files": [
            {
                "path": "daily_basic/20230601.json",
                "sha256": "e" * 64,
                "file_size_bytes": 10,
            }
        ],
    }
    daily_bytes = canonical_json_bytes(daily_rows)
    holder_bytes = canonical_json_bytes(holder_rows)
    source_bytes = canonical_json_bytes(source_index)
    files = [
        _file_entry(
            role="daily_features",
            path="daily-features.json",
            content=daily_bytes,
            rows=len(daily_rows),
        ),
        _file_entry(
            role="holder_snapshots",
            path="holder-snapshots.json",
            content=holder_bytes,
            rows=len(holder_rows),
        ),
        _file_entry(
            role="source_index",
            path="source-index.json",
            content=source_bytes,
            rows=1,
        ),
    ]
    identity = {
        "base_dataset_id": base_dataset_id,
        "base_manifest_sha256": base_manifest_sha256,
        "universe_sha256": universe_sha256,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "normalization_version": NORMALIZATION_VERSION,
        "as_of": "2023-06-02",
        "source": "tushare-export-import",
        "source_version": "v1",
        "files": files,
    }
    feature_dataset_id = "feature-dataset-sha256-" + hashlib.sha256(
        canonical_json_bytes(identity)
    ).hexdigest()
    manifest = {
        "contract_id": "feature-dataset-manifest",
        "schema_version": "1.0.0",
        "feature_dataset_id": feature_dataset_id,
        **identity,
        "generated_at": "2026-08-18T14:00:00Z",
        "quality_status": "pass",
        "quality_reasons": ["OPTIONAL_HOLDER_COVERAGE_INCOMPLETE"],
        "coverage": {
            "daily_bar_rows": len(daily_rows),
            "daily_feature_rows": len(daily_rows),
            "daily_basic_present_rows": len(daily_rows),
            "moneyflow_present_rows": len(daily_rows),
            "holder_rows": len(holder_rows),
            "holder_symbols": 1,
            "missing_holder_symbols": 1,
            "quarantined_holder_rows": 0,
            "source_file_count": 1,
        },
    }
    dataset_root = tmp_path / feature_dataset_id
    dataset_root.mkdir()
    (dataset_root / "daily-features.json").write_bytes(daily_bytes)
    (dataset_root / "holder-snapshots.json").write_bytes(holder_bytes)
    (dataset_root / "source-index.json").write_bytes(source_bytes)
    manifest_path = tmp_path / f"{feature_dataset_id}.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    expected = {
        "base_dataset_id": base_dataset_id,
        "base_manifest_sha256": base_manifest_sha256,
        "universe_sha256": universe_sha256,
    }
    return manifest_path, dataset_root, expected


def test_feature_dataset_loader_verifies_identity_and_announcement_date(
    tmp_path: Path,
) -> None:
    manifest_path, dataset_root, expected = _write_feature_dataset(tmp_path)

    loaded = load_feature_dataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        expected_base_dataset_id=expected["base_dataset_id"],
        expected_base_manifest_sha256=expected["base_manifest_sha256"],
        expected_universe_sha256=expected["universe_sha256"],
        as_of=date(2023, 6, 2),
    )

    assert loaded.holder_values(symbol="000001.SZ", through=date(2023, 6, 1)) == (
        None,
        None,
    )
    assert loaded.holder_values(
        symbol="000001.SZ", through=date(2023, 6, 2)
    )[0] == pytest.approx(9.210340371976184)


def test_feature_dataset_loader_rejects_tampered_content(tmp_path: Path) -> None:
    manifest_path, dataset_root, expected = _write_feature_dataset(tmp_path)
    daily_path = dataset_root / "daily-features.json"
    daily_path.write_bytes(daily_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_feature_dataset(
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            expected_base_dataset_id=expected["base_dataset_id"],
            expected_base_manifest_sha256=expected["base_manifest_sha256"],
            expected_universe_sha256=expected["universe_sha256"],
            as_of=date(2023, 6, 2),
        )


def _synthetic_feature_dataset(
    *, future_holder_num: int = 9000, days: int = 80
) -> FeatureDataset:
    snapshot = synthetic_snapshot(days=days)
    calendar = sorted({bar.trade_date for bar in snapshot.records})
    daily_values = {
        (bar.symbol, bar.trade_date): (
            float(position % 17),
            None if position % 29 == 0 else 0.8 + position % 7 / 10,
            8.0 + position % 13,
            0.5 + position % 11 / 10,
            1.0 + position % 9 / 10,
            float(position % 5) / 10,
            1000.0 + position,
            800.0 + position,
            float((position % 19) - 9) * 100,
        )
        for position, bar in enumerate(snapshot.records)
    }
    holders = {
        symbol: (
            HolderSnapshot(
                symbol=symbol,
                ann_date=calendar[30],
                end_date=calendar[25],
                holder_num=10000,
            ),
            HolderSnapshot(
                symbol=symbol,
                ann_date=calendar[60],
                end_date=calendar[55],
                holder_num=future_holder_num,
            ),
        )
        for symbol in {bar.symbol for bar in snapshot.records}
    }
    return FeatureDataset(
        feature_dataset_id=f"feature-dataset-sha256-{'f' * 64}",
        base_dataset_id=snapshot.dataset_id,
        base_manifest_sha256=snapshot.manifest_sha256,
        universe_sha256="d" * 64,
        manifest_sha256="e" * 64,
        as_of=snapshot.as_of,
        daily_values_by_key=MappingProxyType(daily_values),
        holder_by_symbol=MappingProxyType(holders),
    )


def test_market_auxiliary_panel_is_ranked_and_point_in_time() -> None:
    snapshot = synthetic_snapshot()
    feature_dataset = _synthetic_feature_dataset()
    calendar = sorted({bar.trade_date for bar in snapshot.records})
    midpoint = calendar[45]

    full = build_feature_panel(
        snapshot,
        feature_transform="market_auxiliary_rank",
        feature_dataset=feature_dataset,
    )
    truncated = build_feature_panel(
        snapshot,
        as_of=midpoint,
        feature_transform="market_auxiliary_rank",
        feature_dataset=feature_dataset,
    )

    assert truncated == tuple(row for row in full if row.trade_date <= midpoint)
    assert all(len(row.values) == len(MARKET_AUXILIARY_FEATURE_NAMES) for row in full)
    assert all(-0.5 <= value <= 0.5 for row in full for value in row.values)


def test_future_holder_revision_cannot_change_earlier_features() -> None:
    snapshot = synthetic_snapshot()
    calendar = sorted({bar.trade_date for bar in snapshot.records})
    cutoff = calendar[55]
    first = build_feature_panel(
        snapshot,
        as_of=cutoff,
        feature_transform="market_auxiliary_rank",
        feature_dataset=_synthetic_feature_dataset(future_holder_num=9000),
    )
    revised = build_feature_panel(
        snapshot,
        as_of=cutoff,
        feature_transform="market_auxiliary_rank",
        feature_dataset=_synthetic_feature_dataset(future_holder_num=1000),
    )

    assert first == revised


def test_walk_forward_binds_auxiliary_dataset_and_passes_leak_checks() -> None:
    snapshot = synthetic_snapshot()
    feature_dataset = _synthetic_feature_dataset()
    documents = contract_documents()

    _model, _production, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(
            top_k=4,
            per_weight=0.24,
            model_kind="ridge",
            feature_transform="market_auxiliary_rank",
        ),
        feature_dataset=feature_dataset,
    )

    assert report.feature_dataset_id == feature_dataset.feature_dataset_id
    assert report.feature_dataset_manifest_sha256 == feature_dataset.manifest_sha256
    assert all(check.status == "pass" for check in report.leak_checks)


def test_slow_feature_panel_is_causal_and_reuses_point_in_time_inputs() -> None:
    snapshot = synthetic_snapshot(days=180)
    feature_dataset = _synthetic_feature_dataset(days=180)
    calendar = sorted({bar.trade_date for bar in snapshot.records})
    cutoff = calendar[155]

    full = build_feature_panel(
        snapshot,
        feature_transform="slow_market_auxiliary_rank",
        feature_dataset=feature_dataset,
    )
    truncated = build_feature_panel(
        snapshot,
        as_of=cutoff,
        feature_transform="slow_market_auxiliary_rank",
        feature_dataset=feature_dataset,
    )

    assert truncated == tuple(row for row in full if row.trade_date <= cutoff)
    assert all(len(row.values) == len(SLOW_MARKET_AUXILIARY_FEATURE_NAMES) for row in full)


def test_slow_walk_forward_uses_mature_60_day_labels_and_rolling_window() -> None:
    snapshot = synthetic_snapshot(days=360)
    feature_dataset = _synthetic_feature_dataset(days=360)
    documents = contract_documents()

    _model, _production, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(
            top_k=4,
            per_weight=0.24,
            model_kind="ridge",
            feature_transform="slow_market_auxiliary_rank",
            label_transform="cross_sectional_demean",
            training_lookback_days=120,
            use_market_timing=True,
        ),
        horizons=(60,),
        feature_dataset=feature_dataset,
    )

    assert report.label_transform == "cross_sectional_demean"
    assert report.training_lookback_days == 120
    assert report.use_market_timing is True
    assert report.market_regime_latest in {"INVESTED", "FLAT"}
    assert report.oos_max_label_end < report.latest_signal_date
    assert all(check.status == "pass" for check in report.leak_checks)
