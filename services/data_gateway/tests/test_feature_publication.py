"""Synthetic tests for immutable point-in-time feature publication."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from ashare_data_gateway.dataset_publication import (
    prepare_normalized_dataset,
    publish_normalized_dataset,
    source_identity_from_base_url,
)
from ashare_data_gateway.feature_publication import (
    FEATURE_SCHEMA_DESCRIPTOR,
    FEATURE_SCHEMA_SHA256,
    HOLDER_SNAPSHOT_PATH,
    SOURCE_INDEX_PATH,
    prepare_feature_dataset,
    publish_feature_dataset,
    validate_prepared_feature_dataset,
)
from ashare_data_gateway.normalization import NormalizedDailyBar, canonical_json_bytes
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[3]
SYMBOLS = ("000001.SZ", "600000.SH", "600519.SH")
DATES = (date(2023, 6, 1), date(2023, 6, 2))
GENERATED_AT = datetime(2026, 8, 18, 14, 0, tzinfo=UTC)


def _publish_base(tmp_path: Path) -> tuple[Path, str]:
    records = tuple(
        NormalizedDailyBar(
            symbol=symbol,
            trade_date=trade_date,
            open=10.0,
            high=10.5,
            low=9.5,
            close=10.0,
            volume=100.0,
            amount=1000.0,
        )
        for symbol in SYMBOLS
        for trade_date in DATES
    )
    prepared = prepare_normalized_dataset(
        records,
        as_of=DATES[-1],
        generated_at=GENERATED_AT,
        source_identity=source_identity_from_base_url("https://api.tushare.pro"),
        source_version="synthetic/v1",
    )
    root = tmp_path / "base"
    publish_normalized_dataset(publication_root=root, prepared=prepared)
    return root, prepared.dataset_id


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps(
            {
                "contract_id": "universe",
                "members": [{"symbol": symbol} for symbol in SYMBOLS],
            }
        ),
        encoding="utf-8",
    )
    daily_root = tmp_path / "daily_basic"
    moneyflow_root = tmp_path / "moneyflow"
    holder_root = tmp_path / "holder"
    daily_root.mkdir()
    moneyflow_root.mkdir()
    holder_root.mkdir()
    daily_fields = [
        "ts_code",
        "trade_date",
        "turnover_rate_f",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "dv_ttm",
        "total_mv",
        "circ_mv",
    ]
    moneyflow_fields = ["ts_code", "trade_date", "net_mf_amount"]
    for trade_date in DATES:
        compact = trade_date.strftime("%Y%m%d")
        (daily_root / f"{compact}.json").write_text(
            json.dumps(
                {
                    "fields": daily_fields,
                    "items": [
                        [symbol, compact, 2.0, 1.1, 10.0, 1.0, 2.0, 0.5, 1000.0, 800.0]
                        for symbol in SYMBOLS
                    ],
                }
            ),
            encoding="utf-8",
        )
        (moneyflow_root / f"{compact}.json").write_text(
            json.dumps(
                {
                    "fields": moneyflow_fields,
                    "items": [[symbol, compact, 12.0] for symbol in SYMBOLS],
                }
            ),
            encoding="utf-8",
        )
    holder_fields = ["ts_code", "ann_date", "end_date", "holder_num"]
    (holder_root / "000001.SZ.json").write_text(
        json.dumps(
            {
                "fields": holder_fields,
                "items": [["000001.SZ", "20230602", "20230531", 10000]],
            }
        ),
        encoding="utf-8",
    )
    (holder_root / "600000.SH.json").write_text(
        json.dumps(
            {
                "fields": holder_fields,
                "items": [
                    ["600000.SH", "20230530", "20230531", 20000],
                    ["600000.SH", "20230602", "20230531", None],
                ],
            }
        ),
        encoding="utf-8",
    )
    return universe, daily_root, moneyflow_root, holder_root


def _prepare(tmp_path: Path, *, generated_at: datetime = GENERATED_AT):
    base_root, dataset_id = _publish_base(tmp_path)
    universe, daily_root, moneyflow_root, holder_root = _write_inputs(tmp_path)
    return prepare_feature_dataset(
        base_publication_root=base_root,
        base_dataset_id=dataset_id,
        universe_path=universe,
        daily_basic_root=daily_root,
        moneyflow_root=moneyflow_root,
        holder_root=holder_root,
        generated_at=generated_at,
    )


def test_feature_schema_descriptor_matches_frozen_file() -> None:
    descriptor = json.loads(
        (ROOT / "contracts/data_schemas/market-auxiliary-features-v1.json").read_text()
    )
    assert descriptor == FEATURE_SCHEMA_DESCRIPTOR
    assert hashlib.sha256(canonical_json_bytes(descriptor)).hexdigest() == (
        FEATURE_SCHEMA_SHA256
    )


def test_prepare_binds_sources_and_uses_real_announcement_date(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    manifest = prepared.manifest
    files = dict(prepared.files)
    holder_rows = json.loads(files[HOLDER_SNAPSHOT_PATH])
    source_index = json.loads(files[SOURCE_INDEX_PATH])

    assert manifest["coverage"] == {
        "daily_bar_rows": 6,
        "daily_feature_rows": 6,
        "daily_basic_present_rows": 6,
        "moneyflow_present_rows": 6,
        "holder_rows": 1,
        "holder_symbols": 1,
        "missing_holder_symbols": 1,
        "quarantined_holder_rows": 2,
        "source_file_count": 6,
    }
    assert holder_rows == [
        {
            "ann_date": "2023-06-02",
            "end_date": "2023-05-31",
            "holder_num": 10000,
            "symbol": "000001.SZ",
        }
    ]
    assert source_index["missing_holder_symbols"] == ["600519.SH"]
    assert source_index["quarantined_holder_rows"] == [
        {
            "ann_date": "2023-05-30",
            "end_date": "2023-05-31",
            "reason": "ANNOUNCEMENT_BEFORE_REPORT_END",
            "symbol": "600000.SH",
        },
        {
            "ann_date": "2023-06-02",
            "end_date": "2023-05-31",
            "reason": "INVALID_HOLDER_NUM",
            "symbol": "600000.SH",
        },
    ]
    validate_prepared_feature_dataset(prepared)


def test_feature_manifest_is_contract_valid(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    schema = json.loads(
        (ROOT / "contracts/schemas/feature-dataset-manifest.schema.json").read_text()
    )
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(
        prepared.manifest
    )


def test_generated_at_does_not_change_feature_content_identity(tmp_path: Path) -> None:
    first = _prepare(tmp_path / "first")
    second = _prepare(
        tmp_path / "second",
        generated_at=datetime(2026, 8, 18, 15, 0, tzinfo=UTC),
    )

    assert first.feature_dataset_id == second.feature_dataset_id
    assert first.files == second.files
    assert first.manifest["generated_at"] != second.manifest["generated_at"]


def test_feature_publication_is_immutable(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path / "inputs")
    published = publish_feature_dataset(
        publication_root=tmp_path / "features",
        prepared=prepared,
    )

    assert published.manifest_path.read_bytes() == prepared.manifest_bytes
    assert {path.name for path in published.dataset_dir.iterdir()} == {
        name for name, _ in prepared.files
    }
    with pytest.raises(FileExistsError, match="already exists"):
        publish_feature_dataset(
            publication_root=tmp_path / "features",
            prepared=prepared,
        )
