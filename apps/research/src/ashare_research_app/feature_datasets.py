"""Verified consumer for immutable market auxiliary feature datasets."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .canonical import canonical_json_bytes

FEATURE_SCHEMA_ID = "market-auxiliary-features/v1"
FEATURE_SCHEMA_SHA256 = "e4df2c76bb941f4a46bda29f9763204768d919a33171a4fe87149da7ef8f6ae7"
NORMALIZATION_VERSION = "tushare-market-auxiliary-normalizer/v1"
SOURCE_INDEX_FORMAT = "ashare-pilot-feature-source-index/v1"
DAILY_AUXILIARY_FEATURE_NAMES: tuple[str, ...] = (
    "turnover_rate_f",
    "volume_ratio",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "total_mv",
    "circ_mv",
    "net_mf_amount",
)
HOLDER_FEATURE_NAMES: tuple[str, ...] = ("holder_log", "holder_change")


@dataclass(frozen=True, order=True)
class HolderSnapshot:
    symbol: str
    ann_date: date
    end_date: date
    holder_num: int

    def __post_init__(self) -> None:
        if self.ann_date < self.end_date:
            raise ValueError("holder snapshot cannot be visible before its report end")
        if self.holder_num <= 0:
            raise ValueError("holder_num must be positive")


@dataclass(frozen=True)
class FeatureDataset:
    feature_dataset_id: str
    base_dataset_id: str
    base_manifest_sha256: str
    universe_sha256: str
    manifest_sha256: str
    as_of: date
    daily_values_by_key: Mapping[tuple[str, date], tuple[float | None, ...]]
    holder_by_symbol: Mapping[str, tuple[HolderSnapshot, ...]]

    def assert_matches_snapshot(self, snapshot: object) -> None:
        """Bind the feature rows to the exact immutable daily-bar snapshot."""
        snapshot_dataset_id = getattr(snapshot, "dataset_id", None)
        snapshot_manifest_sha256 = getattr(snapshot, "manifest_sha256", None)
        snapshot_as_of = getattr(snapshot, "as_of", None)
        records = getattr(snapshot, "records", None)
        if snapshot_dataset_id != self.base_dataset_id:
            raise ValueError("feature dataset is bound to a different base dataset")
        if snapshot_manifest_sha256 != self.base_manifest_sha256:
            raise ValueError("feature dataset base manifest hash mismatch")
        if not isinstance(snapshot_as_of, date) or snapshot_as_of > self.as_of:
            raise ValueError("feature dataset does not cover the snapshot as_of")
        if not isinstance(records, tuple):
            raise TypeError("snapshot records must be an immutable tuple")
        for record in records:
            if record.trade_date > snapshot_as_of:
                raise ValueError("snapshot contains a future daily bar")
            if (record.symbol, record.trade_date) not in self.daily_values_by_key:
                raise ValueError(
                    f"feature dataset lacks base bar: {record.symbol} {record.trade_date}"
                )

    def daily_values(
        self,
        *,
        symbol: str,
        trade_date: date,
    ) -> tuple[float | None, ...]:
        try:
            return self.daily_values_by_key[(symbol, trade_date)]
        except KeyError as exc:
            raise ValueError(f"feature dataset lacks base bar: {symbol} {trade_date}") from exc

    def holder_values(
        self,
        *,
        symbol: str,
        through: date,
    ) -> tuple[float | None, float | None]:
        snapshots = self.holder_by_symbol.get(symbol, ())
        position = bisect_right(
            snapshots,
            through,
            key=lambda snapshot: snapshot.ann_date,
        ) - 1
        if position < 0:
            return None, None
        current = snapshots[position]
        holder_log = math.log(float(current.holder_num))
        if position == 0:
            return holder_log, None
        previous = snapshots[position - 1]
        return holder_log, current.holder_num / previous.holder_num - 1.0


def _manifest_identity(document: Mapping[str, Any]) -> dict[str, object]:
    return {
        "base_dataset_id": document["base_dataset_id"],
        "base_manifest_sha256": document["base_manifest_sha256"],
        "universe_sha256": document["universe_sha256"],
        "feature_schema_id": document["feature_schema_id"],
        "feature_schema_sha256": document["feature_schema_sha256"],
        "normalization_version": document["normalization_version"],
        "as_of": document["as_of"],
        "source": document["source"],
        "source_version": document["source_version"],
        "files": document["files"],
    }


def _derive_id(document: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(_manifest_identity(document))).hexdigest()
    return f"feature-dataset-sha256-{digest}"


def _load_regular(path: Path, *, field: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} must be a regular file")
    return path.read_bytes()


def _optional_number(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric or null")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def load_feature_dataset(
    *,
    manifest_path: Path,
    dataset_root: Path,
    expected_base_dataset_id: str,
    expected_base_manifest_sha256: str,
    expected_universe_sha256: str,
    as_of: date,
) -> FeatureDataset:
    manifest_bytes = _load_regular(Path(manifest_path), field="feature manifest")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("feature manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("feature manifest must be an object")
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise ValueError("feature manifest bytes are not canonical")
    if manifest.get("contract_id") != "feature-dataset-manifest":
        raise ValueError("feature manifest contract_id is invalid")
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError("feature manifest schema_version is invalid")
    feature_dataset_id = str(manifest.get("feature_dataset_id"))
    if feature_dataset_id != _derive_id(manifest):
        raise ValueError("feature dataset identity is invalid")
    if manifest.get("base_dataset_id") != expected_base_dataset_id:
        raise ValueError("feature dataset is bound to a different base dataset")
    if manifest.get("base_manifest_sha256") != expected_base_manifest_sha256:
        raise ValueError("feature dataset base manifest hash mismatch")
    if manifest.get("universe_sha256") != expected_universe_sha256:
        raise ValueError("feature dataset universe hash mismatch")
    if manifest.get("feature_schema_id") != FEATURE_SCHEMA_ID:
        raise ValueError("feature schema id is invalid")
    if manifest.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
        raise ValueError("feature schema hash is invalid")
    if manifest.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError("feature normalization version is invalid")
    if manifest.get("quality_status") != "pass":
        raise ValueError("feature dataset quality did not pass")
    manifest_as_of = date.fromisoformat(str(manifest.get("as_of")))
    if as_of > manifest_as_of:
        raise ValueError("feature dataset does not cover the requested as_of")

    dataset_root = Path(dataset_root)
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ValueError("feature dataset root is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("feature manifest files are invalid")
    by_role: dict[str, tuple[dict[str, Any], bytes]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("feature file entry must be an object")
        role = str(entry.get("role"))
        relative = Path(str(entry.get("path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("feature file path is unsafe")
        path = dataset_root / relative
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(dataset_root.resolve(strict=True)):
            raise ValueError("feature file escapes dataset root")
        content = _load_regular(path, field=f"feature file {role}")
        if hashlib.sha256(content).hexdigest() != entry.get("sha256"):
            raise ValueError(f"feature file hash mismatch: {role}")
        if len(content) != entry.get("file_size_bytes"):
            raise ValueError(f"feature file size mismatch: {role}")
        if role in by_role:
            raise ValueError(f"duplicate feature file role: {role}")
        by_role[role] = (entry, content)
    if set(by_role) != {"daily_features", "holder_snapshots", "source_index"}:
        raise ValueError("feature file roles are incomplete")

    daily_entry, daily_bytes = by_role["daily_features"]
    daily_document = json.loads(daily_bytes)
    if not isinstance(daily_document, list):
        raise ValueError("daily feature file must contain an array")
    if len(daily_document) != daily_entry.get("row_count"):
        raise ValueError("daily feature row count mismatch")
    expected_daily_fields = {"symbol", "trade_date", *DAILY_AUXILIARY_FEATURE_NAMES}
    daily_values: dict[tuple[str, date], tuple[float | None, ...]] = {}
    previous_daily_key: tuple[date, str] | None = None
    for row in daily_document:
        if not isinstance(row, dict) or set(row) != expected_daily_fields:
            raise ValueError("daily feature row fields are invalid")
        symbol = str(row["symbol"])
        trade_date = date.fromisoformat(str(row["trade_date"]))
        if trade_date > manifest_as_of:
            raise ValueError("daily feature row exceeds manifest as_of")
        key = (symbol, trade_date)
        order_key = (trade_date, symbol)
        if previous_daily_key is not None and order_key <= previous_daily_key:
            raise ValueError("daily feature rows are not unique and ordered")
        previous_daily_key = order_key
        daily_values[key] = tuple(
            _optional_number(row[name], field=name)
            for name in DAILY_AUXILIARY_FEATURE_NAMES
        )

    holder_entry, holder_bytes = by_role["holder_snapshots"]
    holder_document = json.loads(holder_bytes)
    if not isinstance(holder_document, list):
        raise ValueError("holder snapshot file must contain an array")
    if len(holder_document) != holder_entry.get("row_count"):
        raise ValueError("holder snapshot row count mismatch")
    holders: dict[str, list[HolderSnapshot]] = {}
    previous_holder_key: tuple[str, date, date] | None = None
    for row in holder_document:
        if not isinstance(row, dict) or set(row) != {
            "symbol",
            "ann_date",
            "end_date",
            "holder_num",
        }:
            raise ValueError("holder snapshot fields are invalid")
        if isinstance(row["holder_num"], bool) or not isinstance(row["holder_num"], int):
            raise ValueError("holder_num must be an integer")
        snapshot = HolderSnapshot(
            symbol=str(row["symbol"]),
            ann_date=date.fromisoformat(str(row["ann_date"])),
            end_date=date.fromisoformat(str(row["end_date"])),
            holder_num=int(row["holder_num"]),
        )
        if snapshot.ann_date > manifest_as_of:
            raise ValueError("holder snapshot exceeds manifest as_of")
        holder_key = (snapshot.symbol, snapshot.ann_date, snapshot.end_date)
        if previous_holder_key is not None and holder_key <= previous_holder_key:
            raise ValueError("holder snapshots are not unique and ordered")
        previous_holder_key = holder_key
        holders.setdefault(snapshot.symbol, []).append(snapshot)

    source_entry, source_bytes = by_role["source_index"]
    source_index = json.loads(source_bytes)
    if not isinstance(source_index, dict):
        raise ValueError("feature source index must be an object")
    if source_index.get("format") != SOURCE_INDEX_FORMAT:
        raise ValueError("feature source index format is invalid")
    if source_index.get("base_dataset_id") != expected_base_dataset_id:
        raise ValueError("feature source index base dataset mismatch")
    if source_index.get("universe_sha256") != expected_universe_sha256:
        raise ValueError("feature source index universe hash mismatch")
    source_files = source_index.get("files")
    if not isinstance(source_files, list) or len(source_files) != source_entry.get("row_count"):
        raise ValueError("feature source index row count mismatch")
    previous_source_path: str | None = None
    for source_file in source_files:
        if not isinstance(source_file, dict) or set(source_file) != {
            "path",
            "sha256",
            "file_size_bytes",
        }:
            raise ValueError("feature source entry is invalid")
        source_path = str(source_file["path"])
        relative_source = Path(source_path)
        if relative_source.is_absolute() or ".." in relative_source.parts:
            raise ValueError("feature source path is unsafe")
        if previous_source_path is not None and source_path <= previous_source_path:
            raise ValueError("feature source entries are not unique and ordered")
        previous_source_path = source_path
        digest = str(source_file["sha256"])
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("feature source hash is invalid")
        if isinstance(source_file["file_size_bytes"], bool) or not isinstance(
            source_file["file_size_bytes"], int
        ) or source_file["file_size_bytes"] < 1:
            raise ValueError("feature source size is invalid")
    coverage = manifest.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("feature coverage is invalid")
    if len(daily_values) != coverage.get("daily_feature_rows"):
        raise ValueError("daily feature coverage does not reconcile")
    if len(daily_values) != coverage.get("daily_bar_rows"):
        raise ValueError("daily features do not cover every base bar")
    if sum(len(values) for values in holders.values()) != coverage.get("holder_rows"):
        raise ValueError("holder coverage does not reconcile")
    if len(source_index.get("missing_holder_symbols", [])) != coverage.get(
        "missing_holder_symbols"
    ):
        raise ValueError("missing holder coverage does not reconcile")
    if len(source_index.get("quarantined_holder_rows", [])) != coverage.get(
        "quarantined_holder_rows"
    ):
        raise ValueError("quarantined holder coverage does not reconcile")

    return FeatureDataset(
        feature_dataset_id=feature_dataset_id,
        base_dataset_id=expected_base_dataset_id,
        base_manifest_sha256=expected_base_manifest_sha256,
        universe_sha256=expected_universe_sha256,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        as_of=manifest_as_of,
        daily_values_by_key=MappingProxyType(daily_values),
        holder_by_symbol=MappingProxyType(
            {symbol: tuple(values) for symbol, values in holders.items()}
        ),
    )
