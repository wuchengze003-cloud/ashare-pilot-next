"""Import vendor exports into an immutable point-in-time feature dataset."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .dataset_publication import (
    _ensure_directory,
    _fsync_directory,
    _fsync_tree_directories,
    _write_new_fsynced,
    load_published_dataset,
)
from .normalization import NormalizedDailyBar, canonical_json_bytes

CONTRACT_ID = "feature-dataset-manifest"
SCHEMA_VERSION = "1.0.0"
FEATURE_SCHEMA_ID = "market-auxiliary-features/v1"
NORMALIZATION_VERSION = "tushare-market-auxiliary-normalizer/v1"
SOURCE = "tushare-export-import"
SOURCE_VERSION = "v1"
SOURCE_INDEX_FORMAT = "ashare-pilot-feature-source-index/v1"
DAILY_FEATURE_PATH = "daily-features.json"
HOLDER_SNAPSHOT_PATH = "holder-snapshots.json"
SOURCE_INDEX_PATH = "source-index.json"

FEATURE_SCHEMA_DESCRIPTOR: dict[str, object] = {
    "schema_id": FEATURE_SCHEMA_ID,
    "daily_feature_primary_key": ["symbol", "trade_date"],
    "holder_snapshot_primary_key": ["symbol", "ann_date", "end_date"],
    "daily_feature_fields": [
        ["symbol", "string"],
        ["trade_date", "date"],
        ["turnover_rate_f", "nullable-percent"],
        ["volume_ratio", "nullable-ratio"],
        ["pe_ttm", "nullable-ratio"],
        ["pb", "nullable-ratio"],
        ["ps_ttm", "nullable-ratio"],
        ["dv_ttm", "nullable-percent"],
        ["total_mv", "nullable-10k-cny"],
        ["circ_mv", "nullable-10k-cny"],
        ["net_mf_amount", "nullable-10k-cny"],
    ],
    "holder_snapshot_fields": [
        ["symbol", "string"],
        ["ann_date", "date"],
        ["end_date", "date"],
        ["holder_num", "positive-integer"],
    ],
    "point_in_time_rule": (
        "holder snapshot becomes visible on ann_date; end_date is never used as "
        "availability time"
    ),
}
FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    canonical_json_bytes(FEATURE_SCHEMA_DESCRIPTOR)
).hexdigest()

DAILY_BASIC_FIELDS: tuple[str, ...] = (
    "turnover_rate_f",
    "volume_ratio",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "total_mv",
    "circ_mv",
)
MONEYFLOW_FIELDS: tuple[str, ...] = ("net_mf_amount",)


@dataclass(frozen=True)
class PreparedFeatureDataset:
    feature_dataset_id: str
    files: tuple[tuple[str, bytes], ...]
    manifest_bytes: bytes

    @property
    def manifest(self) -> dict[str, Any]:
        document = json.loads(self.manifest_bytes)
        if not isinstance(document, dict):
            raise ValueError("feature manifest must be an object")
        return document


@dataclass(frozen=True)
class PublishedFeatureDataset:
    feature_dataset_id: str
    dataset_dir: Path
    manifest_path: Path


def _regular_file_bytes(path: Path, *, field: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} must be a regular file: {path}")
    return path.read_bytes()


def _regular_directory(path: Path, *, field: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{field} must be a regular directory")
    return path


def _parse_yyyymmdd(value: object, *, field: str) -> date:
    text = str(value)
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"{field} must use YYYYMMDD")
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid date") from exc


def _finite_or_none(value: object, *, field: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} cannot be boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric or null") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _source_entry(*, namespace: str, path: Path, content: bytes) -> dict[str, object]:
    return {
        "path": f"{namespace}/{path.name}",
        "sha256": hashlib.sha256(content).hexdigest(),
        "file_size_bytes": len(content),
    }


def _load_daily_export(
    *,
    path: Path,
    namespace: str,
    expected_date: date,
    expected_symbols: frozenset[str],
    value_fields: Sequence[str],
) -> tuple[dict[str, dict[str, float | None]], dict[str, object]]:
    content = _regular_file_bytes(path, field=namespace)
    try:
        document = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{namespace} export is not valid JSON: {path.name}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{namespace} export must be an object")
    fields = document.get("fields")
    items = document.get("items")
    required = {"ts_code", "trade_date", *value_fields}
    if not isinstance(fields, list) or not required <= set(fields):
        raise ValueError(f"{namespace} export lacks required fields: {path.name}")
    if not isinstance(items, list):
        raise ValueError(f"{namespace} export items must be a list")
    indexes = {name: fields.index(name) for name in required}
    expected_text = expected_date.strftime("%Y%m%d")
    result: dict[str, dict[str, float | None]] = {}
    seen_all: set[tuple[str, str]] = set()
    for row_number, row in enumerate(items):
        if not isinstance(row, list) or len(row) != len(fields):
            raise ValueError(f"{namespace} row shape mismatch at {path.name}:{row_number}")
        symbol = str(row[indexes["ts_code"]])
        trade_date = str(row[indexes["trade_date"]])
        if trade_date != expected_text:
            raise ValueError(f"{namespace} row date disagrees with file: {path.name}")
        key = (symbol, trade_date)
        if key in seen_all:
            raise ValueError(f"duplicate {namespace} row: {symbol} {trade_date}")
        seen_all.add(key)
        if symbol not in expected_symbols:
            continue
        result[symbol] = {
            name: _finite_or_none(row[indexes[name]], field=f"{namespace}.{name}")
            for name in value_fields
        }
    return result, _source_entry(namespace=namespace, path=path, content=content)


def _load_universe(path: Path) -> tuple[tuple[str, ...], str]:
    content = _regular_file_bytes(path, field="universe")
    document = json.loads(content)
    if not isinstance(document, dict) or document.get("contract_id") != "universe":
        raise ValueError("universe document is invalid")
    members = document.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("universe members are unavailable")
    symbols = tuple(str(member["symbol"]) for member in members)
    if len(symbols) != len(set(symbols)):
        raise ValueError("universe contains duplicate symbols")
    return symbols, hashlib.sha256(content).hexdigest()


def _normalize_daily_features(
    *,
    records: Sequence[NormalizedDailyBar],
    symbols: frozenset[str],
    daily_basic_root: Path,
    moneyflow_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, int]:
    records_by_date: dict[date, list[NormalizedDailyBar]] = {}
    for record in records:
        records_by_date.setdefault(record.trade_date, []).append(record)
    normalized: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    daily_basic_present = 0
    moneyflow_present = 0
    for trade_date in sorted(records_by_date):
        filename = f"{trade_date.strftime('%Y%m%d')}.json"
        daily_basic, daily_source = _load_daily_export(
            path=daily_basic_root / filename,
            namespace="daily_basic",
            expected_date=trade_date,
            expected_symbols=symbols,
            value_fields=DAILY_BASIC_FIELDS,
        )
        moneyflow, moneyflow_source = _load_daily_export(
            path=moneyflow_root / filename,
            namespace="moneyflow",
            expected_date=trade_date,
            expected_symbols=symbols,
            value_fields=MONEYFLOW_FIELDS,
        )
        sources.extend((daily_source, moneyflow_source))
        for record in sorted(records_by_date[trade_date], key=lambda item: item.symbol):
            daily_values = daily_basic.get(record.symbol)
            moneyflow_values = moneyflow.get(record.symbol)
            daily_basic_present += daily_values is not None
            moneyflow_present += moneyflow_values is not None
            normalized.append(
                {
                    "symbol": record.symbol,
                    "trade_date": trade_date.isoformat(),
                    **{
                        name: daily_values.get(name) if daily_values is not None else None
                        for name in DAILY_BASIC_FIELDS
                    },
                    "net_mf_amount": (
                        moneyflow_values.get("net_mf_amount")
                        if moneyflow_values is not None
                        else None
                    ),
                }
            )
    return normalized, sources, daily_basic_present, moneyflow_present


def _normalize_holder_snapshots(
    *,
    symbols: tuple[str, ...],
    holder_root: Path,
    as_of: date,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[str],
    list[dict[str, object]],
]:
    normalized: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    missing: list[str] = []
    quarantined: list[dict[str, object]] = []
    seen: set[tuple[str, date, date]] = set()
    for symbol in symbols:
        path = holder_root / f"{symbol}.json"
        if not path.exists():
            missing.append(symbol)
            continue
        content = _regular_file_bytes(path, field="stk_holdernumber")
        sources.append(_source_entry(namespace="stk_holdernumber", path=path, content=content))
        document = json.loads(content)
        if not isinstance(document, dict):
            raise ValueError(f"holder export must be an object: {path.name}")
        fields = document.get("fields")
        items = document.get("items")
        required = {"ts_code", "ann_date", "end_date", "holder_num"}
        if not isinstance(fields, list) or not required <= set(fields):
            raise ValueError(f"holder export lacks required fields: {path.name}")
        if not isinstance(items, list):
            raise ValueError(f"holder export items must be a list: {path.name}")
        indexes = {name: fields.index(name) for name in required}
        for row_number, row in enumerate(items):
            if not isinstance(row, list) or len(row) != len(fields):
                raise ValueError(f"holder row shape mismatch at {path.name}:{row_number}")
            row_symbol = str(row[indexes["ts_code"]])
            if row_symbol != symbol:
                raise ValueError(f"holder row symbol mismatch: {path.name}")
            ann_date = _parse_yyyymmdd(row[indexes["ann_date"]], field="ann_date")
            end_date = _parse_yyyymmdd(row[indexes["end_date"]], field="end_date")
            if ann_date > as_of:
                continue
            if ann_date < end_date:
                quarantined.append(
                    {
                        "symbol": symbol,
                        "ann_date": ann_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "reason": "ANNOUNCEMENT_BEFORE_REPORT_END",
                    }
                )
                continue
            raw_holder_num = row[indexes["holder_num"]]
            try:
                numeric_holder_num = float(raw_holder_num)
            except (TypeError, ValueError):
                numeric_holder_num = math.nan
            if (
                isinstance(raw_holder_num, bool)
                or not math.isfinite(numeric_holder_num)
                or numeric_holder_num <= 0
                or not numeric_holder_num.is_integer()
            ):
                quarantined.append(
                    {
                        "symbol": symbol,
                        "ann_date": ann_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "reason": "INVALID_HOLDER_NUM",
                    }
                )
                continue
            holder_num = int(numeric_holder_num)
            key = (symbol, ann_date, end_date)
            if key in seen:
                raise ValueError(f"duplicate holder snapshot: {symbol} {ann_date} {end_date}")
            seen.add(key)
            normalized.append(
                {
                    "symbol": symbol,
                    "ann_date": ann_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "holder_num": holder_num,
                }
            )
    normalized.sort(key=lambda row: (row["symbol"], row["ann_date"], row["end_date"]))
    return normalized, sources, missing, quarantined


def _file_entry(
    *,
    role: str,
    path: str,
    content: bytes,
    row_count: int,
    min_effective_date: str | None,
    max_effective_date: str | None,
) -> dict[str, object]:
    return {
        "role": role,
        "path": path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "row_count": row_count,
        "file_size_bytes": len(content),
        "min_effective_date": min_effective_date,
        "max_effective_date": max_effective_date,
    }


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


def prepare_feature_dataset(
    *,
    base_publication_root: Path,
    base_dataset_id: str,
    universe_path: Path,
    daily_basic_root: Path,
    moneyflow_root: Path,
    holder_root: Path,
    generated_at: datetime,
) -> PreparedFeatureDataset:
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    loaded = load_published_dataset(
        publication_root=base_publication_root,
        dataset_id=base_dataset_id,
    )
    symbols, universe_sha256 = _load_universe(universe_path)
    record_symbols = frozenset(record.symbol for record in loaded.records)
    if set(symbols) != set(record_symbols):
        raise ValueError("feature universe does not match the base dataset")
    daily_basic_root = _regular_directory(daily_basic_root, field="daily_basic_root")
    moneyflow_root = _regular_directory(moneyflow_root, field="moneyflow_root")
    holder_root = _regular_directory(holder_root, field="holder_root")
    daily_rows, daily_sources, daily_present, moneyflow_present = _normalize_daily_features(
        records=loaded.records,
        symbols=record_symbols,
        daily_basic_root=daily_basic_root,
        moneyflow_root=moneyflow_root,
    )
    holder_rows, holder_sources, missing_holders, quarantined = (
        _normalize_holder_snapshots(
            symbols=symbols,
            holder_root=holder_root,
            as_of=date.fromisoformat(str(loaded.manifest["as_of"])),
        )
    )
    if len(daily_rows) != len(loaded.records):
        raise ValueError("daily feature rows do not match base daily bars")
    if daily_present / len(daily_rows) < 0.95:
        raise ValueError("daily_basic coverage is below 95 percent")
    if moneyflow_present / len(daily_rows) < 0.95:
        raise ValueError("moneyflow coverage is below 95 percent")

    all_sources = sorted(
        [*daily_sources, *holder_sources], key=lambda item: str(item["path"])
    )
    if len({str(item["path"]) for item in all_sources}) != len(all_sources):
        raise ValueError("source index contains duplicate paths")
    source_index = {
        "format": SOURCE_INDEX_FORMAT,
        "base_dataset_id": base_dataset_id,
        "universe_sha256": universe_sha256,
        "missing_holder_symbols": sorted(missing_holders),
        "quarantined_holder_rows": sorted(
            quarantined,
            key=lambda item: (item["symbol"], item["ann_date"], item["end_date"]),
        ),
        "files": all_sources,
    }
    daily_bytes = canonical_json_bytes(daily_rows)
    holder_bytes = canonical_json_bytes(holder_rows)
    source_bytes = canonical_json_bytes(source_index)
    holder_dates = [str(row["ann_date"]) for row in holder_rows]
    files = [
        _file_entry(
            role="daily_features",
            path=DAILY_FEATURE_PATH,
            content=daily_bytes,
            row_count=len(daily_rows),
            min_effective_date=str(daily_rows[0]["trade_date"]),
            max_effective_date=str(daily_rows[-1]["trade_date"]),
        ),
        _file_entry(
            role="holder_snapshots",
            path=HOLDER_SNAPSHOT_PATH,
            content=holder_bytes,
            row_count=len(holder_rows),
            min_effective_date=min(holder_dates) if holder_dates else None,
            max_effective_date=max(holder_dates) if holder_dates else None,
        ),
        _file_entry(
            role="source_index",
            path=SOURCE_INDEX_PATH,
            content=source_bytes,
            row_count=len(all_sources),
            min_effective_date=None,
            max_effective_date=None,
        ),
    ]
    reasons: list[str] = []
    if missing_holders:
        reasons.append("OPTIONAL_HOLDER_COVERAGE_INCOMPLETE")
    if quarantined:
        reasons.append("INVALID_HOLDER_ROWS_QUARANTINED")
    manifest: dict[str, object] = {
        "contract_id": CONTRACT_ID,
        "schema_version": SCHEMA_VERSION,
        "feature_dataset_id": "",
        "base_dataset_id": base_dataset_id,
        "base_manifest_sha256": hashlib.sha256(loaded.manifest_bytes).hexdigest(),
        "universe_sha256": universe_sha256,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "normalization_version": NORMALIZATION_VERSION,
        "as_of": str(loaded.manifest["as_of"]),
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "source": SOURCE,
        "source_version": SOURCE_VERSION,
        "quality_status": "pass",
        "quality_reasons": reasons,
        "coverage": {
            "daily_bar_rows": len(loaded.records),
            "daily_feature_rows": len(daily_rows),
            "daily_basic_present_rows": daily_present,
            "moneyflow_present_rows": moneyflow_present,
            "holder_rows": len(holder_rows),
            "holder_symbols": len({str(row["symbol"]) for row in holder_rows}),
            "missing_holder_symbols": len(missing_holders),
            "quarantined_holder_rows": len(quarantined),
            "source_file_count": len(all_sources),
        },
        "files": files,
    }
    manifest["feature_dataset_id"] = _derive_id(manifest)
    prepared = PreparedFeatureDataset(
        feature_dataset_id=str(manifest["feature_dataset_id"]),
        files=(
            (DAILY_FEATURE_PATH, daily_bytes),
            (HOLDER_SNAPSHOT_PATH, holder_bytes),
            (SOURCE_INDEX_PATH, source_bytes),
        ),
        manifest_bytes=canonical_json_bytes(manifest),
    )
    validate_prepared_feature_dataset(prepared)
    return prepared


def validate_prepared_feature_dataset(prepared: PreparedFeatureDataset) -> None:
    if not isinstance(prepared, PreparedFeatureDataset):
        raise TypeError("prepared must be PreparedFeatureDataset")
    manifest = prepared.manifest
    if canonical_json_bytes(manifest) != prepared.manifest_bytes:
        raise ValueError("feature manifest bytes are not canonical")
    if manifest.get("contract_id") != CONTRACT_ID:
        raise ValueError("feature manifest contract_id is invalid")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("feature manifest schema_version is invalid")
    if manifest.get("feature_dataset_id") != prepared.feature_dataset_id:
        raise ValueError("prepared feature dataset identity is inconsistent")
    if _derive_id(manifest) != prepared.feature_dataset_id:
        raise ValueError("feature dataset content identity is invalid")
    if manifest.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
        raise ValueError("feature schema hash is invalid")
    file_map = dict(prepared.files)
    if len(file_map) != 3 or set(file_map) != {
        DAILY_FEATURE_PATH,
        HOLDER_SNAPSHOT_PATH,
        SOURCE_INDEX_PATH,
    }:
        raise ValueError("prepared feature files are incomplete")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("feature manifest files are invalid")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("feature manifest file entry must be an object")
        content = file_map.get(str(entry.get("path")))
        if content is None:
            raise ValueError("feature manifest references an unknown file")
        if hashlib.sha256(content).hexdigest() != entry.get("sha256"):
            raise ValueError("feature file hash mismatch")
        if len(content) != entry.get("file_size_bytes"):
            raise ValueError("feature file size mismatch")
        rows = json.loads(content)
        if str(entry["path"]) == SOURCE_INDEX_PATH:
            if not isinstance(rows, dict) or not isinstance(rows.get("files"), list):
                raise ValueError("feature source index is invalid")
            actual_count = len(rows["files"])
        else:
            if not isinstance(rows, list):
                raise ValueError("feature data file must contain an array")
            actual_count = len(rows)
        if actual_count != entry.get("row_count"):
            raise ValueError("feature file row count mismatch")


def _validate_published_dir(path: Path, prepared: PreparedFeatureDataset) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("published feature dataset directory is invalid")
    entries = list(path.iterdir())
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise ValueError("published feature dataset may contain only regular files")
    if {item.name for item in entries} != {name for name, _ in prepared.files}:
        raise ValueError("published feature dataset files are incomplete")
    for name, content in prepared.files:
        if (path / name).read_bytes() != content:
            raise ValueError(f"published feature file changed: {name}")


def publish_feature_dataset(
    *,
    publication_root: Path,
    prepared: PreparedFeatureDataset,
) -> PublishedFeatureDataset:
    validate_prepared_feature_dataset(prepared)
    publication_root = Path(publication_root)
    _ensure_directory(publication_root)
    datasets_root = publication_root / "datasets"
    manifests_root = publication_root / "manifests"
    _ensure_directory(datasets_root)
    _ensure_directory(manifests_root)
    dataset_dir = datasets_root / prepared.feature_dataset_id
    manifest_path = manifests_root / f"{prepared.feature_dataset_id}.json"
    lock_path = publication_root / ".feature-publication.lock"
    if lock_path.is_symlink():
        raise ValueError("feature publication lock cannot be a symlink")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if dataset_dir.exists() or dataset_dir.is_symlink():
            raise FileExistsError(f"feature dataset already exists: {prepared.feature_dataset_id}")
        if manifest_path.exists() or manifest_path.is_symlink():
            raise FileExistsError(
                f"feature dataset manifest already exists: {prepared.feature_dataset_id}"
            )
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{prepared.feature_dataset_id}.tmp-", dir=datasets_root)
        )
        temporary_manifest: Path | None = None
        dataset_moved = False
        manifest_published = False
        try:
            for name, content in prepared.files:
                _write_new_fsynced(temporary_dir / name, content)
            _fsync_tree_directories(temporary_dir)
            _validate_published_dir(temporary_dir, prepared)
            os.replace(temporary_dir, dataset_dir)
            dataset_moved = True
            _fsync_directory(datasets_root)
            _validate_published_dir(dataset_dir, prepared)

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{prepared.feature_dataset_id}.manifest-",
                suffix=".tmp",
                dir=manifests_root,
            )
            temporary_manifest = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(prepared.manifest_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            if temporary_manifest.read_bytes() != prepared.manifest_bytes:
                raise ValueError("staged feature manifest changed")
            os.replace(temporary_manifest, manifest_path)
            temporary_manifest = None
            manifest_published = True
            _fsync_directory(manifests_root)
        except BaseException:
            if temporary_manifest is not None:
                temporary_manifest.unlink(missing_ok=True)
            if not dataset_moved:
                shutil.rmtree(temporary_dir, ignore_errors=True)
            elif not manifest_published:
                shutil.rmtree(dataset_dir, ignore_errors=True)
                with suppress(OSError):
                    _fsync_directory(datasets_root)
            raise
    return PublishedFeatureDataset(
        feature_dataset_id=prepared.feature_dataset_id,
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish immutable market feature data")
    parser.add_argument("--base-publication-root", required=True, type=Path)
    parser.add_argument("--base-dataset-id", required=True)
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--daily-basic-root", required=True, type=Path)
    parser.add_argument("--moneyflow-root", required=True, type=Path)
    parser.add_argument("--holder-root", required=True, type=Path)
    parser.add_argument("--publication-root", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args(argv)
    generated_at = datetime.fromisoformat(args.generated_at.replace("Z", "+00:00"))
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated-at must be timezone-aware")
    prepared = prepare_feature_dataset(
        base_publication_root=args.base_publication_root,
        base_dataset_id=args.base_dataset_id,
        universe_path=args.universe,
        daily_basic_root=args.daily_basic_root,
        moneyflow_root=args.moneyflow_root,
        holder_root=args.holder_root,
        generated_at=generated_at,
    )
    published = publish_feature_dataset(
        publication_root=args.publication_root,
        prepared=prepared,
    )
    print(
        canonical_json_bytes(
            {
                "feature_dataset_id": published.feature_dataset_id,
                "dataset_dir": str(published.dataset_dir),
                "manifest_path": str(published.manifest_path),
                "coverage": prepared.manifest["coverage"],
                "quality_reasons": prepared.manifest["quality_reasons"],
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
