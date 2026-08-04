"""Load immutable datasets into point-in-time snapshots for research.

Research is a documented consumer of Dataset Manifest 2.0. This loader
verifies content hashes and row counts before constructing a
quant_core DatasetSnapshot; it never touches provider caches.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from ashare_quant_core import (
    DAILY_BAR_SCHEMA_ID,
    DAILY_BAR_SCHEMA_SHA256,
    DailyBar,
    DatasetSnapshot,
)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"dataset manifest is unavailable: {manifest_path}")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("dataset manifest must be a JSON object")
    if document.get("contract_id") != "dataset-manifest":
        raise ValueError("document is not a dataset manifest")
    return document


def load_snapshot(
    *,
    manifest: Mapping[str, Any],
    dataset_root: Path,
    as_of: date,
) -> DatasetSnapshot:
    """Read the manifest-committed data file once and verify before use."""
    dataset_root = Path(dataset_root)
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ValueError("dataset root is invalid")
    files = manifest["files"]
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError("research pilot expects exactly one manifest file")
    artifact = files[0]
    if not isinstance(artifact, Mapping):
        raise ValueError("manifest file entry must be an object")
    relative_path = str(artifact["path"])
    data_path = dataset_root / relative_path
    resolved = data_path.resolve(strict=True)
    if not resolved.is_relative_to(dataset_root.resolve(strict=True)):
        raise ValueError("dataset file escapes dataset root")
    if data_path.is_symlink():
        raise ValueError("dataset file cannot be a symlink")

    content = data_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
        raise ValueError("dataset artifact hash mismatch")
    rows = json.loads(content)
    if not isinstance(rows, list):
        raise ValueError("dataset artifact must contain a JSON array")
    if len(rows) != int(artifact["row_count"]):
        raise ValueError("dataset row count mismatch")

    records: list[DailyBar] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("dataset row must be an object")
        records.append(
            DailyBar(
                symbol=str(row["symbol"]),
                trade_date=date.fromisoformat(str(row["trade_date"])),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                amount=float(row["amount"]),
            )
        )
    return DatasetSnapshot(
        dataset_id=str(manifest["dataset_id"]),
        dataset_family_id=str(manifest["dataset_family_id"]),
        manifest_sha256=hashlib.sha256(
            json.dumps(manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        as_of=as_of,
        data_schema_id=str(manifest.get("data_schema_id", DAILY_BAR_SCHEMA_ID)),
        data_schema_sha256=str(
            manifest.get("data_schema_sha256", DAILY_BAR_SCHEMA_SHA256)
        ),
        normalization_version=str(manifest["normalization_version"]),
        records=tuple(records),
    )
