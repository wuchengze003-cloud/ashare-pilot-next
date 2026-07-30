"""Build immutable snapshots from the exact bytes covered by input manifests."""

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
    UniverseMember,
    UniverseSnapshot,
)

from .runner import canonical_json_sha256


def _read_manifest_bytes(
    *,
    dataset_root: Path,
    relative_path: Path,
) -> bytes:
    resolved_root = dataset_root.resolve(strict=True)
    candidate = dataset_root / relative_path
    if candidate.is_symlink():
        raise ValueError(f"dataset artifact cannot be a symlink: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"dataset artifact escapes dataset root: {relative_path}")
    return resolved.read_bytes()


def _parse_daily_bar(row: object) -> DailyBar:
    if not isinstance(row, Mapping):
        raise ValueError("daily bar row must be an object")
    expected_fields = {
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    }
    if set(row) != expected_fields:
        raise ValueError("daily bar row fields do not match normalized schema")
    numeric_fields = ("open", "high", "low", "close", "volume", "amount")
    if any(
        isinstance(row[field_name], bool)
        or not isinstance(row[field_name], (int, float))
        for field_name in numeric_fields
    ):
        raise ValueError("daily bar numeric fields must be numbers")
    try:
        trade_date = date.fromisoformat(str(row["trade_date"]))
    except ValueError as exc:
        raise ValueError("daily bar trade_date must be an ISO date") from exc
    return DailyBar(
        symbol=str(row["symbol"]),
        trade_date=trade_date,
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
        amount=float(row["amount"]),
    )


def load_dataset_snapshot(
    *,
    dataset_manifest: Mapping[str, Any],
    dataset_root: Path,
    as_of: date,
) -> DatasetSnapshot:
    """Read, verify, parse, and freeze one normalized point-in-time dataset."""
    manifest_sha256 = canonical_json_sha256(dataset_manifest)
    if dataset_manifest["dataset_kind"] != "normalized":
        raise ValueError("production inference requires a normalized dataset")
    if dataset_manifest["data_schema_id"] != DAILY_BAR_SCHEMA_ID:
        raise ValueError("dataset manifest declares an unsupported data schema")
    if dataset_manifest["data_schema_sha256"] != DAILY_BAR_SCHEMA_SHA256:
        raise ValueError("dataset manifest data schema hash is incompatible")
    manifest_as_of = date.fromisoformat(str(dataset_manifest["as_of"]))

    records: list[DailyBar] = []
    seen_paths: set[Path] = set()
    for artifact in dataset_manifest["files"]:
        relative_path = Path(str(artifact["path"]))
        if relative_path in seen_paths:
            raise ValueError(f"dataset manifest contains duplicate path: {relative_path}")
        seen_paths.add(relative_path)
        content = _read_manifest_bytes(
            dataset_root=dataset_root,
            relative_path=relative_path,
        )
        if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
            raise ValueError(f"dataset artifact hash mismatch: {relative_path}")
        if len(content) != artifact["file_size_bytes"]:
            raise ValueError(f"dataset artifact size mismatch: {relative_path}")
        try:
            rows = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"dataset artifact is not valid JSON: {relative_path}") from exc
        if not isinstance(rows, list):
            raise ValueError(f"dataset artifact must contain a JSON array: {relative_path}")
        if len(rows) != artifact["row_count"]:
            raise ValueError(f"dataset artifact row count mismatch: {relative_path}")
        parsed = tuple(_parse_daily_bar(row) for row in rows)
        if not parsed:
            raise ValueError(f"dataset artifact cannot be empty: {relative_path}")
        actual_min = min(record.trade_date for record in parsed).isoformat()
        actual_max = max(record.trade_date for record in parsed).isoformat()
        if actual_min != artifact["min_trade_date"]:
            raise ValueError(f"dataset artifact min_trade_date mismatch: {relative_path}")
        if actual_max != artifact["max_trade_date"]:
            raise ValueError(f"dataset artifact max_trade_date mismatch: {relative_path}")
        if date.fromisoformat(actual_max) > manifest_as_of:
            raise ValueError(f"dataset artifact exceeds manifest as_of: {relative_path}")
        records.extend(record for record in parsed if record.trade_date <= as_of)

    return DatasetSnapshot(
        dataset_id=str(dataset_manifest["dataset_id"]),
        dataset_family_id=str(dataset_manifest["dataset_family_id"]),
        manifest_sha256=manifest_sha256,
        as_of=as_of,
        data_schema_id=str(dataset_manifest["data_schema_id"]),
        data_schema_sha256=str(dataset_manifest["data_schema_sha256"]),
        normalization_version=str(dataset_manifest["normalization_version"]),
        records=tuple(records),
    )


def build_universe_snapshot(
    *,
    universe: Mapping[str, Any],
    as_of: date,
) -> UniverseSnapshot:
    """Build a canonical point-in-time Universe from its verified contract."""
    members = tuple(
        UniverseMember(
            symbol=str(member["symbol"]),
            valid_from=date.fromisoformat(str(member["valid_from"])),
            valid_to=(
                date.fromisoformat(str(member["valid_to"]))
                if member.get("valid_to")
                else None
            ),
            eligible=bool(member["eligible"]),
            reason_codes=tuple(str(code) for code in member["reason_codes"]),
        )
        for member in universe["members"]
    )
    return UniverseSnapshot(
        universe_id=str(universe["universe_id"]),
        universe_policy_id=str(universe["universe_policy_id"]),
        universe_policy_version=str(universe["universe_policy_version"]),
        source_sha256=canonical_json_sha256(universe),
        as_of=as_of,
        members=members,
    )
