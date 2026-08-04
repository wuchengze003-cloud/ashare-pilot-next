"""Immutable, content-addressed publication for normalized daily-bar datasets.

The consumer-visible Manifest 2.0 is published only after its data directory
has been written, verified, and atomically moved into place.  Optional raw
vendor snapshots and adjustment factors live in a sibling ``auxiliary`` tree
with a separate hash index.  They never appear in ``manifest.files`` and the
normalized loader never reads them.  Telemetry and audits remain run evidence
outside this publisher.
"""

from __future__ import annotations

import fcntl
import hashlib
import ipaddress
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from .normalization import (
    NormalizedDailyBar,
    canonical_daily_bar_bytes,
    canonical_json_bytes,
)

MANIFEST_CONTRACT_ID = "dataset-manifest"
MANIFEST_SCHEMA_VERSION = "2.0.0"
DATASET_KIND = "normalized"
DATA_SCHEMA_ID = "normalized-daily-bar/v1"
DEFAULT_DATASET_FAMILY_ID = "tushare-daily/v1"
DEFAULT_NORMALIZATION_VERSION = "tushare-daily-normalizer/v1"
DEFAULT_DATA_PATH = "daily-bars.json"
AUXILIARY_INDEX_FILENAME = "auxiliary-index.json"
AUXILIARY_INDEX_FORMAT = "ashare-pilot-dataset-auxiliary-index/v1"
OFFICIAL_VENDOR_HOST = "api.tushare.pro"
OFFICIAL_SOURCE = "tushare-official"
NON_OFFICIAL_SOURCE = "tushare-via-non-official-endpoint"

DATA_SCHEMA_DESCRIPTOR: dict[str, object] = {
    "schema_id": DATA_SCHEMA_ID,
    "primary_key": ["symbol", "trade_date"],
    "fields": [
        ["symbol", "string"],
        ["trade_date", "date"],
        ["open", "finite-positive-number"],
        ["high", "finite-positive-number"],
        ["low", "finite-positive-number"],
        ["close", "finite-positive-number"],
        ["volume", "finite-nonnegative-number"],
        ["amount", "finite-nonnegative-number"],
    ],
}
DATA_SCHEMA_SHA256 = hashlib.sha256(canonical_json_bytes(DATA_SCHEMA_DESCRIPTOR)).hexdigest()

_IDENTITY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*/v[0-9]+$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_DATASET_ID_PATTERN = re.compile(r"^dataset-sha256-[a-f0-9]{64}$")
_MANIFEST_FIELDS = {
    "contract_id",
    "schema_version",
    "dataset_id",
    "dataset_family_id",
    "dataset_kind",
    "data_schema_id",
    "data_schema_sha256",
    "normalization_version",
    "as_of",
    "generated_at",
    "source",
    "source_version",
    "parent_manifest_sha256",
    "quality_status",
    "quality_reasons",
    "files",
}
_FILE_FIELDS = {
    "path",
    "sha256",
    "row_count",
    "file_size_bytes",
    "min_trade_date",
    "max_trade_date",
}
_BAR_FIELDS = {
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
}


@dataclass(frozen=True)
class DatasetSourceIdentity:
    """Non-secret source identity derived from the actual configured URL."""

    base_url_host: str
    is_official_vendor: bool
    manifest_source: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.base_url_host, str)
            or not self.base_url_host
            or self.base_url_host != self.base_url_host.lower()
            or self.base_url_host.strip() != self.base_url_host
            or any(character.isspace() for character in self.base_url_host)
            or any(
                marker in self.base_url_host
                for marker in ("@", "/", "?", "#", "\\", "[", "]")
            )
        ):
            raise ValueError("source identity base_url_host must be a sanitized lowercase host")
        if ":" in self.base_url_host:
            try:
                ipaddress.ip_address(self.base_url_host)
            except ValueError:
                raise ValueError(
                    "source identity base_url_host cannot contain a port"
                ) from None
        if type(self.is_official_vendor) is not bool:
            raise TypeError("source identity is_official_vendor must be bool")
        expected_official = self.base_url_host == OFFICIAL_VENDOR_HOST
        if self.is_official_vendor != expected_official:
            raise ValueError("source identity official flag disagrees with host")
        expected_source = OFFICIAL_SOURCE if expected_official else NON_OFFICIAL_SOURCE
        if self.manifest_source != expected_source:
            raise ValueError("source identity manifest source disagrees with host")


@dataclass(frozen=True)
class PreparedNormalizedDataset:
    """Self-validating bytes ready for immutable publication."""

    dataset_id: str
    relative_data_path: str
    data_bytes: bytes
    manifest_bytes: bytes
    auxiliary_artifacts: tuple[AuxiliaryArtifact, ...] = ()

    @property
    def manifest(self) -> dict[str, Any]:
        document = json.loads(self.manifest_bytes)
        if not isinstance(document, dict):
            raise ValueError("prepared dataset manifest must be a JSON object")
        return document


@dataclass(frozen=True)
class PublishedDatasetPaths:
    dataset_id: str
    dataset_dir: Path
    manifest_path: Path
    auxiliary_dir: Path | None


@dataclass(frozen=True, order=True)
class AuxiliaryArtifact:
    """Raw or adjustment-factor evidence excluded from normalized consumption."""

    relative_path: str
    content: bytes

    def __post_init__(self) -> None:
        path = _require_relative_path(self.relative_path, field="auxiliary path")
        parts = PurePosixPath(path).parts
        namespace = parts[0]
        if len(parts) < 2 or namespace not in {"raw", "adj_factor"}:
            raise ValueError("auxiliary path must be under raw/ or adj_factor/")
        if not isinstance(self.content, bytes):
            raise TypeError("auxiliary content must be bytes")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True)
class LoadedNormalizedDataset:
    dataset_id: str
    records: tuple[NormalizedDailyBar, ...]
    data_bytes: bytes
    manifest_bytes: bytes

    @property
    def manifest(self) -> dict[str, Any]:
        document = json.loads(self.manifest_bytes)
        if not isinstance(document, dict):
            raise ValueError("published dataset manifest must be a JSON object")
        return document


def source_identity_from_base_url(base_url: str) -> DatasetSourceIdentity:
    """Classify an endpoint, deleting the full URL before validation can fail."""
    valid_string = isinstance(base_url, str) and bool(base_url)
    if not valid_string:
        del base_url
        raise ValueError("base_url must be a non-empty string")

    try:
        parsed = urlsplit(base_url)
        scheme = parsed.scheme
        host = parsed.hostname
    except Exception:
        del base_url
        if "parsed" in locals():
            del parsed
        raise ValueError("base_url has an invalid hostname") from None
    del base_url, parsed

    if scheme not in {"http", "https"}:
        raise ValueError("base_url must use http or https")
    if host is None or not host:
        raise ValueError("base_url must include a hostname")
    normalized_host = host.lower()
    is_official = normalized_host == OFFICIAL_VENDOR_HOST
    return DatasetSourceIdentity(
        base_url_host=normalized_host,
        is_official_vendor=is_official,
        manifest_source=OFFICIAL_SOURCE if is_official else NON_OFFICIAL_SOURCE,
    )


def _require_identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTITY_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must match <lowercase-id>/v<major>")
    return value


def _require_sha256(value: object, *, field: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{field} must be a safe POSIX relative path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"{field} must be a safe POSIX relative path")
    return value


def _require_utc_datetime(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value.astimezone(UTC)


def _format_utc_datetime(value: datetime) -> str:
    return _require_utc_datetime(value, field="generated_at").isoformat().replace("+00:00", "Z")


def _parse_iso_date(value: object, *, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must use canonical ISO date form")
    return parsed


def _parse_utc_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO datetime")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO datetime") from exc
    return _require_utc_datetime(parsed, field=field)


def _dataset_identity(document: Mapping[str, Any]) -> dict[str, object]:
    """Select stable content and contract identity; exclude run-time evidence."""
    return {
        "dataset_family_id": document["dataset_family_id"],
        "dataset_kind": document["dataset_kind"],
        "data_schema_id": document["data_schema_id"],
        "data_schema_sha256": document["data_schema_sha256"],
        "normalization_version": document["normalization_version"],
        "as_of": document["as_of"],
        "source": document["source"],
        "source_version": document["source_version"],
        "parent_manifest_sha256": document["parent_manifest_sha256"],
        "files": document["files"],
    }


def _derive_dataset_id(document: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(_dataset_identity(document))).hexdigest()
    return f"dataset-sha256-{digest}"


def _normalize_auxiliary_artifacts(
    artifacts: Iterable[AuxiliaryArtifact],
) -> tuple[AuxiliaryArtifact, ...]:
    materialized = tuple(artifacts)
    if any(not isinstance(artifact, AuxiliaryArtifact) for artifact in materialized):
        raise TypeError("auxiliary_artifacts must contain only AuxiliaryArtifact values")
    ordered = tuple(sorted(materialized, key=lambda artifact: artifact.relative_path))
    paths = [artifact.relative_path for artifact in ordered]
    if len(paths) != len(set(paths)):
        raise ValueError("auxiliary artifact paths must be unique")
    return ordered


def _auxiliary_index_bytes(prepared: PreparedNormalizedDataset) -> bytes:
    return canonical_json_bytes(
        {
            "format": AUXILIARY_INDEX_FORMAT,
            "normalized_dataset_id": prepared.dataset_id,
            "files": [
                {
                    "path": artifact.relative_path,
                    "sha256": artifact.sha256,
                    "file_size_bytes": len(artifact.content),
                }
                for artifact in prepared.auxiliary_artifacts
            ],
        }
    )


def _parse_normalized_data(data_bytes: bytes) -> tuple[NormalizedDailyBar, ...]:
    try:
        document = json.loads(data_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("normalized data is not valid JSON") from exc
    if not isinstance(document, list):
        raise ValueError("normalized data must be a JSON array")
    if not document:
        raise ValueError("normalized data cannot be empty")

    records: list[NormalizedDailyBar] = []
    for index, row in enumerate(document):
        if not isinstance(row, dict) or set(row) != _BAR_FIELDS:
            raise ValueError(f"normalized row {index} has unexpected fields")
        trade_date = _parse_iso_date(row["trade_date"], field=f"normalized row {index} trade_date")
        records.append(
            NormalizedDailyBar(
                symbol=row["symbol"],
                trade_date=trade_date,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                amount=row["amount"],
            )
        )
    result = tuple(records)
    if canonical_daily_bar_bytes(result) != data_bytes:
        raise ValueError("normalized data bytes are not canonical primary-key order")
    return result


def _validate_manifest_document(document: Mapping[str, Any]) -> None:
    if set(document) != _MANIFEST_FIELDS:
        raise ValueError("dataset manifest has missing or unexpected fields")
    if document["contract_id"] != MANIFEST_CONTRACT_ID:
        raise ValueError("dataset manifest contract_id is invalid")
    if document["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("dataset manifest schema_version is invalid")
    dataset_id = document["dataset_id"]
    if not isinstance(dataset_id, str) or not _DATASET_ID_PATTERN.fullmatch(dataset_id):
        raise ValueError("dataset manifest dataset_id is invalid")
    _require_identity(document["dataset_family_id"], field="dataset_family_id")
    if document["dataset_kind"] != DATASET_KIND:
        raise ValueError("dataset manifest must describe a normalized dataset")
    if document["data_schema_id"] != DATA_SCHEMA_ID:
        raise ValueError("dataset manifest data_schema_id is unsupported")
    if document["data_schema_sha256"] != DATA_SCHEMA_SHA256:
        raise ValueError("dataset manifest data schema hash is incompatible")
    _require_identity(document["normalization_version"], field="normalization_version")
    _parse_iso_date(document["as_of"], field="as_of")
    _parse_utc_datetime(document["generated_at"], field="generated_at")
    if document["source"] not in {OFFICIAL_SOURCE, NON_OFFICIAL_SOURCE}:
        raise ValueError("dataset manifest source is invalid")
    source_version = document["source_version"]
    if not isinstance(source_version, str) or not 1 <= len(source_version) <= 80:
        raise ValueError("dataset manifest source_version is invalid")
    _require_sha256(
        document["parent_manifest_sha256"],
        field="parent_manifest_sha256",
        nullable=True,
    )
    if document["quality_status"] not in {"pass", "fail"}:
        raise ValueError("dataset manifest quality_status is invalid")
    quality_reasons = document["quality_reasons"]
    if not isinstance(quality_reasons, list) or any(
        not isinstance(reason, str) or not 1 <= len(reason) <= 160 for reason in quality_reasons
    ):
        raise ValueError("dataset manifest quality_reasons is invalid")
    if len(quality_reasons) != len(set(quality_reasons)):
        raise ValueError("dataset manifest quality_reasons must be unique")
    if document["quality_status"] == "pass" and quality_reasons:
        raise ValueError("passing dataset manifest cannot contain quality_reasons")
    if document["quality_status"] == "fail" and not quality_reasons:
        raise ValueError("failing dataset manifest must contain quality_reasons")

    files = document["files"]
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError("M1 normalized dataset manifest must contain exactly one file")
    artifact = files[0]
    if not isinstance(artifact, dict) or set(artifact) != _FILE_FIELDS:
        raise ValueError("dataset manifest file entry is invalid")
    _require_relative_path(artifact["path"], field="files.path")
    _require_sha256(artifact["sha256"], field="files.sha256")
    if (
        isinstance(artifact["row_count"], bool)
        or not isinstance(artifact["row_count"], int)
        or artifact["row_count"] < 1
    ):
        raise ValueError("dataset manifest row_count must be a positive integer")
    if (
        isinstance(artifact["file_size_bytes"], bool)
        or not isinstance(artifact["file_size_bytes"], int)
        or artifact["file_size_bytes"] < 1
    ):
        raise ValueError("dataset manifest file_size_bytes must be a positive integer")
    minimum = _parse_iso_date(artifact["min_trade_date"], field="min_trade_date")
    maximum = _parse_iso_date(artifact["max_trade_date"], field="max_trade_date")
    if maximum < minimum:
        raise ValueError("dataset manifest maximum trade date precedes minimum")
    if maximum > _parse_iso_date(document["as_of"], field="as_of"):
        raise ValueError("dataset manifest data extends beyond as_of")
    if _derive_dataset_id(document) != dataset_id:
        raise ValueError("dataset_id does not match stable content identity")


def prepare_normalized_dataset(
    records: Iterable[NormalizedDailyBar],
    *,
    as_of: date,
    generated_at: datetime,
    source_identity: DatasetSourceIdentity,
    source_version: str,
    dataset_family_id: str = DEFAULT_DATASET_FAMILY_ID,
    normalization_version: str = DEFAULT_NORMALIZATION_VERSION,
    parent_manifest_sha256: str | None = None,
    relative_data_path: str = DEFAULT_DATA_PATH,
    quality_status: str = "pass",
    quality_reasons: Iterable[str] = (),
    auxiliary_artifacts: Iterable[AuxiliaryArtifact] = (),
) -> PreparedNormalizedDataset:
    """Build canonical data and Manifest 2.0 bytes without touching disk."""
    if type(as_of) is not date:
        raise TypeError("as_of must be a date")
    generated_text = _format_utc_datetime(generated_at)
    if type(source_identity) is not DatasetSourceIdentity:
        raise TypeError("source_identity must be DatasetSourceIdentity")
    _require_identity(dataset_family_id, field="dataset_family_id")
    _require_identity(normalization_version, field="normalization_version")
    _require_sha256(
        parent_manifest_sha256,
        field="parent_manifest_sha256",
        nullable=True,
    )
    relative_data_path = _require_relative_path(relative_data_path, field="relative_data_path")
    if not isinstance(source_version, str) or not 1 <= len(source_version) <= 80:
        raise ValueError("source_version must contain 1 to 80 characters")
    if quality_status not in {"pass", "fail"}:
        raise ValueError("quality_status must be pass or fail")
    materialized_reasons = tuple(quality_reasons)
    if any(
        not isinstance(reason, str) or not 1 <= len(reason) <= 160
        for reason in materialized_reasons
    ):
        raise ValueError("quality_reasons must be unique strings of 1 to 160 characters")
    reasons = tuple(sorted(materialized_reasons))
    if len(reasons) != len(set(reasons)):
        raise ValueError("quality_reasons must be unique strings of 1 to 160 characters")
    if quality_status == "pass" and reasons:
        raise ValueError("a passing dataset cannot contain quality_reasons")
    if quality_status == "fail" and not reasons:
        raise ValueError("a failing dataset must contain at least one quality reason")
    ordered_auxiliary = _normalize_auxiliary_artifacts(auxiliary_artifacts)

    materialized = tuple(records)
    if any(not isinstance(row, NormalizedDailyBar) for row in materialized):
        raise TypeError("records must contain only NormalizedDailyBar values")
    ordered = tuple(sorted(materialized, key=lambda row: (row.symbol, row.trade_date)))
    data_bytes = canonical_daily_bar_bytes(ordered)
    if not ordered:
        raise ValueError("cannot prepare an empty normalized dataset")
    if ordered[-1].trade_date > as_of:
        raise ValueError("normalized dataset contains records after as_of")

    file_entry: dict[str, object] = {
        "path": relative_data_path,
        "sha256": hashlib.sha256(data_bytes).hexdigest(),
        "row_count": len(ordered),
        "file_size_bytes": len(data_bytes),
        "min_trade_date": min(row.trade_date for row in ordered).isoformat(),
        "max_trade_date": max(row.trade_date for row in ordered).isoformat(),
    }
    manifest: dict[str, object] = {
        "contract_id": MANIFEST_CONTRACT_ID,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": "",
        "dataset_family_id": dataset_family_id,
        "dataset_kind": DATASET_KIND,
        "data_schema_id": DATA_SCHEMA_ID,
        "data_schema_sha256": DATA_SCHEMA_SHA256,
        "normalization_version": normalization_version,
        "as_of": as_of.isoformat(),
        "generated_at": generated_text,
        "source": source_identity.manifest_source,
        "source_version": source_version,
        "parent_manifest_sha256": parent_manifest_sha256,
        "quality_status": quality_status,
        "quality_reasons": list(reasons),
        "files": [file_entry],
    }
    manifest["dataset_id"] = _derive_dataset_id(manifest)
    prepared = PreparedNormalizedDataset(
        dataset_id=str(manifest["dataset_id"]),
        relative_data_path=relative_data_path,
        data_bytes=data_bytes,
        manifest_bytes=canonical_json_bytes(manifest),
        auxiliary_artifacts=ordered_auxiliary,
    )
    validate_prepared_dataset(prepared)
    return prepared


def validate_prepared_dataset(
    prepared: PreparedNormalizedDataset,
) -> tuple[NormalizedDailyBar, ...]:
    """Recompute every identity and content claim in prepared artifacts."""
    if not isinstance(prepared, PreparedNormalizedDataset):
        raise TypeError("prepared must be PreparedNormalizedDataset")
    if _normalize_auxiliary_artifacts(prepared.auxiliary_artifacts) != (
        prepared.auxiliary_artifacts
    ):
        raise ValueError("prepared auxiliary artifacts are not in canonical path order")
    document = prepared.manifest
    if canonical_json_bytes(document) != prepared.manifest_bytes:
        raise ValueError("dataset manifest bytes are not canonical")
    _validate_manifest_document(document)
    if document["dataset_id"] != prepared.dataset_id:
        raise ValueError("prepared dataset_id does not match its manifest")
    artifact = document["files"][0]
    if artifact["path"] != prepared.relative_data_path:
        raise ValueError("prepared data path does not match its manifest")
    if hashlib.sha256(prepared.data_bytes).hexdigest() != artifact["sha256"]:
        raise ValueError("normalized data hash does not match its manifest")
    if len(prepared.data_bytes) != artifact["file_size_bytes"]:
        raise ValueError("normalized data size does not match its manifest")
    records = _parse_normalized_data(prepared.data_bytes)
    if len(records) != artifact["row_count"]:
        raise ValueError("normalized row count does not match its manifest")
    if min(row.trade_date for row in records).isoformat() != artifact["min_trade_date"]:
        raise ValueError("normalized minimum trade date does not match its manifest")
    if max(row.trade_date for row in records).isoformat() != artifact["max_trade_date"]:
        raise ValueError("normalized maximum trade date does not match its manifest")
    return records


def _write_new_fsynced(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"publication directory cannot be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"publication path must be a directory: {path}")


def _fsync_tree_directories(root: Path) -> None:
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(root)


def _validate_dataset_directory(
    *,
    dataset_dir: Path,
    prepared: PreparedNormalizedDataset,
) -> None:
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise ValueError("published dataset directory is invalid")
    entries = list(dataset_dir.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("published dataset cannot contain symlinks")
    files = [path for path in entries if not path.is_dir()]
    if any(not path.is_file() for path in files):
        raise ValueError("published dataset may contain only regular files")
    relative_files = {path.relative_to(dataset_dir).as_posix() for path in files}
    if relative_files != {prepared.relative_data_path}:
        raise ValueError("published dataset has missing or unexpected files")
    content = (dataset_dir / PurePosixPath(prepared.relative_data_path)).read_bytes()
    on_disk = PreparedNormalizedDataset(
        dataset_id=prepared.dataset_id,
        relative_data_path=prepared.relative_data_path,
        data_bytes=content,
        manifest_bytes=prepared.manifest_bytes,
    )
    validate_prepared_dataset(on_disk)


def _validate_auxiliary_directory(
    *,
    auxiliary_dir: Path,
    prepared: PreparedNormalizedDataset,
) -> None:
    if not prepared.auxiliary_artifacts:
        raise ValueError("auxiliary directory is unexpected for a dataset without evidence")
    if auxiliary_dir.is_symlink() or not auxiliary_dir.is_dir():
        raise ValueError("published auxiliary directory is invalid")
    entries = list(auxiliary_dir.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("published auxiliary tree cannot contain symlinks")
    files = [path for path in entries if path.is_file()]
    expected_paths = {
        AUXILIARY_INDEX_FILENAME,
        *(artifact.relative_path for artifact in prepared.auxiliary_artifacts),
    }
    actual_paths = {path.relative_to(auxiliary_dir).as_posix() for path in files}
    if actual_paths != expected_paths:
        raise ValueError("published auxiliary tree has missing or unexpected files")

    for artifact in prepared.auxiliary_artifacts:
        content = (auxiliary_dir / PurePosixPath(artifact.relative_path)).read_bytes()
        if content != artifact.content or hashlib.sha256(content).hexdigest() != artifact.sha256:
            raise ValueError(f"auxiliary artifact hash mismatch: {artifact.relative_path}")
    index_bytes = (auxiliary_dir / AUXILIARY_INDEX_FILENAME).read_bytes()
    if index_bytes != _auxiliary_index_bytes(prepared):
        raise ValueError("auxiliary index does not match prepared evidence")


def publish_normalized_dataset(
    *,
    publication_root: Path,
    prepared: PreparedNormalizedDataset,
) -> PublishedDatasetPaths:
    """Atomically publish data first and its consumer-visible manifest last.

    Existing normalized, auxiliary, or manifest paths are never overwritten,
    including when their bytes happen to match.  Any handled failure before
    manifest commit removes directories created by this call, leaving no
    consumer-visible partial run.
    """
    validate_prepared_dataset(prepared)
    if prepared.manifest["quality_status"] != "pass":
        raise ValueError("a failed-quality dataset cannot be published")

    publication_root = Path(publication_root)
    _ensure_directory(publication_root)
    datasets_root = publication_root / "datasets"
    manifests_root = publication_root / "manifests"
    _ensure_directory(datasets_root)
    _ensure_directory(manifests_root)
    dataset_dir = datasets_root / prepared.dataset_id
    manifest_path = manifests_root / f"{prepared.dataset_id}.json"
    auxiliary_root_path = publication_root / "auxiliary"
    if auxiliary_root_path.exists() and (
        auxiliary_root_path.is_symlink() or not auxiliary_root_path.is_dir()
    ):
        raise ValueError("publication auxiliary root is invalid")
    candidate_auxiliary_dir = auxiliary_root_path / prepared.dataset_id
    auxiliary_root: Path | None = None
    auxiliary_dir: Path | None = None
    if prepared.auxiliary_artifacts:
        auxiliary_root = auxiliary_root_path
        _ensure_directory(auxiliary_root)
        auxiliary_dir = candidate_auxiliary_dir
    lock_path = publication_root / ".dataset-publication.lock"
    if lock_path.is_symlink():
        raise ValueError("dataset publication lock cannot be a symlink")

    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if dataset_dir.exists() or dataset_dir.is_symlink():
            raise FileExistsError(f"dataset already exists: {prepared.dataset_id}")
        if manifest_path.exists() or manifest_path.is_symlink():
            raise FileExistsError(f"dataset manifest already exists: {prepared.dataset_id}")
        if candidate_auxiliary_dir.exists() or candidate_auxiliary_dir.is_symlink():
            raise FileExistsError(
                f"dataset auxiliary evidence already exists: {prepared.dataset_id}"
            )

        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{prepared.dataset_id}.tmp-", dir=datasets_root)
        )
        temporary_auxiliary: Path | None = None
        if auxiliary_root is not None:
            temporary_auxiliary = Path(
                tempfile.mkdtemp(
                    prefix=f".{prepared.dataset_id}.tmp-",
                    dir=auxiliary_root,
                )
            )
        temporary_manifest: Path | None = None
        auxiliary_moved = False
        dataset_moved = False
        manifest_published = False
        try:
            staged_data_path = temporary_dir / PurePosixPath(prepared.relative_data_path)
            _write_new_fsynced(staged_data_path, prepared.data_bytes)
            _fsync_tree_directories(temporary_dir)
            _validate_dataset_directory(dataset_dir=temporary_dir, prepared=prepared)

            if temporary_auxiliary is not None:
                for artifact in prepared.auxiliary_artifacts:
                    _write_new_fsynced(
                        temporary_auxiliary / PurePosixPath(artifact.relative_path),
                        artifact.content,
                    )
                _write_new_fsynced(
                    temporary_auxiliary / AUXILIARY_INDEX_FILENAME,
                    _auxiliary_index_bytes(prepared),
                )
                _fsync_tree_directories(temporary_auxiliary)
                _validate_auxiliary_directory(
                    auxiliary_dir=temporary_auxiliary,
                    prepared=prepared,
                )
                assert auxiliary_dir is not None
                assert auxiliary_root is not None
                os.replace(temporary_auxiliary, auxiliary_dir)
                auxiliary_moved = True
                _fsync_directory(auxiliary_root)
                _validate_auxiliary_directory(
                    auxiliary_dir=auxiliary_dir,
                    prepared=prepared,
                )

            os.replace(temporary_dir, dataset_dir)
            dataset_moved = True
            _fsync_directory(datasets_root)
            _validate_dataset_directory(dataset_dir=dataset_dir, prepared=prepared)

            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{prepared.dataset_id}.manifest-",
                suffix=".tmp",
                dir=manifests_root,
            )
            temporary_manifest = Path(temporary_name)
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(prepared.manifest_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            staged_manifest = temporary_manifest.read_bytes()
            if staged_manifest != prepared.manifest_bytes:
                raise ValueError("staged manifest bytes changed during publication")
            staged_document = json.loads(staged_manifest)
            if not isinstance(staged_document, dict):
                raise ValueError("staged manifest must be a JSON object")
            _validate_manifest_document(staged_document)

            os.replace(temporary_manifest, manifest_path)
            manifest_published = True
            temporary_manifest = None
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
            if temporary_auxiliary is not None and not auxiliary_moved:
                shutil.rmtree(temporary_auxiliary, ignore_errors=True)
            elif auxiliary_moved and not manifest_published:
                assert auxiliary_dir is not None
                shutil.rmtree(auxiliary_dir, ignore_errors=True)
                assert auxiliary_root is not None
                with suppress(OSError):
                    _fsync_directory(auxiliary_root)
            raise

    return PublishedDatasetPaths(
        dataset_id=prepared.dataset_id,
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        auxiliary_dir=auxiliary_dir,
    )


def load_published_dataset(
    *,
    publication_root: Path,
    dataset_id: str,
) -> LoadedNormalizedDataset:
    """Load only a manifest-committed dataset and verify it against tampering."""
    if not isinstance(dataset_id, str) or not _DATASET_ID_PATTERN.fullmatch(dataset_id):
        raise ValueError("dataset_id is invalid")
    publication_root = Path(publication_root)
    if publication_root.is_symlink() or not publication_root.is_dir():
        raise ValueError("publication_root is invalid")
    datasets_root = publication_root / "datasets"
    manifests_root = publication_root / "manifests"
    if (
        datasets_root.is_symlink()
        or manifests_root.is_symlink()
        or not datasets_root.is_dir()
        or not manifests_root.is_dir()
    ):
        raise ValueError("publication layout is invalid")
    manifest_path = manifests_root / f"{dataset_id}.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"published dataset manifest is missing: {dataset_id}")
    manifest_bytes = manifest_path.read_bytes()
    try:
        document = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("published dataset manifest is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ValueError("published dataset manifest must be a JSON object")
    if canonical_json_bytes(document) != manifest_bytes:
        raise ValueError("published dataset manifest bytes are not canonical")
    _validate_manifest_document(document)
    if document["dataset_id"] != dataset_id:
        raise ValueError("published manifest does not match requested dataset_id")
    if document["quality_status"] != "pass" or document["quality_reasons"]:
        raise ValueError("published normalized dataset did not pass quality gates")

    relative_data_path = _require_relative_path(document["files"][0]["path"], field="files.path")
    dataset_dir = datasets_root / dataset_id
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise ValueError("published normalized dataset directory is invalid")
    data_path = dataset_dir / PurePosixPath(relative_data_path)
    if data_path.is_symlink() or not data_path.is_file():
        raise ValueError("published normalized data file is missing or invalid")
    prepared = PreparedNormalizedDataset(
        dataset_id=dataset_id,
        relative_data_path=relative_data_path,
        data_bytes=data_path.read_bytes(),
        manifest_bytes=manifest_bytes,
    )
    records = validate_prepared_dataset(prepared)
    _validate_dataset_directory(dataset_dir=dataset_dir, prepared=prepared)
    return LoadedNormalizedDataset(
        dataset_id=dataset_id,
        records=records,
        data_bytes=prepared.data_bytes,
        manifest_bytes=manifest_bytes,
    )


def load_published_auxiliary_artifacts(
    *,
    publication_root: Path,
    dataset_id: str,
) -> tuple[AuxiliaryArtifact, ...]:
    """Load separately indexed evidence after the normalized commit is verified.

    This API is intentionally separate from :func:`load_published_dataset` so
    Research and Signal Runner cannot consume raw or adjustment-factor bytes as
    normalized daily bars by accident.
    """
    loaded = load_published_dataset(
        publication_root=publication_root,
        dataset_id=dataset_id,
    )
    publication_root = Path(publication_root)
    auxiliary_root = publication_root / "auxiliary"
    if not auxiliary_root.exists():
        return ()
    if auxiliary_root.is_symlink() or not auxiliary_root.is_dir():
        raise ValueError("published auxiliary root is invalid")
    auxiliary_dir = auxiliary_root / dataset_id
    if not auxiliary_dir.exists():
        return ()
    if auxiliary_dir.is_symlink() or not auxiliary_dir.is_dir():
        raise ValueError("published auxiliary directory is invalid")
    index_path = auxiliary_dir / AUXILIARY_INDEX_FILENAME
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("published auxiliary index is missing or invalid")
    index_bytes = index_path.read_bytes()
    try:
        index = json.loads(index_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("published auxiliary index is not valid JSON") from exc
    if not isinstance(index, dict) or set(index) != {
        "format",
        "normalized_dataset_id",
        "files",
    }:
        raise ValueError("published auxiliary index has unexpected fields")
    if index["format"] != AUXILIARY_INDEX_FORMAT:
        raise ValueError("published auxiliary index format is invalid")
    if index["normalized_dataset_id"] != dataset_id:
        raise ValueError("published auxiliary index dataset identity is invalid")
    indexed_files = index["files"]
    if not isinstance(indexed_files, list) or not indexed_files:
        raise ValueError("published auxiliary index must contain files")

    artifacts: list[AuxiliaryArtifact] = []
    for entry in indexed_files:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "file_size_bytes",
        }:
            raise ValueError("published auxiliary index file entry is invalid")
        relative_path = _require_relative_path(entry["path"], field="auxiliary path")
        expected_hash = _require_sha256(entry["sha256"], field="auxiliary sha256")
        expected_size = entry["file_size_bytes"]
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ValueError("auxiliary file_size_bytes must be nonnegative")
        artifact_path = auxiliary_dir / PurePosixPath(relative_path)
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ValueError(f"published auxiliary artifact is missing: {relative_path}")
        content = artifact_path.read_bytes()
        if len(content) != expected_size:
            raise ValueError(f"auxiliary artifact size mismatch: {relative_path}")
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"auxiliary artifact hash mismatch: {relative_path}")
        artifacts.append(AuxiliaryArtifact(relative_path=relative_path, content=content))

    ordered = _normalize_auxiliary_artifacts(artifacts)
    prepared = PreparedNormalizedDataset(
        dataset_id=loaded.dataset_id,
        relative_data_path=str(loaded.manifest["files"][0]["path"]),
        data_bytes=loaded.data_bytes,
        manifest_bytes=loaded.manifest_bytes,
        auxiliary_artifacts=ordered,
    )
    if index_bytes != _auxiliary_index_bytes(prepared):
        raise ValueError("published auxiliary index bytes are not canonical")
    _validate_auxiliary_directory(auxiliary_dir=auxiliary_dir, prepared=prepared)
    return ordered
