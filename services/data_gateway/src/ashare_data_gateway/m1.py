"""M1 fixed-scope real-data acquisition and immutable publication.

This module intentionally exposes no symbol or date overrides.  The audit
selection and market window are the frozen M1 acceptance inputs; changing
either requires revising the acceptance document before changing this code.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .acquisition import CountingTransport, RawResponseSnapshot
from .coverage import (
    SecurityLifecycle,
    UniverseMembership,
    audit_historical_coverage,
)
from .dataset_publication import (
    AuxiliaryArtifact,
    PublishedDatasetPaths,
    load_published_auxiliary_artifacts,
    load_published_dataset,
    prepare_normalized_dataset,
    publish_normalized_dataset,
)
from .normalization import (
    NormalizationError,
    canonical_json_bytes,
    normalize_daily_bars,
)
from .tushare_client import TushareClient
from .tushare_models import (
    AdjFactorRecord,
    DailyBarRecord,
    StockBasicBatchResult,
    StockBasicRecord,
    SuspendRecord,
    TradeCalRecord,
)
from .tushare_transport import (
    HttpJsonTransport,
    Transport,
    TushareAPIError,
    TushareTransientError,
)

WINDOW_START = date(2023, 6, 1)
WINDOW_END = date(2023, 10, 31)
DATASET_AS_OF = WINDOW_END
TRADE_CAL_EXCHANGE = "SSE"
SOURCE_VERSION = "tushare-pro-api/v1"

NORMAL_SYMBOLS = (
    "000001.SZ",
    "000333.SZ",
    "000858.SZ",
    "002415.SZ",
    "002594.SZ",
    "300014.SZ",
    "300059.SZ",
    "300750.SZ",
    "600036.SH",
    "600519.SH",
    "601318.SH",
    "601398.SH",
    "688008.SH",
    "688111.SH",
    "688981.SH",
)
DELISTED_SYMBOLS = (
    "600466.SH",
    "000732.SZ",
    "000666.SZ",
)
LONG_SUSPENDED_SYMBOLS = (
    "600530.SH",
    "002564.SZ",
)
AUDIT_SYMBOLS = tuple(sorted(NORMAL_SYMBOLS + DELISTED_SYMBOLS + LONG_SUSPENDED_SYMBOLS))

EXPECTED_DELIST_DATES = {
    "600466.SH": date(2023, 6, 6),
    "000732.SZ": date(2023, 8, 4),
    "000666.SZ": date(2023, 10, 26),
}
EXPECTED_LIST_STATUS = {
    **{symbol: "L" for symbol in NORMAL_SYMBOLS},
    **{symbol: "D" for symbol in DELISTED_SYMBOLS},
    **{symbol: "L" for symbol in LONG_SUSPENDED_SYMBOLS},
}

_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")


@dataclass(frozen=True)
class M1RunSpec:
    """Explicit non-market-time inputs for one immutable M1 run."""

    output_root: Path
    run_id: str
    generated_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be a Path")
        if not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id must be a safe lowercase runtime identifier")
        if (
            not isinstance(self.generated_at, datetime)
            or self.generated_at.tzinfo is None
            or self.generated_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("generated_at must be timezone-aware UTC")


@dataclass(frozen=True)
class M1RunResult:
    """Paths and stable identity of one completed M1 publication."""

    dataset_id: str
    dataset_dir: Path
    manifest_path: Path
    auxiliary_dir: Path
    run_dir: Path
    normalized_row_count: int
    coverage_expected_member_days: int
    http_attempts: int

    @property
    def document(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_dir": str(self.dataset_dir),
            "manifest_path": str(self.manifest_path),
            "auxiliary_dir": str(self.auxiliary_dir),
            "run_dir": str(self.run_dir),
            "normalized_row_count": self.normalized_row_count,
            "coverage_expected_member_days": self.coverage_expected_member_days,
            "http_attempts": self.http_attempts,
        }


@dataclass(frozen=True)
class _AcquiredInputs:
    trade_calendar: tuple[TradeCalRecord, ...]
    stock_batches: tuple[StockBasicBatchResult, ...]
    daily_bars: tuple[DailyBarRecord, ...]
    adjustment_factors: tuple[AdjFactorRecord, ...]
    suspension_records: tuple[SuspendRecord, ...]


def _iter_dates(start: date, end: date) -> tuple[date, ...]:
    return tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))


def _validate_trade_calendar(records: tuple[TradeCalRecord, ...]) -> tuple[date, ...]:
    expected_dates = _iter_dates(WINDOW_START, WINDOW_END)
    by_date: dict[date, TradeCalRecord] = {}
    for record in records:
        if record.exchange != TRADE_CAL_EXCHANGE:
            raise ValueError("M1 trade_cal returned an unexpected exchange")
        if record.cal_date in by_date:
            raise ValueError("M1 trade_cal contains a duplicate calendar date")
        by_date[record.cal_date] = record
    missing = sorted(set(expected_dates) - by_date.keys())
    extra = sorted(by_date.keys() - set(expected_dates))
    if missing or extra:
        raise ValueError(
            "M1 trade_cal must contain every calendar date in the frozen window; "
            f"missing={len(missing)} extra={len(extra)}"
        )
    open_days = tuple(day for day in expected_dates if by_date[day].is_open == 1)
    if not open_days or open_days[-1] != DATASET_AS_OF:
        raise ValueError("M1 trade_cal does not establish the frozen as_of as an open day")
    return open_days


def _reconcile_stock_basic(
    batches: tuple[StockBasicBatchResult, ...],
) -> tuple[dict[str, StockBasicRecord], dict[str, object]]:
    if tuple(batch.list_status for batch in batches) != ("L", "D", "P"):
        raise ValueError("M1 stock_basic batches must be acquired in L/D/P order")

    by_symbol: dict[str, StockBasicRecord] = {}
    batch_evidence: dict[str, object] = {}
    for batch in batches:
        if batch.raw_row_count != len(batch.accepted) + len(batch.rejected):
            raise ValueError("M1 stock_basic reconciliation invariant failed")
        for record in batch.accepted:
            if record.ts_code in by_symbol:
                raise ValueError(
                    "M1 stock_basic duplicates a symbol across states: "
                    f"{record.ts_code}"
                )
            by_symbol[record.ts_code] = record
        rejected = tuple(
            sorted(
                batch.rejected,
                key=lambda item: (
                    item.vendor_ts_code or "",
                    item.row_fingerprint,
                ),
            )
        )
        batch_evidence[batch.list_status] = {
            "raw_rows": batch.raw_row_count,
            "accepted_rows": len(batch.accepted),
            "rejected_rows": len(rejected),
            "rejections": [
                {
                    "reason": item.reason,
                    "row_fingerprint": item.row_fingerprint,
                    "vendor_ts_code": item.vendor_ts_code,
                }
                for item in rejected
            ],
        }

    missing_symbols = sorted(set(AUDIT_SYMBOLS) - by_symbol.keys())
    if missing_symbols:
        raise ValueError(f"M1 stock_basic is missing frozen audit symbols: {missing_symbols}")
    for symbol in AUDIT_SYMBOLS:
        record = by_symbol[symbol]
        expected_status = EXPECTED_LIST_STATUS[symbol]
        if record.list_status != expected_status:
            raise ValueError(
                f"M1 stock_basic status changed for {symbol}: "
                f"expected={expected_status} actual={record.list_status}"
            )
        if record.list_date is None or record.list_date > WINDOW_START:
            raise ValueError(f"M1 stock_basic has an invalid list_date for {symbol}")
        expected_delist = EXPECTED_DELIST_DATES.get(symbol)
        if record.delist_date != expected_delist:
            raise ValueError(
                f"M1 stock_basic delist boundary changed for {symbol}: "
                f"expected={expected_delist} actual={record.delist_date}"
            )
    return by_symbol, batch_evidence


def _validate_daily_market_days_and_lifecycles(
    records: tuple[DailyBarRecord, ...],
    *,
    open_days: tuple[date, ...],
    stock_by_symbol: dict[str, StockBasicRecord],
) -> None:
    open_day_set = frozenset(open_days)
    for record in records:
        if record.trade_date not in open_day_set:
            raise ValueError(
                f"M1 daily contains a bar on a closed calendar day: "
                f"{record.ts_code} {record.trade_date.isoformat()}"
            )
        security = stock_by_symbol[record.ts_code]
        list_date = security.list_date
        if list_date is None:
            raise ValueError(f"M1 lifecycle lacks list_date for {record.ts_code}")
        if record.trade_date < list_date or (
            security.delist_date is not None and record.trade_date > security.delist_date
        ):
            raise ValueError(
                f"M1 daily bar is outside the verified lifecycle: "
                f"{record.ts_code} {record.trade_date.isoformat()}"
            )


def _alignment_document(
    daily_bars: tuple[DailyBarRecord, ...],
    adjustment_factors: tuple[AdjFactorRecord, ...],
) -> dict[str, object]:
    daily_keys = frozenset((row.ts_code, row.trade_date) for row in daily_bars)
    factor_keys = frozenset((row.ts_code, row.trade_date) for row in adjustment_factors)
    daily_only = tuple(sorted(daily_keys - factor_keys))
    factor_only = tuple(sorted(factor_keys - daily_keys))
    matched = daily_keys & factor_keys

    def samples(keys: Iterable[tuple[str, date]]) -> list[dict[str, str]]:
        return [
            {"symbol": symbol, "trade_date": trade_date.isoformat()}
            for symbol, trade_date in tuple(keys)[:20]
        ]

    return {
        "daily_only_keys": len(daily_only),
        "adj_factor_only_keys": len(factor_only),
        "matched_keys": len(matched),
        "daily_only_samples": samples(daily_only),
        "adj_factor_only_samples": samples(factor_only),
    }


def _adj_factor_bytes(records: tuple[AdjFactorRecord, ...]) -> bytes:
    return canonical_json_bytes(
        [
            {
                "ts_code": record.ts_code,
                "trade_date": record.trade_date.isoformat(),
                "adj_factor": record.adj_factor,
            }
            for record in sorted(records)
        ]
    )


def _safe_raw_json_value(value: object) -> object:
    """Preserve invalid vendor cells as explicit JSON-safe evidence markers."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            label = "NaN"
        elif value > 0:
            label = "Infinity"
        else:
            label = "-Infinity"
        return {"invalid_vendor_number": label}
    if isinstance(value, (list, tuple)):
        return [_safe_raw_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _safe_raw_json_value(item)
            for key, item in value.items()
        }
    return {"invalid_vendor_type": type(value).__name__}


def _raw_response_bytes(records: tuple[RawResponseSnapshot, ...]) -> bytes:
    return canonical_json_bytes(
        {
            "format": "tushare-raw-responses/v1",
            "responses": [
                _safe_raw_json_value(record.to_dict()) for record in records
            ],
        }
    )


def _acquire(client: TushareClient) -> _AcquiredInputs:
    trade_calendar = client.fetch_trade_cal(
        exchange=TRADE_CAL_EXCHANGE,
        start_date=WINDOW_START,
        end_date=WINDOW_END,
    )
    stock_batches = tuple(
        client.fetch_stock_basic_reconciled(list_status=status)
        for status in ("L", "D", "P")
    )

    daily_bars: list[DailyBarRecord] = []
    adjustment_factors: list[AdjFactorRecord] = []
    for symbol in AUDIT_SYMBOLS:
        daily_bars.extend(
            client.fetch_daily(
                ts_code=symbol,
                start_date=WINDOW_START,
                end_date=WINDOW_END,
            )
        )
        adjustment_factors.extend(
            client.fetch_adj_factor(
                ts_code=symbol,
                start_date=WINDOW_START,
                end_date=WINDOW_END,
            )
        )
    suspension_records = client.fetch_suspend_d(
        ts_codes=AUDIT_SYMBOLS,
        start_date=WINDOW_START,
        end_date=WINDOW_END,
    )
    return _AcquiredInputs(
        trade_calendar=trade_calendar,
        stock_batches=stock_batches,
        daily_bars=tuple(daily_bars),
        adjustment_factors=tuple(adjustment_factors),
        suspension_records=suspension_records,
    )


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"M1 runtime directory cannot be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"M1 runtime path must be a directory: {path}")


def _write_new_fsynced(path: Path, content: bytes) -> None:
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


def _write_success_evidence(
    *,
    spec: M1RunSpec,
    temporary_run: Path,
    evidence_documents: dict[str, bytes],
    prepared_dataset_id: str,
    published: PublishedDatasetPaths,
    auxiliary_artifacts: tuple[AuxiliaryArtifact, ...],
) -> None:
    """Write the success receipt while the whole run is still hidden."""
    evidence_index: list[dict[str, object]] = []
    for filename, content in sorted(evidence_documents.items()):
        _write_new_fsynced(temporary_run / filename, content)
        evidence_index.append(
            {
                "path": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
                "file_size_bytes": len(content),
            }
        )
    receipt = canonical_json_bytes(
        {
            "format": "ashare-pilot-m1-run/v1",
            "status": "PASS",
            "run_id": spec.run_id,
            "generated_at": spec.generated_at.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "window_start": WINDOW_START.isoformat(),
            "window_end": WINDOW_END.isoformat(),
            "as_of": DATASET_AS_OF.isoformat(),
            "audit_symbols": list(AUDIT_SYMBOLS),
            "dataset_id": prepared_dataset_id,
            "dataset_manifest_sha256": hashlib.sha256(
                published.manifest_path.read_bytes()
            ).hexdigest(),
            "evidence_files": evidence_index,
            "auxiliary_files": [
                {
                    "path": artifact.relative_path,
                    "sha256": artifact.sha256,
                    "file_size_bytes": len(artifact.content),
                }
                for artifact in auxiliary_artifacts
            ],
        }
    )
    _write_new_fsynced(temporary_run / "run-receipt.json", receipt)
    _fsync_directory(temporary_run)


def _read_canonical_object(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"M1 staged evidence is not valid JSON: {path.name}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"M1 staged evidence must be an object: {path.name}")
    if canonical_json_bytes(document) != raw:
        raise ValueError(f"M1 staged evidence is not canonical: {path.name}")
    return document


def _validate_coverage_evidence(document: dict[str, object]) -> None:
    if document.get("contract_id") != "coverage-audit":
        raise ValueError("M1 staged coverage has the wrong contract_id")
    if document.get("schema_version") != "2.0.0":
        raise ValueError("M1 staged coverage has the wrong schema_version")
    if document.get("passed") is not True:
        raise ValueError("M1 cannot commit a failed coverage audit")
    if document.get("missing_member_days") != []:
        raise ValueError("M1 staged coverage contains missing member-days")
    if document.get("silent_skip_symbols") != []:
        raise ValueError("M1 staged coverage contains silent skips")
    if document.get("reason_codes") != []:
        raise ValueError("M1 staged coverage contains failure reasons")
    expected_delisted = document.get("expected_delisted_member_days")
    if not isinstance(expected_delisted, list):
        raise ValueError("M1 staged expected-delisted rows are invalid")
    counts = (
        document.get("expected_member_days"),
        document.get("bar_member_days"),
        document.get("suspended_member_days"),
        document.get("expected_delisted_member_day_count"),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise ValueError("M1 staged coverage counts are invalid")
    expected, bars, suspensions, expected_delisted_count = counts
    assert isinstance(expected, int)
    assert isinstance(bars, int)
    assert isinstance(suspensions, int)
    assert isinstance(expected_delisted_count, int)
    if expected_delisted_count != len(expected_delisted):
        raise ValueError("M1 staged expected-delisted count does not reconcile")
    if expected != bars + suspensions + expected_delisted_count:
        raise ValueError("M1 staged coverage member-days do not reconcile")


def _validate_stock_evidence(document: dict[str, object]) -> None:
    if set(document) != {"L", "D", "P"}:
        raise ValueError("M1 staged stock reconciliation must contain L/D/P")
    for status in ("L", "D", "P"):
        batch = document[status]
        if not isinstance(batch, dict):
            raise ValueError("M1 staged stock reconciliation batch is invalid")
        counts = (
            batch.get("raw_rows"),
            batch.get("accepted_rows"),
            batch.get("rejected_rows"),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("M1 staged stock reconciliation counts are invalid")
        raw, accepted, rejected = counts
        assert isinstance(raw, int)
        assert isinstance(accepted, int)
        assert isinstance(rejected, int)
        rejections = batch.get("rejections")
        if not isinstance(rejections, list) or len(rejections) != rejected:
            raise ValueError("M1 staged stock rejection details do not reconcile")
        if raw != accepted + rejected:
            raise ValueError("M1 staged stock rows do not reconcile")


def _validate_telemetry_evidence(document: dict[str, object]) -> None:
    counters = (
        document.get("logical_calls"),
        document.get("pages"),
        document.get("http_attempts"),
        document.get("http_successes"),
        document.get("http_failures"),
        document.get("retries"),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counters
    ):
        raise ValueError("M1 staged acquisition counters are invalid")
    attempts = document["http_attempts"]
    successes = document["http_successes"]
    failures = document["http_failures"]
    assert isinstance(attempts, int)
    assert isinstance(successes, int)
    assert isinstance(failures, int)
    if attempts != successes + failures:
        raise ValueError("M1 staged acquisition attempts do not reconcile")
    host = document.get("base_url_host")
    if not isinstance(host, str) or not host or host != host.lower():
        raise ValueError("M1 staged source host is invalid")
    official = document.get("is_official_vendor")
    if official is not (host == "api.tushare.pro"):
        raise ValueError("M1 staged official-vendor flag does not match source host")
    alignment = document.get("daily_adj_alignment")
    if not isinstance(alignment, dict):
        raise ValueError("M1 staged daily/adj alignment is missing")
    for field in ("daily_only_keys", "adj_factor_only_keys", "matched_keys"):
        value = alignment.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("M1 staged daily/adj alignment count is invalid")


def _validate_staged_m1_run(
    *,
    temporary_run: Path,
    dataset_id: str,
    evidence_documents: dict[str, bytes],
    auxiliary_artifacts: tuple[AuxiliaryArtifact, ...],
) -> None:
    """Re-read the staged dataset, evidence and receipt before one atomic move."""
    loaded = load_published_dataset(
        publication_root=temporary_run,
        dataset_id=dataset_id,
    )
    if loaded.dataset_id != dataset_id:
        raise ValueError("M1 staged dataset identity changed during validation")
    if load_published_auxiliary_artifacts(
        publication_root=temporary_run,
        dataset_id=dataset_id,
    ) != auxiliary_artifacts:
        raise ValueError("M1 staged auxiliary evidence changed during validation")

    for filename, expected_bytes in sorted(evidence_documents.items()):
        path = temporary_run / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"M1 staged evidence is missing or invalid: {filename}")
        if path.read_bytes() != expected_bytes:
            raise ValueError(f"M1 staged evidence changed after write: {filename}")
        _read_canonical_object(path)

    _validate_coverage_evidence(
        _read_canonical_object(temporary_run / "coverage-audit.json")
    )
    _validate_stock_evidence(
        _read_canonical_object(temporary_run / "stock-basic-reconciliation.json")
    )
    _validate_telemetry_evidence(
        _read_canonical_object(temporary_run / "acquisition-telemetry.json")
    )
    quality = _read_canonical_object(temporary_run / "quality-summary.json")
    if quality.get("quality_status") != "pass":
        raise ValueError("M1 staged quality evidence did not pass")

    receipt = _read_canonical_object(temporary_run / "run-receipt.json")
    if receipt.get("status") != "PASS" or receipt.get("dataset_id") != dataset_id:
        raise ValueError("M1 staged receipt identity or status is invalid")
    manifest_path = temporary_run / "manifests" / f"{dataset_id}.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if receipt.get("dataset_manifest_sha256") != manifest_hash:
        raise ValueError("M1 staged receipt manifest hash does not match")

    expected_index = [
        {
            "path": filename,
            "sha256": hashlib.sha256(content).hexdigest(),
            "file_size_bytes": len(content),
        }
        for filename, content in sorted(evidence_documents.items())
    ]
    if receipt.get("evidence_files") != expected_index:
        raise ValueError("M1 staged receipt evidence index does not match")
    expected_auxiliary = [
        {
            "path": artifact.relative_path,
            "sha256": artifact.sha256,
            "file_size_bytes": len(artifact.content),
        }
        for artifact in auxiliary_artifacts
    ]
    if receipt.get("auxiliary_files") != expected_auxiliary:
        raise ValueError("M1 staged receipt auxiliary index does not match")


def _failure_details(exc: BaseException) -> dict[str, object]:
    details: dict[str, object] = {
        "error_type": type(exc).__name__,
        "reason_code": "M1_RUN_FAILED",
    }
    if isinstance(exc, NormalizationError):
        details["reason_code"] = exc.reason_code
        details["normalization_issue"] = {
            "reason_code": exc.reason_code,
            "symbol": exc.symbol,
            "trade_date": (
                exc.trade_date.isoformat() if exc.trade_date is not None else None
            ),
        }
    elif isinstance(exc, TushareAPIError):
        details["reason_code"] = "VENDOR_API_ERROR"
        details["vendor_code"] = exc.code
    elif isinstance(exc, TushareTransientError):
        details["reason_code"] = "VENDOR_TRANSIENT_ERROR"
    elif isinstance(exc, FileExistsError):
        details["reason_code"] = "IMMUTABLE_PATH_CONFLICT"
    elif isinstance(exc, ValueError):
        details["reason_code"] = "VALIDATION_FAILED"
    return details


def _commit_failed_run(
    *,
    spec: M1RunSpec,
    final_failure: Path,
    exc: BaseException,
    counting_transport: CountingTransport | None,
    partial_evidence: dict[str, bytes],
) -> None:
    """Atomically retain sanitized evidence, never a consumer Manifest."""
    failures_root = final_failure.parent
    temporary_failure = Path(
        tempfile.mkdtemp(prefix=f".{spec.run_id}.tmp-", dir=failures_root)
    )
    try:
        documents = dict(partial_evidence)
        details = _failure_details(exc)
        if counting_transport is not None:
            documents.setdefault(
                "acquisition-telemetry.json",
                canonical_json_bytes(counting_transport.snapshot().to_dict()),
            )
            documents["raw-responses.json"] = _raw_response_bytes(
                counting_transport.raw_snapshot()
            )
        if "quality-summary.json" not in documents:
            issue = details.get("normalization_issue")
            documents["quality-summary.json"] = canonical_json_bytes(
                {
                    "quality_status": "fail",
                    "reason_codes": [details["reason_code"]],
                    "normalization_issue_count": 1 if issue is not None else 0,
                    "normalization_issues": [issue] if issue is not None else [],
                }
            )

        evidence_index: list[dict[str, object]] = []
        for filename, content in sorted(documents.items()):
            _write_new_fsynced(temporary_failure / filename, content)
            evidence_index.append(
                {
                    "path": filename,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "file_size_bytes": len(content),
                }
            )
        _write_new_fsynced(
            temporary_failure / "failure-receipt.json",
            canonical_json_bytes(
                {
                    "format": "ashare-pilot-m1-run/v1",
                    "status": "FAIL",
                    "run_id": spec.run_id,
                    "generated_at": spec.generated_at.astimezone(UTC)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "window_start": WINDOW_START.isoformat(),
                    "window_end": WINDOW_END.isoformat(),
                    "as_of": DATASET_AS_OF.isoformat(),
                    "audit_symbols": list(AUDIT_SYMBOLS),
                    **details,
                    "evidence_files": evidence_index,
                }
            ),
        )
        for entry in evidence_index:
            path = temporary_failure / str(entry["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("M1 failed-run evidence hash changed before commit")
            _read_canonical_object(path)
        _read_canonical_object(temporary_failure / "failure-receipt.json")
        _fsync_directory(temporary_failure)
        os.replace(temporary_failure, final_failure)
        _fsync_directory(failures_root)
    except BaseException:
        shutil.rmtree(temporary_failure, ignore_errors=True)
        raise


def _execute_locked(
    *,
    spec: M1RunSpec,
    transport: Transport,
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    runs_root: Path,
    failures_root: Path,
) -> M1RunResult:
    final_run = runs_root / spec.run_id
    final_failure = failures_root / spec.run_id
    if (
        final_run.exists()
        or final_run.is_symlink()
        or final_failure.exists()
        or final_failure.is_symlink()
    ):
        raise FileExistsError(f"M1 run already exists: {spec.run_id}")
    temporary_run = Path(tempfile.mkdtemp(prefix=f".{spec.run_id}.tmp-", dir=runs_root))
    counting_transport: CountingTransport | None = None
    partial_evidence: dict[str, bytes] = {}
    run_committed = False
    try:
        counting_transport = CountingTransport(transport, monotonic_fn=monotonic_fn)
        client = TushareClient(counting_transport, sleep_fn=sleep_fn)
        acquired = _acquire(client)
        telemetry = counting_transport.snapshot()
        alignment = _alignment_document(
            acquired.daily_bars,
            acquired.adjustment_factors,
        )
        telemetry_document = telemetry.to_dict()
        telemetry_document["daily_adj_alignment"] = alignment
        partial_evidence["acquisition-telemetry.json"] = canonical_json_bytes(
            telemetry_document
        )

        open_days = _validate_trade_calendar(acquired.trade_calendar)
        stock_by_symbol, stock_evidence = _reconcile_stock_basic(acquired.stock_batches)
        partial_evidence["stock-basic-reconciliation.json"] = canonical_json_bytes(
            stock_evidence
        )
        _validate_daily_market_days_and_lifecycles(
            acquired.daily_bars,
            open_days=open_days,
            stock_by_symbol=stock_by_symbol,
        )

        open_day_set = frozenset(open_days)
        suspension_symbols = {
            record.ts_code
            for record in acquired.suspension_records
            if record.suspend_type == "S"
            and record.trade_date in open_day_set
            and stock_by_symbol[record.ts_code].list_date is not None
            and record.trade_date >= stock_by_symbol[record.ts_code].list_date
            and (
                stock_by_symbol[record.ts_code].delist_date is None
                or record.trade_date <= stock_by_symbol[record.ts_code].delist_date
            )
        }
        missing_long_suspensions = sorted(
            set(LONG_SUSPENDED_SYMBOLS) - suspension_symbols
        )
        if missing_long_suspensions:
            raise ValueError(
                "M1 suspend_d lacks open member-day S evidence for frozen "
                f"long-suspended symbols: {missing_long_suspensions}"
            )

        normalized = normalize_daily_bars(
            acquired.daily_bars,
            requested_symbols=AUDIT_SYMBOLS,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        )
        memberships = tuple(
            UniverseMembership(
                symbol=symbol,
                valid_from=WINDOW_START,
                valid_to=WINDOW_END,
            )
            for symbol in AUDIT_SYMBOLS
        )
        lifecycles = tuple(
            SecurityLifecycle(
                symbol=symbol,
                listed_on=stock_by_symbol[symbol].list_date,
                delisted_on=stock_by_symbol[symbol].delist_date,
            )
            for symbol in AUDIT_SYMBOLS
        )
        coverage = audit_historical_coverage(
            audit_id=f"{spec.run_id}-coverage",
            universe_policy_id="m1-fixed-audit-selection",
            universe_policy_version="v1",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            generated_at=spec.generated_at,
            trading_days=open_days,
            memberships=memberships,
            lifecycles=lifecycles,
            bar_keys=frozenset((row.symbol, row.trade_date) for row in normalized),
            suspension_keys=frozenset(
                (row.ts_code, row.trade_date)
                for row in acquired.suspension_records
                if row.suspend_type == "S"
            ),
            expected_member_count=None,
            provenance_warnings=telemetry.provenance_warnings,
        )
        partial_evidence["coverage-audit.json"] = canonical_json_bytes(
            coverage.document
        )
        partial_evidence["quality-summary.json"] = canonical_json_bytes(
            {
                "quality_status": "pass" if coverage.passed else "fail",
                "normalization_issue_count": 0,
                "coverage_passed": coverage.passed,
                "normalized_row_count": len(normalized),
                "reason_codes": list(coverage.reason_codes),
            }
        )
        if not coverage.passed:
            raise ValueError(
                "M1 coverage audit failed: " + ",".join(coverage.reason_codes)
            )

        raw_artifact = AuxiliaryArtifact(
            relative_path="raw/vendor-responses.json",
            content=_raw_response_bytes(counting_transport.raw_snapshot()),
        )
        factor_artifact = AuxiliaryArtifact(
            relative_path="adj_factor/records.json",
            content=_adj_factor_bytes(acquired.adjustment_factors),
        )
        auxiliary_artifacts = tuple(sorted((raw_artifact, factor_artifact)))
        prepared = prepare_normalized_dataset(
            normalized,
            as_of=DATASET_AS_OF,
            generated_at=spec.generated_at,
            source_identity=counting_transport.source_identity,
            source_version=SOURCE_VERSION,
            auxiliary_artifacts=auxiliary_artifacts,
        )
        if prepared.manifest["source"] != telemetry.dataset_source:
            raise ValueError("M1 manifest source disagrees with transport provenance")

        published = publish_normalized_dataset(
            publication_root=temporary_run,
            prepared=prepared,
        )
        if published.auxiliary_dir is None:
            raise ValueError("M1 publisher omitted required auxiliary evidence")
        _write_success_evidence(
            spec=spec,
            temporary_run=temporary_run,
            evidence_documents=partial_evidence,
            prepared_dataset_id=prepared.dataset_id,
            published=published,
            auxiliary_artifacts=auxiliary_artifacts,
        )
        _validate_staged_m1_run(
            temporary_run=temporary_run,
            dataset_id=prepared.dataset_id,
            evidence_documents=partial_evidence,
            auxiliary_artifacts=auxiliary_artifacts,
        )
        dataset_relative = published.dataset_dir.relative_to(temporary_run)
        manifest_relative = published.manifest_path.relative_to(temporary_run)
        auxiliary_relative = published.auxiliary_dir.relative_to(temporary_run)
        os.replace(temporary_run, final_run)
        run_committed = True
        _fsync_directory(final_run.parent)
        return M1RunResult(
            dataset_id=prepared.dataset_id,
            dataset_dir=final_run / dataset_relative,
            manifest_path=final_run / manifest_relative,
            auxiliary_dir=final_run / auxiliary_relative,
            run_dir=final_run,
            normalized_row_count=len(normalized),
            coverage_expected_member_days=coverage.expected_member_days,
            http_attempts=telemetry.http_attempts,
        )
    except BaseException as exc:
        if run_committed:
            raise
        try:
            _commit_failed_run(
                spec=spec,
                final_failure=final_failure,
                exc=exc,
                counting_transport=counting_transport,
                partial_evidence=partial_evidence,
            )
        finally:
            shutil.rmtree(temporary_run, ignore_errors=True)
        raise


def run_m1(
    spec: M1RunSpec,
    *,
    transport: Transport,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> M1RunResult:
    """Acquire and publish the frozen M1 slice through an actual transport."""
    if not isinstance(spec, M1RunSpec):
        raise TypeError("spec must be M1RunSpec")

    _ensure_directory(spec.output_root)
    runs_root = spec.output_root / "runs"
    failures_root = spec.output_root / "failures"
    _ensure_directory(runs_root)
    _ensure_directory(failures_root)
    lock_path = runs_root / f".{spec.run_id}.lock"
    if lock_path.is_symlink():
        raise ValueError("M1 run lock cannot be a symlink")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        return _execute_locked(
            spec=spec,
            transport=transport,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            runs_root=runs_root,
            failures_root=failures_root,
        )


def _parse_generated_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("generated_at must be an ISO datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise argparse.ArgumentTypeError("generated_at must use UTC")
    return parsed.astimezone(UTC)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the frozen M1 real-data acquisition")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generated-at", required=True, type=_parse_generated_at)
    args = parser.parse_args(argv)
    result = run_m1(
        M1RunSpec(
            output_root=args.output_root,
            run_id=args.run_id,
            generated_at=args.generated_at,
        ),
        transport=HttpJsonTransport(),
    )
    print(canonical_json_bytes(result.document).decode("utf-8"))


if __name__ == "__main__":
    main()
