"""Trading-calendar freshness evidence for dataset publication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class FreshnessAssessment:
    expected_as_of: date
    latest_complete_date: date
    allowed_lag_sessions: int
    lag_sessions: int
    passed: bool
    reason_codes: tuple[str, ...]

    def stage_health_document(
        self,
        *,
        run_id: str,
        started_at: datetime,
        finished_at: datetime,
        dataset_manifest_sha256: str,
    ) -> dict[str, object]:
        if not run_id:
            raise ValueError("run_id cannot be empty")
        if not SHA256_PATTERN.fullmatch(dataset_manifest_sha256):
            raise ValueError("dataset_manifest_sha256 must be a lowercase SHA-256")
        for field_name, value in (
            ("started_at", started_at),
            ("finished_at", finished_at),
        ):
            if value.tzinfo is None or value.utcoffset() != timedelta(0):
                raise ValueError(f"{field_name} must be timezone-aware UTC")
        if finished_at < started_at:
            raise ValueError("finished_at cannot precede started_at")
        return {
            "contract_id": "stage-health",
            "schema_version": "1.0.0",
            "run_id": run_id,
            "stage": "data",
            "status": "pass" if self.passed else "fail",
            "started_at": started_at.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "finished_at": finished_at.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "dependency_hashes": {
                "dataset-manifest": dataset_manifest_sha256,
            },
            "reason_codes": list(self.reason_codes),
        }


def assess_dataset_freshness(
    *,
    expected_as_of: date,
    latest_complete_date: date,
    trading_days: tuple[date, ...],
    allowed_lag_sessions: int = 0,
) -> FreshnessAssessment:
    """Fail closed when a dataset misses too many completed trading sessions."""
    if allowed_lag_sessions < 0:
        raise ValueError("allowed_lag_sessions cannot be negative")
    ordered_days = tuple(sorted(set(trading_days)))
    if len(ordered_days) != len(trading_days):
        raise ValueError("trading_days must be unique")

    reasons: list[str] = []
    calendar = set(ordered_days)
    if expected_as_of not in calendar:
        reasons.append("EXPECTED_AS_OF_NOT_TRADING_DAY")
    if latest_complete_date not in calendar:
        reasons.append("LATEST_COMPLETE_NOT_TRADING_DAY")
    if latest_complete_date > expected_as_of:
        reasons.append("LATEST_COMPLETE_IN_FUTURE")
    lag_sessions = sum(
        latest_complete_date < trade_date <= expected_as_of
        for trade_date in ordered_days
    )
    if lag_sessions > allowed_lag_sessions:
        reasons.append("DATA_STALE")
    reason_codes = tuple(reasons)
    return FreshnessAssessment(
        expected_as_of=expected_as_of,
        latest_complete_date=latest_complete_date,
        allowed_lag_sessions=allowed_lag_sessions,
        lag_sessions=lag_sessions,
        passed=not reason_codes,
        reason_codes=reason_codes,
    )
