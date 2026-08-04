"""Composable, token-safe acquisition telemetry for Tushare transports.

The decorator counts the actual ``Transport.send`` boundary. It does not
change retry, pagination, HTTP, or vendor-error behavior. TushareClient calls
``begin_logical_call`` and ``begin_page`` when those optional hooks are present,
which makes logical-call, page, and retry counts exact while preserving the
plain Transport protocol for all other implementations.

Only an injected monotonic clock is read. No wall clock is consulted and no
telemetry field participates in dataset identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

from .dataset_publication import DatasetSourceIdentity, source_identity_from_base_url
from .tushare_transport import (
    SecretToken,
    Transport,
    TransportRequest,
    TransportResponse,
    TushareTransientError,
)

OFFICIAL_TUSHARE_HOST = "api.tushare.pro"
NON_OFFICIAL_VENDOR_WARNING = "NON_OFFICIAL_VENDOR_ENDPOINT"

_REDACTED = "***"
_SENSITIVE_PARAM_PARTS = ("token", "secret", "password", "credential", "auth")
_RATE_LIMIT_REASONS = frozenset({"http_429", "rate_limit", "vendor_code_-2000"})


@dataclass(frozen=True, order=True)
class ApiAcquisitionTelemetry:
    """Immutable counters for one vendor API name."""

    api_name: str
    logical_calls: int
    pages: int
    http_attempts: int
    http_successes: int
    http_failures: int
    retries: int
    transient_errors: int
    rate_limit_errors: int
    other_transient_errors: int

    def __post_init__(self) -> None:
        counters = (
            self.logical_calls,
            self.pages,
            self.http_attempts,
            self.http_successes,
            self.http_failures,
            self.retries,
            self.transient_errors,
            self.rate_limit_errors,
            self.other_transient_errors,
        )
        if not self.api_name:
            raise ValueError("api telemetry: api_name cannot be empty")
        if any(value < 0 for value in counters):
            raise ValueError("api telemetry: counters cannot be negative")
        if self.http_attempts != self.http_successes + self.http_failures:
            raise ValueError("api telemetry: attempts must equal successes + failures")
        if self.transient_errors != (
            self.rate_limit_errors + self.other_transient_errors
        ):
            raise ValueError(
                "api telemetry: transient errors must equal rate-limit + other transient"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "api_name": self.api_name,
            "logical_calls": self.logical_calls,
            "pages": self.pages,
            "http_attempts": self.http_attempts,
            "http_successes": self.http_successes,
            "http_failures": self.http_failures,
            "retries": self.retries,
            "transient_errors": self.transient_errors,
            "rate_limit_errors": self.rate_limit_errors,
            "other_transient_errors": self.other_transient_errors,
        }


@dataclass(frozen=True)
class RequestEvidence:
    """Sanitized evidence for one actual transport attempt.

    Response rows, vendor messages, exception messages, base URLs, and the
    authentication token are deliberately excluded.
    """

    sequence: int
    api_name: str
    params: tuple[tuple[str, str], ...]
    requested_fields: tuple[str, ...]
    offset: int
    limit: int
    elapsed_seconds: float
    outcome: str
    response_code: int | None
    response_field_count: int
    response_row_count: int
    error_kind: str | None
    transient: bool
    rate_limited: bool

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("request evidence: sequence must be positive")
        if self.elapsed_seconds < 0 or not math.isfinite(self.elapsed_seconds):
            raise ValueError("request evidence: elapsed_seconds must be finite and non-negative")
        if self.outcome not in ("SUCCESS", "FAILURE"):
            raise ValueError(f"request evidence: invalid outcome {self.outcome!r}")
        if self.outcome == "SUCCESS" and self.error_kind is not None:
            raise ValueError("request evidence: successful attempt cannot have error_kind")
        if self.outcome == "FAILURE" and self.error_kind is None:
            raise ValueError("request evidence: failed attempt must have error_kind")
        if self.rate_limited and not self.transient:
            raise ValueError("request evidence: rate limit must also be transient")

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "api_name": self.api_name,
            "params": [list(item) for item in self.params],
            "requested_fields": list(self.requested_fields),
            "offset": self.offset,
            "limit": self.limit,
            "elapsed_seconds": self.elapsed_seconds,
            "outcome": self.outcome,
            "response_code": self.response_code,
            "response_field_count": self.response_field_count,
            "response_row_count": self.response_row_count,
            "error_kind": self.error_kind,
            "transient": self.transient,
            "rate_limited": self.rate_limited,
        }


@dataclass(frozen=True)
class RawResponseSnapshot:
    """One successful raw vendor response bound to its sanitized request.

    This surface is intentionally separate from compact telemetry so the M1
    orchestrator can persist exact vendor fields/items under ``runtime/raw``.
    The vendor message is omitted because it is uncontrolled diagnostic text.
    """

    sequence: int
    api_name: str
    params: tuple[tuple[str, str], ...]
    requested_fields: tuple[str, ...]
    offset: int
    limit: int
    response_code: int
    response_fields: tuple[str, ...]
    response_items: tuple[tuple[object, ...], ...]

    def to_dict(self) -> dict[str, object]:
        """Return an order-preserving JSON-ready representation."""
        return {
            "sequence": self.sequence,
            "request": {
                "api_name": self.api_name,
                "params": [list(item) for item in self.params],
                "fields": list(self.requested_fields),
                "offset": self.offset,
                "limit": self.limit,
            },
            "response": {
                "code": self.response_code,
                "fields": list(self.response_fields),
                "items": [list(row) for row in self.response_items],
            },
        }


@dataclass(frozen=True)
class AcquisitionTelemetry:
    """Immutable snapshot of one decorated transport's acquisition evidence."""

    base_url_host: str
    is_official_vendor: bool
    dataset_source: str
    provenance_warnings: tuple[str, ...]
    total_elapsed_seconds: float
    logical_calls: int
    pages: int
    http_attempts: int
    http_successes: int
    http_failures: int
    retries: int
    transient_errors: int
    rate_limit_errors: int
    other_transient_errors: int
    per_api: tuple[ApiAcquisitionTelemetry, ...]
    requests: tuple[RequestEvidence, ...]

    def __post_init__(self) -> None:
        if not self.base_url_host or self.base_url_host != self.base_url_host.lower():
            raise ValueError("acquisition telemetry: base_url_host must be lowercase")
        if self.is_official_vendor != (self.base_url_host == OFFICIAL_TUSHARE_HOST):
            raise ValueError("acquisition telemetry: official-vendor flag disagrees with host")
        expected_source = (
            "tushare-official"
            if self.is_official_vendor
            else "tushare-via-non-official-endpoint"
        )
        if self.dataset_source != expected_source:
            raise ValueError("acquisition telemetry: dataset_source disagrees with provenance")
        expected_warnings = (
            () if self.is_official_vendor else (NON_OFFICIAL_VENDOR_WARNING,)
        )
        if self.provenance_warnings != expected_warnings:
            raise ValueError("acquisition telemetry: provenance warnings disagree with host")
        if self.total_elapsed_seconds < 0 or not math.isfinite(
            self.total_elapsed_seconds
        ):
            raise ValueError(
                "acquisition telemetry: total_elapsed_seconds must be finite and non-negative"
            )
        if self.http_attempts != self.http_successes + self.http_failures:
            raise ValueError(
                "acquisition telemetry: attempts must equal successes + failures"
            )
        if self.transient_errors != (
            self.rate_limit_errors + self.other_transient_errors
        ):
            raise ValueError(
                "acquisition telemetry: transient errors must equal rate-limit + other transient"
            )
        if self.http_attempts != len(self.requests):
            raise ValueError("acquisition telemetry: each attempt must have request evidence")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready representation without vendor data or secrets."""
        return {
            "base_url_host": self.base_url_host,
            "is_official_vendor": self.is_official_vendor,
            "dataset_source": self.dataset_source,
            "provenance_warnings": list(self.provenance_warnings),
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "logical_calls": self.logical_calls,
            "pages": self.pages,
            "http_attempts": self.http_attempts,
            "http_successes": self.http_successes,
            "http_failures": self.http_failures,
            "retries": self.retries,
            "transient_errors": self.transient_errors,
            "rate_limit_errors": self.rate_limit_errors,
            "other_transient_errors": self.other_transient_errors,
            "per_api": [metrics.to_dict() for metrics in self.per_api],
            "requests": [request.to_dict() for request in self.requests],
        }


@dataclass
class _MutableApiCounters:
    logical_calls: int = 0
    pages: int = 0
    http_attempts: int = 0
    http_successes: int = 0
    http_failures: int = 0
    retries: int = 0
    transient_errors: int = 0
    rate_limit_errors: int = 0


def _request_key(request: TransportRequest) -> str:
    document = json.dumps(
        {
            "api_name": request.api_name,
            "params": request.params,
            "fields": request.fields,
            "offset": request.offset,
            "limit": request.limit,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def _sanitize_request(
    request: TransportRequest, *, token: str
) -> tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]]:
    def sensitive_name(name: str) -> bool:
        lowered = name.lower()
        return any(part in lowered for part in _SENSITIVE_PARAM_PARTS)

    api_name = _REDACTED if request.api_name == token else request.api_name
    params = tuple(
        (
            name,
            _REDACTED if sensitive_name(name) or value == token else value,
        )
        for name, value in request.params
    )
    fields = tuple(_REDACTED if field == token else field for field in request.fields)
    return api_name, params, fields


def _source_identity_from_transport(transport: Transport) -> DatasetSourceIdentity:
    base_url = getattr(transport, "base_url", None)
    del transport
    if not isinstance(base_url, str) or not base_url:
        del base_url
        raise ValueError(
            "counting transport requires wrapped transport.base_url for provenance"
        )
    try:
        identity = source_identity_from_base_url(base_url)
    finally:
        del base_url
    return identity


def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, TushareTransientError) and exc.reason in _RATE_LIMIT_REASONS


def _call_wrapped_transport(
    transport: Transport,
    request: TransportRequest,
    secret: SecretToken,
) -> TransportResponse:
    """Call the wrapped transport without retaining a raw token on traceback."""
    raw_token = secret.reveal()
    del secret
    try:
        response = transport.send(request, token=raw_token)
    except BaseException:
        del raw_token, transport
        raise
    del raw_token
    return response


class CountingTransport:
    """Transport decorator recording exact attempts and sanitized evidence.

    ``monotonic_fn`` is required so callers control the only time source. The
    wrapped transport must expose its actual resolved ``base_url``. It is parsed
    once and discarded; only a sanitized source identity is retained.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        monotonic_fn: Callable[[], float],
    ) -> None:
        try:
            source_identity = _source_identity_from_transport(transport)
        except BaseException:
            del transport
            raise
        self._transport = transport
        self._monotonic_fn = monotonic_fn
        self._source_identity = source_identity
        self._api: dict[str, _MutableApiCounters] = {}
        self._active_page_attempts: dict[str, int] = {}
        self._requests: list[RequestEvidence] = []
        self._raw_responses: list[RawResponseSnapshot] = []
        self._run_started: float | None = None
        self._last_clock: float | None = None
        self._sequence = 0

    @property
    def source_identity(self) -> DatasetSourceIdentity:
        """Sanitized identity derived from the wrapped transport configuration."""
        return self._source_identity

    @property
    def base_url_host(self) -> str:
        return self._source_identity.base_url_host

    @property
    def is_official_vendor(self) -> bool:
        return self._source_identity.is_official_vendor

    @property
    def dataset_source(self) -> str:
        return self._source_identity.manifest_source

    @property
    def provenance_warnings(self) -> tuple[str, ...]:
        return () if self.is_official_vendor else (NON_OFFICIAL_VENDOR_WARNING,)

    def _read_monotonic(self) -> float:
        value = self._monotonic_fn()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("monotonic clock must return a number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("monotonic clock must return a finite number")
        if self._last_clock is not None and result < self._last_clock:
            raise ValueError("monotonic clock moved backwards")
        self._last_clock = result
        return result

    def _start_run_if_needed(self) -> None:
        if self._run_started is None:
            self._run_started = self._read_monotonic()

    def _counters(self, api_name: str) -> _MutableApiCounters:
        if not isinstance(api_name, str) or not api_name:
            raise ValueError("api_name cannot be empty")
        return self._api.setdefault(api_name, _MutableApiCounters())

    def begin_logical_call(self, api_name: str) -> None:
        """Mark one high-level client method call before pagination begins."""
        self._start_run_if_needed()
        self._counters(api_name).logical_calls += 1

    def begin_page(self, request: TransportRequest) -> None:
        """Mark one logical page; following sends for it are retries."""
        self._start_run_if_needed()
        self._counters(request.api_name).pages += 1
        self._active_page_attempts[_request_key(request)] = 0

    def send(self, request: TransportRequest, *, token: str) -> TransportResponse:
        safe_api_name, safe_params, safe_fields = _sanitize_request(request, token=token)
        secret = SecretToken(token)
        del token

        try:
            self._start_run_if_needed()
            started = self._read_monotonic()
        except BaseException:
            del secret
            raise

        counters = self._counters(request.api_name)
        key = _request_key(request)
        if key not in self._active_page_attempts:
            # Generic Transport callers do not know the optional client hooks.
            counters.pages += 1
            if request.offset == 0:
                counters.logical_calls += 1
            self._active_page_attempts[key] = 0
        prior_attempts = self._active_page_attempts[key]
        if prior_attempts:
            counters.retries += 1
        self._active_page_attempts[key] = prior_attempts + 1
        counters.http_attempts += 1
        self._sequence += 1
        sequence = self._sequence

        try:
            response = _call_wrapped_transport(self._transport, request, secret)
        except BaseException as exc:
            del secret
            finished = self._read_monotonic()
            counters.http_failures += 1
            transient = isinstance(exc, TushareTransientError)
            rate_limited = _is_rate_limit_error(exc)
            if transient:
                counters.transient_errors += 1
                if rate_limited:
                    counters.rate_limit_errors += 1
            else:
                self._active_page_attempts.pop(key, None)
            self._requests.append(
                RequestEvidence(
                    sequence=sequence,
                    api_name=safe_api_name,
                    params=safe_params,
                    requested_fields=safe_fields,
                    offset=request.offset,
                    limit=request.limit,
                    elapsed_seconds=finished - started,
                    outcome="FAILURE",
                    response_code=None,
                    response_field_count=0,
                    response_row_count=0,
                    error_kind=type(exc).__name__,
                    transient=transient,
                    rate_limited=rate_limited,
                )
            )
            raise

        del secret
        finished = self._read_monotonic()
        counters.http_successes += 1
        self._active_page_attempts.pop(key, None)
        self._requests.append(
            RequestEvidence(
                sequence=sequence,
                api_name=safe_api_name,
                params=safe_params,
                requested_fields=safe_fields,
                offset=request.offset,
                limit=request.limit,
                elapsed_seconds=finished - started,
                outcome="SUCCESS",
                response_code=response.code,
                response_field_count=len(response.fields),
                response_row_count=len(response.items),
                error_kind=None,
                transient=False,
                rate_limited=False,
            )
        )
        self._raw_responses.append(
            RawResponseSnapshot(
                sequence=sequence,
                api_name=safe_api_name,
                params=safe_params,
                requested_fields=safe_fields,
                offset=request.offset,
                limit=request.limit,
                response_code=response.code,
                response_fields=response.fields,
                response_items=response.items,
            )
        )
        return response

    def raw_snapshot(self) -> tuple[RawResponseSnapshot, ...]:
        """Return successful raw responses in actual attempt order."""
        return tuple(self._raw_responses)

    def snapshot(self) -> AcquisitionTelemetry:
        """Freeze current counters using the injected monotonic clock."""
        if self._run_started is None:
            total_elapsed = 0.0
        else:
            total_elapsed = self._read_monotonic() - self._run_started

        per_api = tuple(
            ApiAcquisitionTelemetry(
                api_name=api_name,
                logical_calls=counters.logical_calls,
                pages=counters.pages,
                http_attempts=counters.http_attempts,
                http_successes=counters.http_successes,
                http_failures=counters.http_failures,
                retries=counters.retries,
                transient_errors=counters.transient_errors,
                rate_limit_errors=counters.rate_limit_errors,
                other_transient_errors=(
                    counters.transient_errors - counters.rate_limit_errors
                ),
            )
            for api_name, counters in sorted(self._api.items())
        )

        return AcquisitionTelemetry(
            base_url_host=self.base_url_host,
            is_official_vendor=self.is_official_vendor,
            dataset_source=self.dataset_source,
            provenance_warnings=self.provenance_warnings,
            total_elapsed_seconds=total_elapsed,
            logical_calls=sum(item.logical_calls for item in per_api),
            pages=sum(item.pages for item in per_api),
            http_attempts=sum(item.http_attempts for item in per_api),
            http_successes=sum(item.http_successes for item in per_api),
            http_failures=sum(item.http_failures for item in per_api),
            retries=sum(item.retries for item in per_api),
            transient_errors=sum(item.transient_errors for item in per_api),
            rate_limit_errors=sum(item.rate_limit_errors for item in per_api),
            other_transient_errors=sum(item.other_transient_errors for item in per_api),
            per_api=per_api,
            requests=tuple(self._requests),
        )
