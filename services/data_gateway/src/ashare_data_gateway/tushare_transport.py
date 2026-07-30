"""Injectable transport abstraction for vendor HTTP access.

Uses only the standard library. No third-party HTTP or data dependencies.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

DEFAULT_TUSHARE_API_URL = "https://api.tushare.pro"
URL_ENV_VAR = "TUSHARE_HTTP_URL"
DEFAULT_TIMEOUT_SECONDS = 30

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _validate_base_url(url: str) -> None:
    """Reject plaintext-public URLs. Allow https or local http."""
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS:
        return
    raise ValueError(
        f"base_url must use https:// for non-local hosts, got: {url!r}"
    )


def resolve_base_url() -> str:
    """Return the vendor API base URL from TUSHARE_HTTP_URL or the default.

    Never logs or exposes the token; URL itself is not secret.
    """
    custom = os.environ.get(URL_ENV_VAR, "").strip()
    return custom if custom else DEFAULT_TUSHARE_API_URL


@dataclass(frozen=True)
class TransportRequest:
    """Immutable vendor API request descriptor."""

    api_name: str
    params: tuple[tuple[str, str], ...]
    fields: tuple[str, ...]
    offset: int
    limit: int


@dataclass(frozen=True)
class TransportResponse:
    """Immutable vendor API raw response."""

    code: int
    msg: str
    fields: tuple[str, ...]
    items: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class SecretToken:
    """Token wrapper that never leaks its value via repr/str/tracebacks."""

    _value: str

    def reveal(self) -> str:
        """Return the raw token for use in request serialization only."""
        return self._value

    def __repr__(self) -> str:
        return "SecretToken(***)"

    def __str__(self) -> str:
        return "***"


class TushareAPIError(Exception):
    """Raised when the vendor returns a non-recoverable error."""

    def __init__(self, code: int, msg: str) -> None:
        self.code = code
        self.msg = msg
        super().__init__(f"vendor error code={code} msg={msg}")


class TushareTransientError(Exception):
    """Raised for errors that may succeed on retry (rate-limit, timeout)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"transient vendor error: {reason}")


@runtime_checkable
class Transport(Protocol):
    """Protocol for injectable vendor transports (no real network in tests)."""

    def send(self, request: TransportRequest, *, token: str) -> TransportResponse: ...


def _build_payload(request: TransportRequest, token: SecretToken) -> bytes:
    """Serialize the request body. Isolated so token leaves the frame before IO."""
    params_dict: dict[str, object] = dict(request.params)
    params_dict["offset"] = request.offset
    params_dict["limit"] = request.limit
    return json.dumps(
        {
            "api_name": request.api_name,
            "token": token.reveal(),
            "params": params_dict,
            "fields": ",".join(request.fields),
        }
    ).encode("utf-8")


class HttpJsonTransport:
    """Standard-library HTTP JSON transport for the Tushare pro API.

    Base URL is resolved from TUSHARE_HTTP_URL env var (or explicit override).
    Token is sent in the request body only and never appears in exception
    messages produced by this class: the raw token and its wrapper are
    deleted from the send frame before any IO, and the serialized request
    object is deleted before the network call, so no frame on a raised
    exception path retains the token in its locals.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        resolved = base_url or resolve_base_url()
        _validate_base_url(resolved)
        self._base_url = resolved
        self._timeout_seconds = timeout_seconds

    @property
    def base_url(self) -> str:
        """Read-only access to the resolved base URL."""
        return self._base_url

    def send(self, request: TransportRequest, *, token: str) -> TransportResponse:
        secret = SecretToken(token)
        del token  # remove raw string from this frame immediately

        http_request = urllib.request.Request(
            self._base_url,
            data=_build_payload(request, secret),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        del secret  # SecretToken no longer needed after serialization

        # http_request.data contains the token; ensure the reference is gone
        # on every path before any exception propagates further.
        try:
            body = _perform_io(http_request, timeout_seconds=self._timeout_seconds)
        except Exception:
            del http_request
            raise
        del http_request

        try:
            document = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TushareAPIError(0, "invalid_json_response") from exc

        if not isinstance(document, dict):
            raise TushareAPIError(0, "response_not_object")

        code = document.get("code")
        if not isinstance(code, int):
            raise TushareAPIError(0, "missing_code_field")

        msg = document.get("msg") or ""
        if not isinstance(msg, str):
            msg = ""

        if code != 0:
            if _is_transient_code(code):
                raise TushareTransientError(f"vendor_code_{code}")
            raise TushareAPIError(code, _sanitize_vendor_msg(msg))

        data = document.get("data")
        if data is None:
            return TransportResponse(code=code, msg=msg, fields=(), items=())
        if not isinstance(data, dict):
            raise TushareAPIError(0, "data_not_object")

        raw_fields = data.get("fields")
        if not isinstance(raw_fields, list):
            raise TushareAPIError(0, "missing_fields_in_data")
        fields = tuple(str(f) for f in raw_fields)

        raw_items = data.get("items")
        if not isinstance(raw_items, list):
            raise TushareAPIError(0, "missing_items_in_data")

        items: list[tuple[object, ...]] = []
        for index, row in enumerate(raw_items):
            if not isinstance(row, list):
                raise TushareAPIError(0, f"malformed_row_at_index_{index}")
            items.append(tuple(row))

        return TransportResponse(code=code, msg=msg, fields=fields, items=tuple(items))


def _perform_io(
    http_request: urllib.request.Request, *, timeout_seconds: float
) -> bytes:
    """Issue the HTTP call.

    This frame inevitably holds the request (whose .data contains the token)
    while IO is in flight. On every error path the reference is deleted
    before raising, so no frame on the resulting traceback retains it.
    """
    try:
        with urllib.request.urlopen(http_request, timeout=timeout_seconds) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        code = exc.code
        del http_request, exc
        if code in (429, 500, 502, 503, 504):
            raise TushareTransientError(f"http_{code}") from None
        raise TushareAPIError(0, f"http_error_{code}") from None
    except TimeoutError:
        del http_request
        raise TushareTransientError("timeout") from None
    except urllib.error.URLError as exc:
        reason = str(exc.reason)
        del http_request, exc
        if "timed out" in reason:
            raise TushareTransientError("timeout") from None
        raise TushareAPIError(0, "connection_failed") from None
    except Exception:
        del http_request
        raise TushareAPIError(0, "unexpected_io_error") from None


_HEX_RUN = re.compile(r"[0-9a-fA-F]{8,}")


def _sanitize_vendor_msg(msg: str) -> str:
    """Bound and de-noise uncontrolled vendor messages for error text."""
    truncated = msg[:200]
    return _HEX_RUN.sub("***", truncated)


# Explicit vendor error codes known to be transient (retryable).
# -2000: Tushare rate-limit / frequency cap (documented).
# Fail-closed: any code NOT in this set is treated as permanent.
_TRANSIENT_CODES: frozenset[int] = frozenset({-2000})


def _is_transient_code(code: int) -> bool:
    """Return True only for explicitly known transient vendor codes."""
    return code in _TRANSIENT_CODES
