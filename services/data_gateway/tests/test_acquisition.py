"""Tests for compositional Tushare acquisition telemetry and raw snapshots."""

from __future__ import annotations

import json
import urllib.request
from datetime import date
from pathlib import Path

import pytest
from ashare_data_gateway.acquisition import (
    NON_OFFICIAL_VENDOR_WARNING,
    CountingTransport,
)
from ashare_data_gateway.tushare_client import TOKEN_ENV_VAR, TushareClient
from ashare_data_gateway.tushare_transport import (
    HttpJsonTransport,
    TransportRequest,
    TransportResponse,
    TushareAPIError,
    TushareTransientError,
)

_CANARY = "<<acquisition-canary-token>>"
_TRADE_CAL_FIELDS = ("exchange", "cal_date", "is_open", "pretrade_date")


class ManualClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeTransport:
    def __init__(
        self,
        responses: list[TransportResponse | Exception],
        *,
        base_url: str,
        clock: ManualClock,
    ) -> None:
        self.base_url = base_url
        self._responses = list(responses)
        self._clock = clock
        self.requests: list[TransportRequest] = []

    def send(self, request: TransportRequest, *, token: str) -> TransportResponse:
        assert token == _CANARY
        del token
        self.requests.append(request)
        self._clock.advance(0.25)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def response(items: list[list[object]]) -> TransportResponse:
    return TransportResponse(
        code=0,
        msg="unpersisted vendor message",
        fields=_TRADE_CAL_FIELDS,
        items=tuple(tuple(row) for row in items),
    )


def leaked_traceback_locals(exc_tb: object, canary: str) -> list[str]:
    test_file = Path(__file__).resolve()
    leaked: list[str] = []
    tb = exc_tb
    while tb is not None:
        frame = tb.tb_frame
        if Path(frame.f_code.co_filename).resolve() != test_file:
            for name, value in frame.f_locals.items():
                if canary in repr(value):
                    leaked.append(f"{frame.f_code.co_name}.{name}")
        tb = tb.tb_next
    return leaked


def test_retry_and_pagination_count_actual_send_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
    clock = ManualClock()
    inner = FakeTransport(
        [
            TushareTransientError("vendor_code_-2000"),
            response(
                [
                    ["SSE", "20260101", 1, None],
                    ["SSE", "20260102", 1, "20260101"],
                ]
            ),
            TushareTransientError("timeout"),
            response([["SSE", "20260103", 1, "20260102"]]),
        ],
        base_url="https://api.tushare.pro",
        clock=clock,
    )
    transport = CountingTransport(inner, monotonic_fn=clock)
    client = TushareClient(
        transport,
        page_size=2,
        max_retries=2,
        retry_base_delay=0.5,
        sleep_fn=clock.advance,
    )

    records = client.fetch_trade_cal(
        exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 3)
    )
    telemetry = transport.snapshot()

    assert len(records) == 3
    assert len(inner.requests) == 4
    assert telemetry.logical_calls == 1
    assert telemetry.pages == 2
    assert telemetry.http_attempts == 4
    assert telemetry.http_successes == 2
    assert telemetry.http_failures == 2
    assert telemetry.retries == 2
    assert telemetry.transient_errors == 2
    assert telemetry.rate_limit_errors == 1
    assert telemetry.other_transient_errors == 1
    assert telemetry.total_elapsed_seconds == 2.0
    assert len(telemetry.per_api) == 1
    assert telemetry.per_api[0].api_name == "trade_cal"
    assert telemetry.per_api[0].http_attempts == 4
    assert tuple(item.outcome for item in telemetry.requests) == (
        "FAILURE",
        "SUCCESS",
        "FAILURE",
        "SUCCESS",
    )
    assert tuple(item.offset for item in telemetry.requests) == (0, 0, 2, 2)


@pytest.mark.parametrize(
    ("base_url", "expected_host", "official"),
    [
        (
            "https://user:private@API.TUSHARE.PRO:443/vendor?token=hidden#fragment",
            "api.tushare.pro",
            True,
        ),
        (
            "https://user:private@Proxy.Example.COM:8443/vendor?token=hidden#fragment",
            "proxy.example.com",
            False,
        ),
    ],
)
def test_provenance_comes_from_wrapped_transport_and_is_sanitized(
    base_url: str,
    expected_host: str,
    official: bool,
) -> None:
    clock = ManualClock()
    inner = FakeTransport([], base_url=base_url, clock=clock)
    transport = CountingTransport(inner, monotonic_fn=clock)

    telemetry = transport.snapshot()
    serialized = json.dumps(telemetry.to_dict(), sort_keys=True)

    assert not hasattr(transport, "base_url")
    assert base_url not in transport.__dict__.values()
    assert transport.source_identity.base_url_host == expected_host
    assert transport.source_identity.is_official_vendor is official
    assert telemetry.base_url_host == expected_host
    assert telemetry.is_official_vendor is official
    assert telemetry.dataset_source == (
        "tushare-official" if official else "tushare-via-non-official-endpoint"
    )
    assert telemetry.provenance_warnings == (
        () if official else (NON_OFFICIAL_VENDOR_WARNING,)
    )
    assert "user" not in serialized
    assert "private" not in serialized
    assert "hidden" not in serialized
    assert ":443" not in serialized
    assert ":8443" not in serialized


def test_full_source_url_is_absent_from_later_traceback_locals() -> None:
    url_canary = "source-url-credential-canary"
    base_url = (
        f"https://user:{url_canary}@Proxy.Example.COM:8443/vendor"
        f"?token={url_canary}#{url_canary}"
    )
    clock = ManualClock()
    error = TushareAPIError(40101, "credential rejected")
    inner = FakeTransport([error], base_url=base_url, clock=clock)
    transport = CountingTransport(inner, monotonic_fn=clock)
    request = TransportRequest(
        api_name="daily",
        params=(("ts_code", "000001.SZ"),),
        fields=("ts_code",),
        offset=0,
        limit=1,
    )

    with pytest.raises(TushareAPIError) as exc_info:
        transport.send(request, token=_CANARY)

    assert transport.source_identity.base_url_host == "proxy.example.com"
    assert not hasattr(transport, "base_url")
    assert base_url not in transport.__dict__.values()
    assert leaked_traceback_locals(exc_info.value.__traceback__, url_canary) == []


def test_real_http_transport_source_credentials_are_absent_from_later_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
    url_canary = "real-http-url-credential-canary"
    base_url = (
        f"https://user:{url_canary}@Proxy.Example.COM:8443/vendor"
        f"?token={url_canary}#{url_canary}"
    )

    def timeout(req: object, timeout: float | None = None) -> None:
        raise TimeoutError

    monkeypatch.setattr(urllib.request, "urlopen", timeout)
    clock = ManualClock()
    transport = CountingTransport(
        HttpJsonTransport(base_url=base_url),
        monotonic_fn=clock,
    )
    client = TushareClient(transport, max_retries=1, sleep_fn=clock.advance)

    with pytest.raises(TushareTransientError) as exc_info:
        client.fetch_trade_cal(
            exchange="SSE",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 2),
        )

    assert transport.source_identity.base_url_host == "proxy.example.com"
    assert not hasattr(transport, "base_url")
    assert leaked_traceback_locals(exc_info.value.__traceback__, url_canary) == []


def test_missing_actual_base_url_is_rejected() -> None:
    class NoBaseUrlTransport:
        def send(self, request: TransportRequest, *, token: str) -> TransportResponse:
            raise AssertionError("not called")

    with pytest.raises(ValueError, match="wrapped transport.base_url"):
        CountingTransport(NoBaseUrlTransport(), monotonic_fn=ManualClock())


def test_raw_snapshot_is_exact_ordered_separate_and_token_safe() -> None:
    clock = ManualClock()
    raw_response = TransportResponse(
        code=0,
        msg="message deliberately omitted",
        fields=("trade_date", "ts_code"),
        items=(("20260102", "000001.SZ"), ("20260101", "000002.SZ")),
    )
    inner = FakeTransport(
        [raw_response], base_url="https://proxy.example.com/v1", clock=clock
    )
    transport = CountingTransport(inner, monotonic_fn=clock)
    request = TransportRequest(
        api_name="daily",
        params=(("auth_token", _CANARY), ("ts_code", "000001.SZ")),
        fields=("trade_date", "ts_code"),
        offset=7,
        limit=2,
    )

    returned = transport.send(request, token=_CANARY)
    raw = transport.raw_snapshot()
    telemetry = transport.snapshot()

    assert returned is raw_response
    assert len(raw) == 1
    assert raw[0].sequence == 1
    assert raw[0].params == (("auth_token", "***"), ("ts_code", "000001.SZ"))
    assert raw[0].response_fields == raw_response.fields
    assert raw[0].response_items == raw_response.items
    raw_document = json.dumps(raw[0].to_dict(), ensure_ascii=False)
    telemetry_document = json.dumps(telemetry.to_dict(), ensure_ascii=False)
    assert _CANARY not in raw_document
    assert _CANARY not in telemetry_document
    assert '"items"' not in telemetry_document
    assert "000002.SZ" not in telemetry_document
    assert "message deliberately omitted" not in raw_document


def test_permanent_error_is_propagated_and_recorded_without_token() -> None:
    clock = ManualClock()
    error = TushareAPIError(40101, "token invalid")
    inner = FakeTransport(
        [error], base_url="https://api.tushare.pro", clock=clock
    )
    transport = CountingTransport(inner, monotonic_fn=clock)
    request = TransportRequest(
        api_name="daily",
        params=(("ts_code", "000001.SZ"),),
        fields=("ts_code",),
        offset=0,
        limit=1,
    )

    with pytest.raises(TushareAPIError) as exc_info:
        transport.send(request, token=_CANARY)

    assert exc_info.value is error
    telemetry = transport.snapshot()
    assert telemetry.http_attempts == 1
    assert telemetry.http_failures == 1
    assert telemetry.http_successes == 0
    assert telemetry.transient_errors == 0
    assert telemetry.requests[0].error_kind == "TushareAPIError"
    assert transport.raw_snapshot() == ()
    assert _CANARY not in json.dumps(telemetry.to_dict())
    assert leaked_traceback_locals(exc_info.value.__traceback__, _CANARY) == []


def test_monotonic_clock_cannot_move_backwards() -> None:
    values = iter((10.0, 9.0))

    class EmptyTransport:
        base_url = "https://api.tushare.pro"

        def send(self, request: TransportRequest, *, token: str) -> TransportResponse:
            raise AssertionError("not called")

    transport = CountingTransport(EmptyTransport(), monotonic_fn=lambda: next(values))
    transport.begin_logical_call("daily")
    with pytest.raises(ValueError, match="moved backwards"):
        transport.snapshot()


def test_counting_decorator_preserves_http_transport_traceback_token_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)

    def timeout(req: object, timeout: float | None = None) -> None:
        raise TimeoutError

    monkeypatch.setattr(urllib.request, "urlopen", timeout)
    clock = ManualClock()
    transport = CountingTransport(
        HttpJsonTransport(base_url="http://127.0.0.1:1"),
        monotonic_fn=clock,
    )
    client = TushareClient(transport, max_retries=1, sleep_fn=clock.advance)

    with pytest.raises(TushareTransientError) as exc_info:
        client.fetch_trade_cal(
            exchange="SSE",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 2),
        )

    telemetry = transport.snapshot()
    assert telemetry.http_attempts == 1
    assert telemetry.http_failures == 1
    assert telemetry.transient_errors == 1
    assert transport.raw_snapshot() == ()
    assert leaked_traceback_locals(exc_info.value.__traceback__, _CANARY) == []
