"""Tests for the Tushare read-only vendor access layer.

No real network access. No system time reads. All transport is injected;
the few HttpJsonTransport tests target 127.0.0.1 (connection refused) or
monkeypatched urlopen, never external hosts.

_CANARY is a searchable placeholder, deliberately not shaped like a secret.
"""

from __future__ import annotations

import io
import json
import urllib.request
from datetime import date
from pathlib import Path

import pytest
from ashare_data_gateway.tushare_client import (
    TOKEN_ENV_VAR,
    TokenMissingError,
    TushareClient,
    _format_date,
)
from ashare_data_gateway.tushare_models import (
    AdjFactorRecord,
    DailyBarRecord,
    TradeCalRecord,
)
from ashare_data_gateway.tushare_transport import (
    URL_ENV_VAR,
    HttpJsonTransport,
    SecretToken,
    TransportRequest,
    TransportResponse,
    TushareAPIError,
    TushareTransientError,
    _is_transient_code,
    resolve_base_url,
)

_CANARY = "<<canary-token-placeholder>>"

# ---------------------------------------------------------------------------
# Fake transport (no network)
# ---------------------------------------------------------------------------


class FakeTransport:
    """Records requests and returns pre-programmed responses."""

    def __init__(self, responses: list[TransportResponse | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[tuple[TransportRequest, str]] = []

    def send(self, request: TransportRequest, *, token: str) -> TransportResponse:
        self.requests.append((request, token))
        if not self._responses:
            raise TushareAPIError(0, "no_more_responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_response(
    fields: tuple[str, ...],
    items: list[list[object]],
) -> TransportResponse:
    return TransportResponse(
        code=0,
        msg="",
        fields=fields,
        items=tuple(tuple(row) for row in items),
    )


def make_client(
    responses: list[TransportResponse | Exception],
    *,
    page_size: int = 5000,
    max_retries: int = 3,
) -> tuple[TushareClient, FakeTransport]:
    transport = FakeTransport(responses)
    sleeps: list[float] = []
    client = TushareClient(
        transport,
        page_size=page_size,
        max_retries=max_retries,
        retry_base_delay=0.01,
        sleep_fn=sleeps.append,
    )
    return client, transport


def _collect_leaked_frames(exc_tb: object, canary: str) -> list[str]:
    """Walk traceback frames, skipping this test file, and report token leaks.

    Checks repr of every local, raw bytes locals, and the .data payload of
    any urllib Request local (which would hold the serialized token).
    """
    test_file = Path(__file__).resolve()
    leaked: list[str] = []
    tb = exc_tb
    while tb is not None:
        frame = tb.tb_frame
        if Path(frame.f_code.co_filename).resolve() != test_file:
            for name, value in frame.f_locals.items():
                if canary in repr(value):
                    leaked.append(f"{frame.f_code.co_name}.{name}")
                if isinstance(value, bytes) and canary.encode() in value:
                    leaked.append(f"{frame.f_code.co_name}.{name}(bytes)")
                if (
                    isinstance(value, urllib.request.Request)
                    and isinstance(value.data, bytes)
                    and canary.encode() in value.data
                ):
                    leaked.append(f"{frame.f_code.co_name}.{name}(request.data)")
        tb = tb.tb_next
    return leaked


# ---------------------------------------------------------------------------
# Token tests
# ---------------------------------------------------------------------------


class TestTokenHandling:
    def test_token_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
        client, _ = make_client([])
        with pytest.raises(TokenMissingError, match=TOKEN_ENV_VAR):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 31)
            )

    def test_token_empty_string_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, "   ")
        client, _ = make_client([])
        with pytest.raises(TokenMissingError):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 31)
            )

    def test_token_not_in_exception_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        transport = FakeTransport([TushareAPIError(40001, "auth failed")])
        client = TushareClient(transport, sleep_fn=lambda _: None)
        with pytest.raises(TushareAPIError) as exc_info:
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 31)
            )
        assert _CANARY not in str(exc_info.value)
        assert _CANARY not in repr(exc_info.value)

    def test_token_not_in_transient_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        transport = FakeTransport([
            TushareTransientError("timeout"),
            TushareTransientError("timeout"),
            TushareTransientError("timeout"),
        ])
        client = TushareClient(transport, max_retries=3, sleep_fn=lambda _: None)
        with pytest.raises(TushareTransientError) as exc_info:
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 31)
            )
        assert _CANARY not in str(exc_info.value)

    def test_secret_token_repr_is_masked(self) -> None:
        token = SecretToken(_CANARY)
        assert _CANARY not in repr(token)
        assert _CANARY not in str(token)
        assert repr(token) == "SecretToken(***)"
        assert token.reveal() == _CANARY

    def test_token_absent_from_transport_traceback_locals(self) -> None:
        """Exercise the REAL HttpJsonTransport IO path (connection refused on
        loopback, never external) and verify no frame on the raised exception
        retains the token in its locals."""
        transport = HttpJsonTransport(base_url="http://127.0.0.1:1")
        request = TransportRequest(
            api_name="daily", params=(), fields=("ts_code",), offset=0, limit=1
        )
        with pytest.raises(TushareAPIError) as excinfo:
            transport.send(request, token=_CANARY)

        leaked = _collect_leaked_frames(excinfo.tb, _CANARY)
        assert leaked == [], f"token leaked in frames: {leaked}"

    def test_token_absent_from_client_traceback_locals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client-side frames must also be clean when transport raises."""
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        transport = FakeTransport([TushareAPIError(40001, "auth failed")])
        client = TushareClient(transport, sleep_fn=lambda _: None)
        with pytest.raises(TushareAPIError) as excinfo:
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 31)
            )
        leaked = _collect_leaked_frames(excinfo.tb, _CANARY)
        assert leaked == [], f"token leaked in frames: {leaked}"


# ---------------------------------------------------------------------------
# URL configuration tests
# ---------------------------------------------------------------------------


class TestUrlConfiguration:
    def test_custom_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(URL_ENV_VAR, "https://custom.example.com")
        assert resolve_base_url() == "https://custom.example.com"

    def test_default_url_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(URL_ENV_VAR, raising=False)
        assert resolve_base_url() == "https://api.tushare.pro"

    def test_empty_url_env_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(URL_ENV_VAR, "  ")
        assert resolve_base_url() == "https://api.tushare.pro"

    def test_explicit_base_url_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(URL_ENV_VAR, "https://env-url.example.com")
        transport = HttpJsonTransport(base_url="https://explicit.example.com")
        assert transport.base_url == "https://explicit.example.com"

    def test_http_non_local_rejected(self) -> None:
        with pytest.raises(ValueError, match="must use https"):
            HttpJsonTransport(base_url="http://api.tushare.pro")

    def test_http_localhost_allowed(self) -> None:
        transport = HttpJsonTransport(base_url="http://127.0.0.1:8080")
        assert transport.base_url == "http://127.0.0.1:8080"

    def test_http_localhost_name_allowed(self) -> None:
        transport = HttpJsonTransport(base_url="http://localhost:9090")
        assert transport.base_url == "http://localhost:9090"


# ---------------------------------------------------------------------------
# HttpJsonTransport response handling (urlopen monkeypatched, no network)
# ---------------------------------------------------------------------------


def _stub_urlopen(
    monkeypatch: pytest.MonkeyPatch, document: object
) -> None:
    body = json.dumps(document).encode("utf-8")
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout=None: io.BytesIO(body)
    )


class TestTransportResponseHandling:
    def test_null_row_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_urlopen(monkeypatch, {
            "code": 0,
            "msg": "",
            "data": {"fields": ["ts_code"], "items": [["000001.SZ"], None]},
        })
        transport = HttpJsonTransport(base_url="http://127.0.0.1:1")
        request = TransportRequest(
            api_name="daily", params=(), fields=("ts_code",), offset=0, limit=10
        )
        with pytest.raises(TushareAPIError, match="malformed_row_at_index_1"):
            transport.send(request, token=_CANARY)

    def test_dict_row_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_urlopen(monkeypatch, {
            "code": 0,
            "msg": "",
            "data": {"fields": ["ts_code"], "items": [{"ts_code": "000001.SZ"}]},
        })
        transport = HttpJsonTransport(base_url="http://127.0.0.1:1")
        request = TransportRequest(
            api_name="daily", params=(), fields=("ts_code",), offset=0, limit=10
        )
        with pytest.raises(TushareAPIError, match="malformed_row_at_index_0"):
            transport.send(request, token=_CANARY)

    def test_timeout_is_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_timeout(req: object, timeout: float | None = None) -> None:
            raise TimeoutError

        monkeypatch.setattr(urllib.request, "urlopen", raise_timeout)
        transport = HttpJsonTransport(base_url="http://127.0.0.1:1")
        request = TransportRequest(
            api_name="daily", params=(), fields=("ts_code",), offset=0, limit=1
        )
        with pytest.raises(TushareTransientError, match="timeout"):
            transport.send(request, token=_CANARY)

    def test_vendor_msg_is_sanitized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        long_msg = "x" * 500 + "deadbeefdeadbeef"
        _stub_urlopen(monkeypatch, {"code": 40203, "msg": long_msg, "data": None})
        transport = HttpJsonTransport(base_url="http://127.0.0.1:1")
        request = TransportRequest(
            api_name="daily", params=(), fields=("ts_code",), offset=0, limit=1
        )
        with pytest.raises(TushareAPIError) as excinfo:
            transport.send(request, token=_CANARY)
        assert len(excinfo.value.msg) <= 200


# ---------------------------------------------------------------------------
# trade_cal tests
# ---------------------------------------------------------------------------

TRADE_CAL_FIELDS = ("exchange", "cal_date", "is_open", "pretrade_date")


class TestTradeCal:
    def test_normal_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            TRADE_CAL_FIELDS,
            [
                ["SSE", "20260101", 0, None],
                ["SSE", "20260102", 1, "20260101"],
            ],
        )
        client, transport = make_client([response])
        result = client.fetch_trade_cal(
            exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 2)
        )

        assert len(result) == 2
        assert result[0] == TradeCalRecord(
            exchange="SSE", cal_date=date(2026, 1, 1), is_open=0, pretrade_date=None
        )
        assert result[1] == TradeCalRecord(
            exchange="SSE", cal_date=date(2026, 1, 2), is_open=1, pretrade_date=date(2026, 1, 1)
        )
        # Verify deterministic request params
        req, _ = transport.requests[0]
        assert req.api_name == "trade_cal"
        assert ("exchange", "SSE") in req.params
        assert ("start_date", "20260101") in req.params
        assert ("end_date", "20260102") in req.params

    def test_missing_field_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(("exchange", "cal_date"), [["SSE", "20260101"]])
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="missing expected field"):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 1)
            )

    def test_duplicate_primary_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            TRADE_CAL_FIELDS,
            [
                ["SSE", "20260101", 1, None],
                ["SSE", "20260101", 0, None],
            ],
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="duplicate primary key"):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 1)
            )

    def test_invalid_date_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            TRADE_CAL_FIELDS,
            [["SSE", "20261301", 1, None]],  # month 13 invalid
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="impossible calendar date"):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 12, 31)
            )

    def test_end_before_start_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        client, _ = make_client([])
        with pytest.raises(ValueError, match="end_date cannot precede"):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 2, 1), end_date=date(2026, 1, 1)
            )


# ---------------------------------------------------------------------------
# stock_basic tests
# ---------------------------------------------------------------------------

STOCK_BASIC_FIELDS = (
    "ts_code", "symbol", "name", "area", "industry",
    "market", "list_date", "delist_date", "list_status",
)


class TestStockBasic:
    def test_normal_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            STOCK_BASIC_FIELDS,
            [
                ["000001.SZ", "000001", "平安银行", "深圳", "银行", "主板",
                 "19910403", None, "L"],
            ],
        )
        client, transport = make_client([response])
        result = client.fetch_stock_basic(list_status="L")

        assert len(result) == 1
        assert result[0].ts_code == "000001.SZ"
        assert result[0].name == "平安银行"
        assert result[0].list_date == date(1991, 4, 3)
        assert result[0].delist_date is None
        req, _ = transport.requests[0]
        assert req.api_name == "stock_basic"

    def test_invalid_symbol_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            STOCK_BASIC_FIELDS,
            [["INVALID", "000001", "Test", "深圳", "银行", "主板",
              "19910403", None, "L"]],
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="invalid symbol format"):
            client.fetch_stock_basic()

    def test_duplicate_ts_code_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        row = ["000001.SZ", "000001", "平安银行", "深圳", "银行", "主板",
               "19910403", None, "L"]
        response = make_response(STOCK_BASIC_FIELDS, [row, row])
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="duplicate primary key"):
            client.fetch_stock_basic()

    def test_ts_code_symbol_mismatch_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            STOCK_BASIC_FIELDS,
            [["000001.SZ", "600519", "漂移测试", "深圳", "银行", "主板",
              "19910403", None, "L"]],
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="ts_code/symbol mismatch"):
            client.fetch_stock_basic()


# ---------------------------------------------------------------------------
# daily tests
# ---------------------------------------------------------------------------

DAILY_FIELDS = (
    "ts_code", "trade_date", "open", "high", "low", "close",
    "vol", "amount",
)


class TestDaily:
    def test_normal_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            DAILY_FIELDS,
            [
                ["000001.SZ", "20260102", 10.5, 11.0, 10.2, 10.8,
                 123456.78, 987654.32],
            ],
        )
        client, transport = make_client([response])
        result = client.fetch_daily(
            ts_code="000001.SZ", start_date=date(2026, 1, 2), end_date=date(2026, 1, 2)
        )

        assert len(result) == 1
        bar = result[0]
        assert bar.ts_code == "000001.SZ"
        assert bar.trade_date == date(2026, 1, 2)
        assert bar.open == 10.5
        assert bar.vol == 123456.78
        req, _ = transport.requests[0]
        assert req.api_name == "daily"
        assert ("ts_code", "000001.SZ") in req.params

    def test_nan_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            DAILY_FIELDS,
            [
                ["000001.SZ", "20260102", float("nan"), 11.0, 10.2, 10.8,
                 100.0, 200.0],
            ],
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="non-finite"):
            client.fetch_daily(
                ts_code="000001.SZ", start_date=date(2026, 1, 2), end_date=date(2026, 1, 2)
            )

    def test_infinity_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            DAILY_FIELDS,
            [
                ["000001.SZ", "20260102", 10.5, float("inf"), 10.2, 10.8,
                 100.0, 200.0],
            ],
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="non-finite"):
            client.fetch_daily(
                ts_code="000001.SZ", start_date=date(2026, 1, 2), end_date=date(2026, 1, 2)
            )

    def test_none_price_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            DAILY_FIELDS,
            [
                ["000001.SZ", "20260102", None, 11.0, 10.2, 10.8,
                 100.0, 200.0],
            ],
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="is None"):
            client.fetch_daily(
                ts_code="000001.SZ", start_date=date(2026, 1, 2), end_date=date(2026, 1, 2)
            )

    def test_bool_price_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            DAILY_FIELDS,
            [
                ["000001.SZ", "20260102", True, 11.0, 10.2, 10.8,
                 100.0, 200.0],
            ],
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="got bool"):
            client.fetch_daily(
                ts_code="000001.SZ", start_date=date(2026, 1, 2), end_date=date(2026, 1, 2)
            )

    def test_invalid_symbol_param_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        client, _ = make_client([])
        with pytest.raises(ValueError, match="invalid symbol format"):
            client.fetch_daily(
                ts_code="ABCDEF", start_date=date(2026, 1, 1), end_date=date(2026, 1, 2)
            )

    def test_duplicate_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        row = ["000001.SZ", "20260102", 10.5, 11.0, 10.2, 10.8,
               100.0, 200.0]
        response = make_response(DAILY_FIELDS, [row, row])
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="duplicate primary key"):
            client.fetch_daily(
                ts_code="000001.SZ", start_date=date(2026, 1, 2), end_date=date(2026, 1, 2)
            )


# ---------------------------------------------------------------------------
# adj_factor tests
# ---------------------------------------------------------------------------

ADJ_FACTOR_FIELDS = ("ts_code", "trade_date", "adj_factor")


class TestAdjFactor:
    def test_normal_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            ADJ_FACTOR_FIELDS,
            [["000001.SZ", "20260102", 123.456]],
        )
        client, transport = make_client([response])
        result = client.fetch_adj_factor(
            ts_code="000001.SZ", start_date=date(2026, 1, 2), end_date=date(2026, 1, 2)
        )

        assert len(result) == 1
        assert result[0] == AdjFactorRecord(
            ts_code="000001.SZ", trade_date=date(2026, 1, 2), adj_factor=123.456
        )
        req, _ = transport.requests[0]
        assert req.api_name == "adj_factor"

    def test_zero_factor_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            ADJ_FACTOR_FIELDS,
            [["000001.SZ", "20260102", 0.0]],
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="must be positive"):
            client.fetch_adj_factor(
                ts_code="000001.SZ", start_date=date(2026, 1, 2), end_date=date(2026, 1, 2)
            )

    def test_negative_factor_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            ADJ_FACTOR_FIELDS,
            [["000001.SZ", "20260102", -1.5]],
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="must be positive"):
            client.fetch_adj_factor(
                ts_code="000001.SZ", start_date=date(2026, 1, 2), end_date=date(2026, 1, 2)
            )

    def test_nan_factor_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            ADJ_FACTOR_FIELDS,
            [["000001.SZ", "20260102", float("nan")]],
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="non-finite"):
            client.fetch_adj_factor(
                ts_code="000001.SZ", start_date=date(2026, 1, 2), end_date=date(2026, 1, 2)
            )


# ---------------------------------------------------------------------------
# API error and transient error tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_api_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        transport = FakeTransport([TushareAPIError(40001, "invalid token")])
        client = TushareClient(transport, sleep_fn=lambda _: None)
        with pytest.raises(TushareAPIError) as exc_info:
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 1)
            )
        assert exc_info.value.code == 40001

    def test_transient_error_retries_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        good_response = make_response(
            TRADE_CAL_FIELDS,
            [["SSE", "20260101", 1, None]],
        )
        transport = FakeTransport([
            TushareTransientError("timeout"),
            TushareTransientError("rate_limit"),
            good_response,
        ])
        sleeps: list[float] = []
        client = TushareClient(
            transport, max_retries=3, retry_base_delay=1.0, sleep_fn=sleeps.append
        )
        result = client.fetch_trade_cal(
            exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 1)
        )
        assert len(result) == 1
        assert len(sleeps) == 2
        assert sleeps[0] == 1.0
        assert sleeps[1] == 2.0

    def test_transient_error_exhausts_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        transport = FakeTransport([
            TushareTransientError("timeout"),
            TushareTransientError("timeout"),
            TushareTransientError("timeout"),
        ])
        client = TushareClient(transport, max_retries=3, sleep_fn=lambda _: None)
        with pytest.raises(TushareTransientError):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 1)
            )

    def test_non_transient_error_no_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        transport = FakeTransport([TushareAPIError(50000, "server error")])
        client = TushareClient(transport, max_retries=3, sleep_fn=lambda _: None)
        with pytest.raises(TushareAPIError):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 1)
            )
        # Only one request should have been made (no retry for non-transient)
        assert len(transport.requests) == 1

    def test_max_retries_zero_rejected(self) -> None:
        transport = FakeTransport([])
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            TushareClient(transport, max_retries=0, sleep_fn=lambda _: None)

    def test_page_size_zero_rejected(self) -> None:
        transport = FakeTransport([])
        with pytest.raises(ValueError, match="page_size must be >= 1"):
            TushareClient(transport, page_size=0, sleep_fn=lambda _: None)


# ---------------------------------------------------------------------------
# Transient code classification tests
# ---------------------------------------------------------------------------


class TestTransientCodeClassification:
    @pytest.mark.parametrize("code", [-2000])
    def test_known_transient_codes(self, code: int) -> None:
        assert _is_transient_code(code) is True

    @pytest.mark.parametrize("code", [0, -1, 40001, 40203, 50000, -9999])
    def test_unknown_codes_are_permanent(self, code: int) -> None:
        assert _is_transient_code(code) is False

    def test_permission_error_not_transient(self) -> None:
        """Codes with '限制' in msg are NOT retried (fail-closed)."""
        # Previously keyword heuristic would retry these; now only code matters
        assert _is_transient_code(40203) is False


# ---------------------------------------------------------------------------
# Pagination tests
# ---------------------------------------------------------------------------


class TestPagination:
    def test_multi_page_merge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        page1 = make_response(
            TRADE_CAL_FIELDS,
            [["SSE", f"2026010{i}", 1, None] for i in range(1, 4)],
        )
        page2 = make_response(
            TRADE_CAL_FIELDS,
            [["SSE", "20260104", 0, None]],
        )
        client, transport = make_client([page1, page2], page_size=3)
        result = client.fetch_trade_cal(
            exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 4)
        )

        assert len(result) == 4
        assert len(transport.requests) == 2
        assert transport.requests[0][0].offset == 0
        assert transport.requests[0][0].limit == 3
        assert transport.requests[1][0].offset == 3
        assert transport.requests[1][0].limit == 3

    def test_inconsistent_fields_across_pages_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        page1 = make_response(
            TRADE_CAL_FIELDS,
            [["SSE", "20260101", 1, None]] * 3,
        )
        page2 = make_response(
            ("exchange", "cal_date"),  # different fields
            [["SSE", "20260102"]],
        )
        client, _ = make_client([page1, page2], page_size=3)
        with pytest.raises(ValueError, match="inconsistent fields"):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 4)
            )

    def test_empty_result_returns_empty_tuple(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legitimate empty result (no data in range) returns () without error."""
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = TransportResponse(code=0, msg="", fields=(), items=())
        client, _ = make_client([response])
        result = client.fetch_trade_cal(
            exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 1)
        )
        assert result == ()

    def test_page2_null_data_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If page 2 has items but empty fields, pagination must fail closed."""
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        page1 = make_response(
            TRADE_CAL_FIELDS,
            [["SSE", f"2026010{i}", 1, None] for i in range(1, 4)],
        )
        page2 = TransportResponse(
            code=0, msg="", fields=(),
            items=(("SSE", "20260104", 1, None),),
        )
        client, _ = make_client([page1, page2], page_size=3)
        with pytest.raises(ValueError, match="items but empty fields"):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 4)
            )

    def test_max_pages_exceeded_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pagination must not loop forever if vendor always returns full pages."""
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)

        class InfiniteTransport:
            """Returns unique full pages forever to test MAX_PAGES guard."""

            def __init__(self) -> None:
                self.requests: list[tuple[TransportRequest, str]] = []
                self._page = 0

            def send(
                self, request: TransportRequest, *, token: str
            ) -> TransportResponse:
                self.requests.append((request, token))
                rows = [
                    ["SSE", f"{self._page:04d}0{i}", 1, None] for i in range(1, 6)
                ]
                self._page += 1
                return TransportResponse(
                    code=0,
                    msg="",
                    fields=TRADE_CAL_FIELDS,
                    items=tuple(tuple(r) for r in rows),
                )

        transport = InfiniteTransport()
        client = TushareClient(transport, page_size=5, sleep_fn=lambda _: None)
        with pytest.raises(ValueError, match="exceeded maximum page limit"):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 12, 31)
            )

    def test_identical_pages_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If vendor ignores offset and returns same data, detect and raise."""
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        same_page = make_response(
            TRADE_CAL_FIELDS,
            [["SSE", f"2026010{i}", 1, None] for i in range(1, 4)],
        )
        # Two identical full pages
        client, _ = make_client([same_page, same_page], page_size=3)
        with pytest.raises(ValueError, match="identical response"):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 4)
            )


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_request_params_are_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(ADJ_FACTOR_FIELDS, [["600000.SH", "20260315", 99.9]])
        client1, transport1 = make_client([response])
        client1.fetch_adj_factor(
            ts_code="600000.SH", start_date=date(2026, 3, 15), end_date=date(2026, 3, 15)
        )

        response2 = make_response(ADJ_FACTOR_FIELDS, [["600000.SH", "20260315", 99.9]])
        client2, transport2 = make_client([response2])
        client2.fetch_adj_factor(
            ts_code="600000.SH", start_date=date(2026, 3, 15), end_date=date(2026, 3, 15)
        )

        req1, _ = transport1.requests[0]
        req2, _ = transport2.requests[0]
        assert req1 == req2

    def test_format_date_is_explicit(self) -> None:
        assert _format_date(date(2026, 7, 30)) == "20260730"
        assert _format_date(date(2000, 1, 1)) == "20000101"


# ---------------------------------------------------------------------------
# Frozen / immutability tests
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_records_are_frozen(self) -> None:
        record = TradeCalRecord(
            exchange="SSE", cal_date=date(2026, 1, 1), is_open=1, pretrade_date=None
        )
        with pytest.raises(AttributeError):
            record.exchange = "SZSE"  # type: ignore[misc]

    def test_daily_bar_frozen(self) -> None:
        bar = DailyBarRecord(
            ts_code="000001.SZ", trade_date=date(2026, 1, 2),
            open=10.0, high=11.0, low=9.0, close=10.5,
            vol=1000.0, amount=5000.0,
        )
        with pytest.raises(AttributeError):
            bar.close = 999.0  # type: ignore[misc]

    def test_result_is_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            TRADE_CAL_FIELDS,
            [["SSE", "20260101", 1, None]],
        )
        client, _ = make_client([response])
        result = client.fetch_trade_cal(
            exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 1)
        )
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Response structure validation
# ---------------------------------------------------------------------------


class TestResponseStructure:
    def test_wrong_field_type_in_row_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            TRADE_CAL_FIELDS,
            [["SSE", "20260101", "not_an_int", None]],  # is_open should be int
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="is_open must be int"):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 1)
            )

    def test_non_string_date_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TOKEN_ENV_VAR, _CANARY)
        response = make_response(
            TRADE_CAL_FIELDS,
            [["SSE", 20260101, 1, None]],  # date as int
        )
        client, _ = make_client([response])
        with pytest.raises(ValueError, match="date must be str"):
            client.fetch_trade_cal(
                exchange="SSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 1)
            )


# ---------------------------------------------------------------------------
# No wall-clock dependency test
# ---------------------------------------------------------------------------


class TestNoWallClock:
    def test_no_time_module_calls_in_client(self) -> None:
        """Verify client source does not call datetime.now/today/time.time directly."""
        import ast

        source_path = Path(__file__).resolve().parents[1] / (
            "src/ashare_data_gateway/tushare_client.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        forbidden = {
            "datetime.now", "datetime.today", "datetime.utcnow",
            "date.today", "time.time",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    call_name = f"{func.value.id}.{func.attr}"
                    assert call_name not in forbidden, (
                        f"wall-clock call found: {call_name}"
                    )
