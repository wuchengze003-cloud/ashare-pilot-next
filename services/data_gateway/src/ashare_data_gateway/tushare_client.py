"""Read-only Tushare vendor client with pagination, retry, and strict validation.

Design constraints:
- Token ONLY from TUSHARE_TOKEN environment variable.
- No wall-clock reads; all dates/symbols are explicit call parameters.
- Injectable Transport; unit tests never touch the network.
- Returns frozen dataclasses in tuples; no mutable vendor data escapes.
- Only retries TushareTransientError; all other errors fail closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from datetime import date

from .tushare_models import (
    AdjFactorRecord,
    DailyBarRecord,
    StockBasicBatchResult,
    StockBasicRecord,
    StockBasicRejection,
    SuspendRecord,
    TradeCalRecord,
    validate_finite_float,
    validate_symbol,
    validate_trade_date,
)
from .tushare_transport import (
    SecretToken,
    Transport,
    TransportRequest,
    TransportResponse,
    TushareTransientError,
)

TOKEN_ENV_VAR = "TUSHARE_TOKEN"
DEFAULT_PAGE_SIZE = 5000
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 1.0
MAX_PAGES = 200
_SAFE_VENDOR_CODE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class TokenMissingError(Exception):
    """Raised when TUSHARE_TOKEN is not set. Never contains the token value."""

    def __init__(self) -> None:
        super().__init__(f"{TOKEN_ENV_VAR} environment variable is not set")


def _read_token() -> str:
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if not token:
        raise TokenMissingError
    return token


def _format_date(d: date) -> str:
    return d.strftime("%Y%m%d")


def _field_index(fields: tuple[str, ...], name: str, *, context: str) -> int:
    try:
        return fields.index(name)
    except ValueError:
        raise ValueError(f"{context}: missing expected field {name!r} in response") from None


def _row_value(row: tuple[object, ...], index: int, *, context: str, field: str) -> object:
    try:
        return row[index]
    except IndexError:
        raise ValueError(f"{context}: row missing expected field {field!r}") from None


def _require_string(
    value: object, *, context: str, field: str, nonempty: bool = False
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{context}: field {field} must be str, got {type(value).__name__}"
        )
    if nonempty and not value:
        raise ValueError(f"{context}: field {field} cannot be empty")
    return value


def _nullable_reference_string(value: object, *, context: str, field: str) -> str:
    """Preserve the accepted stock_basic null-to-empty reference semantics."""
    if value is None:
        return ""
    return _require_string(value, context=context, field=field)


def _safe_vendor_code(value: object) -> str | None:
    if isinstance(value, str) and _SAFE_VENDOR_CODE.fullmatch(value):
        return value
    return None


def _stock_basic_row_fingerprint(
    fields: tuple[str, ...], row: tuple[object, ...]
) -> str:
    canonical = json.dumps(
        {"fields": fields, "row": row},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class TushareClient:
    """Read-only vendor access to five Tushare pro endpoints.

    All methods are deterministic given the same transport responses.
    No system time is read; pagination and retry use injected sleep only.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_retries: int = MAX_RETRIES,
        retry_base_delay: float = RETRY_BASE_DELAY_SECONDS,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        self._transport = transport
        self._page_size = page_size
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._sleep_fn = sleep_fn

    def _send_with_retry(self, request: TransportRequest, *, token: str) -> TransportResponse:
        last_error: TushareTransientError | None = None
        for attempt in range(self._max_retries):
            try:
                return self._transport.send(request, token=token)
            except TushareTransientError as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    delay = self._retry_base_delay * (2**attempt)
                    self._sleep_fn(delay)
            except Exception:
                del token  # keep token out of traceback frames
                raise
        del token
        assert last_error is not None
        raise last_error

    def _fetch_all_pages(
        self,
        *,
        api_name: str,
        params: tuple[tuple[str, str], ...],
        fields: tuple[str, ...],
        token: str,
    ) -> TransportResponse:
        """Fetch all pages and merge items. Fails on duplicate primary keys upstream.

        Every page must return exactly the requested field set. Rows are reordered
        into request-field order before merging so vendor field order cannot affect
        the result. Raises on schema mismatch, max pages exceeded, or identical
        consecutive pages (stuck offset).
        """
        secret = SecretToken(token)
        del token
        if len(set(fields)) != len(fields):
            raise ValueError(f"{api_name}: requested fields contain duplicates")
        requested_field_set = frozenset(fields)

        all_items: list[tuple[object, ...]] = []
        offset = 0
        prev_items: tuple[tuple[object, ...], ...] | None = None

        begin_logical_call = getattr(self._transport, "begin_logical_call", None)
        if callable(begin_logical_call):
            try:
                begin_logical_call(api_name)
            except Exception:
                del secret
                raise

        for page_num in range(MAX_PAGES):
            request = TransportRequest(
                api_name=api_name,
                params=params,
                fields=fields,
                offset=offset,
                limit=self._page_size,
            )
            begin_page = getattr(self._transport, "begin_page", None)
            if callable(begin_page):
                try:
                    begin_page(request)
                except Exception:
                    del secret
                    raise
            try:
                response = self._send_with_retry(request, token=secret.reveal())
            except Exception:
                del secret
                raise

            # Validate each page independently, then normalize it to request order.
            response_fields = response.fields
            if len(set(response_fields)) != len(response_fields):
                raise ValueError(
                    f"{api_name}: page {page_num} has duplicate response fields: {response_fields}"
                )
            response_field_set = set(response_fields)
            missing_fields = tuple(field for field in fields if field not in response_field_set)
            extra_fields = tuple(
                field for field in response_fields if field not in requested_field_set
            )
            if missing_fields or extra_fields:
                raise ValueError(
                    f"{api_name}: page {page_num} response fields do not match request: "
                    f"missing={missing_fields}, extra={extra_fields}"
                )

            response_indexes = {field: index for index, field in enumerate(response_fields)}
            normalized_items: list[tuple[object, ...]] = []
            for row_num, row in enumerate(response.items):
                if len(row) != len(response_fields):
                    raise ValueError(
                        f"{api_name}: page {page_num} row {row_num} has "
                        f"{len(row)} values for {len(response_fields)} fields"
                    )
                normalized_items.append(tuple(row[response_indexes[field]] for field in fields))
            page_items = tuple(normalized_items)

            # Detect stuck offset (vendor ignoring pagination)
            if page_items and page_items == prev_items:
                raise ValueError(
                    f"{api_name}: identical response at page {page_num}; "
                    f"vendor may be ignoring offset parameter"
                )
            prev_items = page_items

            all_items.extend(page_items)

            if len(response.items) < self._page_size:
                break
            offset += self._page_size
        else:
            raise ValueError(
                f"{api_name}: exceeded maximum page limit ({MAX_PAGES})"
            )

        return TransportResponse(
            code=0,
            msg="",
            fields=fields,
            items=tuple(all_items),
        )

    # ------------------------------------------------------------------
    # trade_cal
    # ------------------------------------------------------------------

    def fetch_trade_cal(
        self,
        *,
        exchange: str,
        start_date: date,
        end_date: date,
    ) -> tuple[TradeCalRecord, ...]:
        """Fetch trading calendar for an explicit date range.

        Primary key: (exchange, cal_date).
        """
        if not exchange:
            raise ValueError("trade_cal: exchange cannot be empty")
        if end_date < start_date:
            raise ValueError("trade_cal: end_date cannot precede start_date")

        fields = ("exchange", "cal_date", "is_open", "pretrade_date")
        params = (
            ("exchange", exchange),
            ("start_date", _format_date(start_date)),
            ("end_date", _format_date(end_date)),
        )

        response = self._fetch_all_pages(
            api_name="trade_cal",
            params=params,
            fields=fields,
            token=_read_token(),
        )

        context = "trade_cal"
        idx_exchange = _field_index(response.fields, "exchange", context=context)
        idx_cal_date = _field_index(response.fields, "cal_date", context=context)
        idx_is_open = _field_index(response.fields, "is_open", context=context)
        idx_pretrade = _field_index(response.fields, "pretrade_date", context=context)

        seen: set[tuple[str, date]] = set()
        records: list[TradeCalRecord] = []

        for row in response.items:
            row_exchange = row[idx_exchange]
            if not isinstance(row_exchange, str) or not row_exchange:
                raise ValueError(f"{context}: invalid exchange value: {row_exchange!r}")
            if row_exchange != exchange:
                raise ValueError(
                    f"{context}: returned exchange {row_exchange!r} does not match "
                    f"requested {exchange!r}"
                )

            cal_date = validate_trade_date(row[idx_cal_date], context=context)
            if cal_date < start_date or cal_date > end_date:
                raise ValueError(
                    f"{context}: returned date {cal_date.isoformat()} outside request window"
                )
            key = (row_exchange, cal_date)
            if key in seen:
                raise ValueError(f"{context}: duplicate primary key: {key}")
            seen.add(key)

            is_open_raw = row[idx_is_open]
            if not isinstance(is_open_raw, int):
                raise ValueError(
                    f"{context}: is_open must be int, got {type(is_open_raw).__name__}"
                )

            pretrade_raw = row[idx_pretrade]
            pretrade_date: date | None = None
            if pretrade_raw is not None and pretrade_raw != "":
                pretrade_date = validate_trade_date(pretrade_raw, context=context)

            records.append(
                TradeCalRecord(
                    exchange=row_exchange,
                    cal_date=cal_date,
                    is_open=is_open_raw,
                    pretrade_date=pretrade_date,
                )
            )

        return tuple(records)

    # ------------------------------------------------------------------
    # stock_basic
    # ------------------------------------------------------------------

    def fetch_stock_basic(
        self,
        *,
        list_status: str = "L",
    ) -> tuple[StockBasicRecord, ...]:
        """Fetch a strict security master list.

        Any non-canonical vendor code causes the strict method to fail. Call
        :meth:`fetch_stock_basic_reconciled` when structured rejection evidence
        is required for the M1 L/D/P reconciliation.
        """
        batch = self.fetch_stock_basic_reconciled(list_status=list_status)
        if batch.rejected:
            safe_codes = tuple(rejection.vendor_ts_code for rejection in batch.rejected)
            raise ValueError(
                "stock_basic: rejected invalid vendor ts_code rows: "
                f"count={len(batch.rejected)} codes={safe_codes!r}"
            )
        return batch.accepted

    def fetch_stock_basic_reconciled(
        self,
        *,
        list_status: str,
    ) -> StockBasicBatchResult:
        """Fetch one L/D/P batch with explicit invalid-code rejection evidence.

        Only a non-canonical ``ts_code`` is rejectable. Every companion field,
        requested-status binding, date relationship, and duplicate key remains
        fail-closed. The returned counts always satisfy
        ``raw_row_count == len(accepted) + len(rejected)``.
        """
        if list_status not in ("L", "D", "P"):
            raise ValueError(f"stock_basic: invalid list_status filter: {list_status!r}")

        fields = (
            "ts_code",
            "symbol",
            "name",
            "area",
            "industry",
            "market",
            "list_date",
            "delist_date",
            "list_status",
        )
        params = (("list_status", list_status),)

        response = self._fetch_all_pages(
            api_name="stock_basic",
            params=params,
            fields=fields,
            token=_read_token(),
        )

        context = "stock_basic"
        idx_ts_code = _field_index(response.fields, "ts_code", context=context)
        idx_symbol = _field_index(response.fields, "symbol", context=context)
        idx_name = _field_index(response.fields, "name", context=context)
        idx_area = _field_index(response.fields, "area", context=context)
        idx_industry = _field_index(response.fields, "industry", context=context)
        idx_market = _field_index(response.fields, "market", context=context)
        idx_list_date = _field_index(response.fields, "list_date", context=context)
        idx_delist_date = _field_index(response.fields, "delist_date", context=context)
        idx_list_status = _field_index(response.fields, "list_status", context=context)

        seen_vendor_codes: set[str] = set()
        accepted: list[StockBasicRecord] = []
        rejected: list[StockBasicRejection] = []

        for row in response.items:
            raw_ts_code = _row_value(
                row, idx_ts_code, context=context, field="ts_code"
            )
            symbol_raw = _require_string(
                _row_value(row, idx_symbol, context=context, field="symbol"),
                context=context,
                field="symbol",
                nonempty=True,
            )
            name_raw = _require_string(
                _row_value(row, idx_name, context=context, field="name"),
                context=context,
                field="name",
                nonempty=True,
            )
            area = _nullable_reference_string(
                _row_value(row, idx_area, context=context, field="area"),
                context=context,
                field="area",
            )
            industry = _nullable_reference_string(
                _row_value(row, idx_industry, context=context, field="industry"),
                context=context,
                field="industry",
            )
            market = _nullable_reference_string(
                _row_value(row, idx_market, context=context, field="market"),
                context=context,
                field="market",
            )

            list_date_raw = _row_value(
                row, idx_list_date, context=context, field="list_date"
            )
            list_date: date | None = None
            if list_date_raw is not None and list_date_raw != "":
                list_date = validate_trade_date(list_date_raw, context=context)

            delist_date_raw = _row_value(
                row, idx_delist_date, context=context, field="delist_date"
            )
            delist_date: date | None = None
            if delist_date_raw is not None and delist_date_raw != "":
                delist_date = validate_trade_date(delist_date_raw, context=context)

            status_raw = _require_string(
                _row_value(row, idx_list_status, context=context, field="list_status"),
                context=context,
                field="list_status",
                nonempty=True,
            )
            if status_raw != list_status:
                raise ValueError(
                    f"{context}: returned list_status {status_raw!r} does not match "
                    f"requested {list_status!r}"
                )
            if (
                delist_date is not None
                and list_date is not None
                and delist_date < list_date
            ):
                raise ValueError(f"{context}: delist_date cannot precede list_date")

            if isinstance(raw_ts_code, str):
                if raw_ts_code in seen_vendor_codes:
                    raise ValueError(
                        f"{context}: duplicate primary key/vendor ts_code: "
                        f"{_safe_vendor_code(raw_ts_code)!r}"
                    )
                seen_vendor_codes.add(raw_ts_code)
                if "." in raw_ts_code and symbol_raw != raw_ts_code.rsplit(".", 1)[0]:
                    raise ValueError(
                        f"{context}: vendor ts_code/symbol mismatch: "
                        f"{_safe_vendor_code(raw_ts_code)!r} vs {symbol_raw!r}"
                    )
            else:
                # A wrong JSON type is schema corruption, not a vendor-code
                # exception that may be reconciled into rejection evidence.
                validate_symbol(raw_ts_code, context=context)

            try:
                ts_code = validate_symbol(raw_ts_code, context=context)
            except ValueError:
                rejected.append(
                    StockBasicRejection(
                        list_status=list_status,
                        reason="INVALID_VENDOR_TS_CODE",
                        row_fingerprint=_stock_basic_row_fingerprint(response.fields, row),
                        vendor_ts_code=_safe_vendor_code(raw_ts_code),
                    )
                )
                continue

            if symbol_raw != ts_code.split(".")[0]:
                raise ValueError(
                    f"{context}: ts_code/symbol mismatch: {ts_code} vs {symbol_raw!r}"
                )

            accepted.append(
                StockBasicRecord(
                    ts_code=ts_code,
                    symbol=symbol_raw,
                    name=name_raw,
                    area=area,
                    industry=industry,
                    market=market,
                    list_date=list_date,
                    delist_date=delist_date,
                    list_status=status_raw,
                )
            )

        return StockBasicBatchResult(
            list_status=list_status,
            raw_row_count=len(response.items),
            accepted=tuple(accepted),
            rejected=tuple(rejected),
        )

    # ------------------------------------------------------------------
    # daily
    # ------------------------------------------------------------------

    def fetch_daily(
        self,
        *,
        ts_code: str,
        start_date: date,
        end_date: date,
    ) -> tuple[DailyBarRecord, ...]:
        """Fetch unadjusted daily bars for one symbol and date range.

        Primary key: (ts_code, trade_date).
        Units: prices in CNY (unadjusted), vol in 手, amount in 千元.
        """
        validate_symbol(ts_code, context="daily")
        if end_date < start_date:
            raise ValueError("daily: end_date cannot precede start_date")

        fields = (
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
        )
        params = (
            ("ts_code", ts_code),
            ("start_date", _format_date(start_date)),
            ("end_date", _format_date(end_date)),
        )

        response = self._fetch_all_pages(
            api_name="daily",
            params=params,
            fields=fields,
            token=_read_token(),
        )

        context = "daily"
        idx_ts_code = _field_index(response.fields, "ts_code", context=context)
        idx_trade_date = _field_index(response.fields, "trade_date", context=context)
        idx_open = _field_index(response.fields, "open", context=context)
        idx_high = _field_index(response.fields, "high", context=context)
        idx_low = _field_index(response.fields, "low", context=context)
        idx_close = _field_index(response.fields, "close", context=context)
        idx_vol = _field_index(response.fields, "vol", context=context)
        idx_amount = _field_index(response.fields, "amount", context=context)

        seen: set[tuple[str, date]] = set()
        records: list[DailyBarRecord] = []

        for row in response.items:
            row_code = validate_symbol(row[idx_ts_code], context=context)
            if row_code != ts_code:
                raise ValueError(
                    f"{context}: returned symbol {row_code!r} does not match "
                    f"requested {ts_code!r}"
                )
            trade_date = validate_trade_date(row[idx_trade_date], context=context)
            if trade_date < start_date or trade_date > end_date:
                raise ValueError(
                    f"{context}: returned date {trade_date.isoformat()} outside request window"
                )
            key = (row_code, trade_date)
            if key in seen:
                raise ValueError(f"{context}: duplicate primary key: {key}")
            seen.add(key)

            records.append(
                DailyBarRecord(
                    ts_code=row_code,
                    trade_date=trade_date,
                    open=validate_finite_float(row[idx_open], context=context, field="open"),
                    high=validate_finite_float(row[idx_high], context=context, field="high"),
                    low=validate_finite_float(row[idx_low], context=context, field="low"),
                    close=validate_finite_float(row[idx_close], context=context, field="close"),
                    vol=validate_finite_float(row[idx_vol], context=context, field="vol"),
                    amount=validate_finite_float(row[idx_amount], context=context, field="amount"),
                )
            )

        return tuple(records)

    # ------------------------------------------------------------------
    # suspend_d
    # ------------------------------------------------------------------

    def fetch_suspend_d(
        self,
        *,
        ts_codes: tuple[str, ...],
        start_date: date,
        end_date: date,
    ) -> tuple[SuspendRecord, ...]:
        """Fetch suspension and resumption evidence for explicit symbols/window.

        Tushare accepts multiple comma-separated symbols. Inputs are validated,
        deduplicated, and sorted before the request. No ``suspend_type`` filter
        is sent, so both ``S`` and ``R`` records remain available to downstream
        diagnostics; coverage may use only ``S``.

        Primary key: (ts_code, trade_date, suspend_type).
        """
        if not isinstance(ts_codes, tuple):
            raise ValueError("suspend_d: ts_codes must be a tuple")
        if not ts_codes:
            raise ValueError("suspend_d: ts_codes cannot be empty")
        requested_codes = tuple(
            sorted({validate_symbol(code, context="suspend_d") for code in ts_codes})
        )
        if end_date < start_date:
            raise ValueError("suspend_d: end_date cannot precede start_date")

        fields = ("ts_code", "trade_date", "suspend_timing", "suspend_type")
        params = (
            ("ts_code", ",".join(requested_codes)),
            ("start_date", _format_date(start_date)),
            ("end_date", _format_date(end_date)),
        )
        response = self._fetch_all_pages(
            api_name="suspend_d",
            params=params,
            fields=fields,
            token=_read_token(),
        )

        context = "suspend_d"
        idx_ts_code = _field_index(response.fields, "ts_code", context=context)
        idx_trade_date = _field_index(response.fields, "trade_date", context=context)
        idx_suspend_timing = _field_index(
            response.fields, "suspend_timing", context=context
        )
        idx_suspend_type = _field_index(
            response.fields, "suspend_type", context=context
        )

        requested_set = frozenset(requested_codes)
        seen: set[tuple[str, date, str]] = set()
        records: list[SuspendRecord] = []
        for row in response.items:
            row_code = validate_symbol(
                _row_value(row, idx_ts_code, context=context, field="ts_code"),
                context=context,
            )
            if row_code not in requested_set:
                raise ValueError(
                    f"{context}: returned symbol {row_code!r} was not requested"
                )

            trade_date = validate_trade_date(
                _row_value(row, idx_trade_date, context=context, field="trade_date"),
                context=context,
            )
            if trade_date < start_date or trade_date > end_date:
                raise ValueError(
                    f"{context}: returned date {trade_date.isoformat()} outside request window"
                )

            timing_raw = _row_value(
                row, idx_suspend_timing, context=context, field="suspend_timing"
            )
            if timing_raw is not None and not isinstance(timing_raw, str):
                raise ValueError(
                    f"{context}: suspend_timing must be str or None, "
                    f"got {type(timing_raw).__name__}"
                )

            suspend_type = _require_string(
                _row_value(row, idx_suspend_type, context=context, field="suspend_type"),
                context=context,
                field="suspend_type",
                nonempty=True,
            )
            if suspend_type not in ("S", "R"):
                raise ValueError(
                    f"{context}: suspend_type must be 'S' or 'R', got {suspend_type!r}"
                )

            key = (row_code, trade_date, suspend_type)
            if key in seen:
                raise ValueError(f"{context}: duplicate primary key: {key}")
            seen.add(key)
            records.append(
                SuspendRecord(
                    ts_code=row_code,
                    trade_date=trade_date,
                    suspend_timing=timing_raw,
                    suspend_type=suspend_type,
                )
            )

        return tuple(records)

    # ------------------------------------------------------------------
    # adj_factor
    # ------------------------------------------------------------------

    def fetch_adj_factor(
        self,
        *,
        ts_code: str,
        start_date: date,
        end_date: date,
    ) -> tuple[AdjFactorRecord, ...]:
        """Fetch adjustment factors as independent vendor records.

        Primary key: (ts_code, trade_date).
        This PR does NOT apply factors to produce adjusted prices.
        """
        validate_symbol(ts_code, context="adj_factor")
        if end_date < start_date:
            raise ValueError("adj_factor: end_date cannot precede start_date")

        fields = ("ts_code", "trade_date", "adj_factor")
        params = (
            ("ts_code", ts_code),
            ("start_date", _format_date(start_date)),
            ("end_date", _format_date(end_date)),
        )

        response = self._fetch_all_pages(
            api_name="adj_factor",
            params=params,
            fields=fields,
            token=_read_token(),
        )

        context = "adj_factor"
        idx_ts_code = _field_index(response.fields, "ts_code", context=context)
        idx_trade_date = _field_index(response.fields, "trade_date", context=context)
        idx_adj_factor = _field_index(response.fields, "adj_factor", context=context)

        seen: set[tuple[str, date]] = set()
        records: list[AdjFactorRecord] = []

        for row in response.items:
            row_code = validate_symbol(row[idx_ts_code], context=context)
            if row_code != ts_code:
                raise ValueError(
                    f"{context}: returned symbol {row_code!r} does not match "
                    f"requested {ts_code!r}"
                )
            trade_date = validate_trade_date(row[idx_trade_date], context=context)
            if trade_date < start_date or trade_date > end_date:
                raise ValueError(
                    f"{context}: returned date {trade_date.isoformat()} outside request window"
                )
            key = (row_code, trade_date)
            if key in seen:
                raise ValueError(f"{context}: duplicate primary key: {key}")
            seen.add(key)

            factor = validate_finite_float(
                row[idx_adj_factor], context=context, field="adj_factor"
            )

            records.append(
                AdjFactorRecord(
                    ts_code=row_code,
                    trade_date=trade_date,
                    adj_factor=factor,
                )
            )

        return tuple(records)
