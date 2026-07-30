"""Read-only Tushare vendor client with pagination, retry, and strict validation.

Design constraints:
- Token ONLY from TUSHARE_TOKEN environment variable.
- No wall-clock reads; all dates/symbols are explicit call parameters.
- Injectable Transport; unit tests never touch the network.
- Returns frozen dataclasses in tuples; no mutable vendor data escapes.
- Only retries TushareTransientError; all other errors fail closed.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from datetime import date

from .tushare_models import (
    AdjFactorRecord,
    DailyBarRecord,
    StockBasicRecord,
    TradeCalRecord,
    validate_finite_float,
    validate_symbol,
    validate_trade_date,
)
from .tushare_transport import (
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


class TushareClient:
    """Read-only vendor access to four Tushare pro endpoints.

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

        Returns fields=requested fields and items=() for legitimate empty results.
        Raises on: inconsistent fields, items-without-fields, max pages exceeded,
        or identical consecutive pages (stuck offset).
        """
        all_items: list[tuple[object, ...]] = []
        merged_fields: tuple[str, ...] = ()
        offset = 0
        prev_items: tuple[tuple[object, ...], ...] | None = None

        for page_num in range(MAX_PAGES):
            request = TransportRequest(
                api_name=api_name,
                params=params,
                fields=fields,
                offset=offset,
                limit=self._page_size,
            )
            try:
                response = self._send_with_retry(request, token=token)
            except Exception:
                del token  # keep token out of traceback frames
                raise

            # Field consistency validation
            if response.fields:
                if not merged_fields:
                    merged_fields = response.fields
                elif response.fields != merged_fields:
                    raise ValueError(
                        f"{api_name}: inconsistent fields across pages: "
                        f"{merged_fields} vs {response.fields}"
                    )
            elif response.items:
                raise ValueError(
                    f"{api_name}: page {page_num} has items but empty fields"
                )

            # Detect stuck offset (vendor ignoring pagination)
            if response.items and response.items == prev_items:
                raise ValueError(
                    f"{api_name}: identical response at page {page_num}; "
                    f"vendor may be ignoring offset parameter"
                )
            prev_items = response.items

            all_items.extend(response.items)

            if len(response.items) < self._page_size:
                break
            offset += self._page_size
        else:
            raise ValueError(
                f"{api_name}: exceeded maximum page limit ({MAX_PAGES})"
            )

        # Legitimate empty result: return requested fields with no items
        if not merged_fields:
            merged_fields = fields

        return TransportResponse(
            code=0,
            msg="",
            fields=merged_fields,
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

            cal_date = validate_trade_date(row[idx_cal_date], context=context)
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
        """Fetch security master list.

        Primary key: ts_code.
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

        seen: set[str] = set()
        records: list[StockBasicRecord] = []

        for row in response.items:
            ts_code = validate_symbol(row[idx_ts_code], context=context)
            if ts_code in seen:
                raise ValueError(f"{context}: duplicate primary key: {ts_code}")
            seen.add(ts_code)

            symbol_raw = row[idx_symbol]
            if not isinstance(symbol_raw, str) or not symbol_raw:
                raise ValueError(f"{context}: invalid symbol: {symbol_raw!r}")
            if symbol_raw != ts_code.split(".")[0]:
                raise ValueError(
                    f"{context}: ts_code/symbol mismatch: {ts_code} vs {symbol_raw!r}"
                )

            name_raw = row[idx_name]
            if not isinstance(name_raw, str) or not name_raw:
                raise ValueError(f"{context}: invalid name: {name_raw!r}")

            area = row[idx_area] if isinstance(row[idx_area], str) else ""
            industry = row[idx_industry] if isinstance(row[idx_industry], str) else ""
            market = row[idx_market] if isinstance(row[idx_market], str) else ""

            list_date_raw = row[idx_list_date]
            list_date: date | None = None
            if list_date_raw is not None and list_date_raw != "":
                list_date = validate_trade_date(list_date_raw, context=context)

            delist_date_raw = row[idx_delist_date]
            delist_date: date | None = None
            if delist_date_raw is not None and delist_date_raw != "":
                delist_date = validate_trade_date(delist_date_raw, context=context)

            status_raw = row[idx_list_status]
            if not isinstance(status_raw, str) or not status_raw:
                raise ValueError(f"{context}: invalid list_status: {status_raw!r}")

            records.append(
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

        return tuple(records)

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
            trade_date = validate_trade_date(row[idx_trade_date], context=context)
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
            trade_date = validate_trade_date(row[idx_trade_date], context=context)
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
