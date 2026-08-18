"""Bootstrap a survivor-controlled immutable research dataset.

This command imports the existing full-market vendor export through the Data
Gateway boundary.  The research universe is selected using only information
available on an explicit selection date: the largest non-ST securities by
circulating market value on that date.  Securities that later delisted remain
in the fixed audit set and are supplemented from the configured provider when
their local export is absent.

The source export is treated as non-official mixed provenance.  It is never
overwritten and every selected source file is bound into an auxiliary receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .coverage import SecurityLifecycle, UniverseMembership, audit_historical_coverage
from .dataset_publication import (
    NON_OFFICIAL_SOURCE,
    AuxiliaryArtifact,
    DatasetSourceIdentity,
    PreparedNormalizedDataset,
    PublishedDatasetPaths,
    canonical_json_bytes,
    prepare_normalized_dataset,
    publish_normalized_dataset,
)
from .normalization import NormalizedDailyBar, normalize_daily_bars
from .suspension_evidence import load_authoritative_suspension_evidence
from .tushare_client import TushareClient
from .tushare_models import DailyBarRecord, StockBasicRecord
from .tushare_transport import HttpJsonTransport, TransportRequest

DATASET_FAMILY_ID = "fixed-top-market-cap/v1"
NORMALIZATION_VERSION = "mixed-full-market-bootstrap/v1"
UNIVERSE_POLICY_VERSION = "v1"
SOURCE_VERSION = "mixed-mootdx-teajoin-bootstrap/v2"
SOURCE_HOST = "mixed-bootstrap.local"


@dataclass(frozen=True)
class FullMarketImportResult:
    prepared: PreparedNormalizedDataset
    universe: Mapping[str, Any]
    selected_symbols: tuple[str, ...]
    supplemented_symbols: tuple[str, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_daily_basic_ranking(path: Path, *, selection_date: date) -> dict[str, float]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("selection daily-basic snapshot must be a regular file")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("selection daily-basic snapshot must be an object")
    fields = document.get("fields")
    items = document.get("items")
    if not isinstance(fields, list) or not isinstance(items, list):
        raise ValueError("selection daily-basic snapshot has invalid fields/items")
    required = {"ts_code", "trade_date", "circ_mv"}
    if not required.issubset(fields):
        raise ValueError("selection daily-basic snapshot is missing required fields")
    indexes = {field: fields.index(field) for field in required}
    expected_date = selection_date.strftime("%Y%m%d")
    ranking: dict[str, float] = {}
    for row_number, row in enumerate(items):
        if not isinstance(row, list) or len(row) != len(fields):
            raise ValueError(f"daily-basic row {row_number} has invalid width")
        if str(row[indexes["trade_date"]]) != expected_date:
            raise ValueError("selection daily-basic snapshot contains another trade date")
        symbol = str(row[indexes["ts_code"]])
        raw_value = row[indexes["circ_mv"]]
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            continue
        value = float(raw_value)
        if value <= 0:
            continue
        if symbol in ranking:
            raise ValueError(f"duplicate daily-basic symbol: {symbol}")
        ranking[symbol] = value
    if not ranking:
        raise ValueError("selection daily-basic snapshot produced no ranking")
    return ranking


def select_fixed_universe(
    *,
    security_master: Sequence[StockBasicRecord],
    selection_daily_basic: Path,
    selection_st_symbols: Iterable[str],
    selection_date: date,
    top_n: int,
) -> tuple[str, ...]:
    """Select a fixed audit universe using selection-date information only."""
    if top_n < 20:
        raise ValueError("top_n must be at least 20")
    listed_at_selection: set[str] = set()
    seen_master: set[str] = set()
    for record in security_master:
        if not isinstance(record, StockBasicRecord):
            raise TypeError("security_master must contain StockBasicRecord values")
        if record.ts_code in seen_master:
            raise ValueError(f"duplicate security master symbol: {record.ts_code}")
        seen_master.add(record.ts_code)
        if not record.ts_code.endswith((".SH", ".SZ")):
            continue
        if record.list_date is None or record.list_date > selection_date:
            continue
        if record.delist_date is not None and record.delist_date < selection_date:
            continue
        listed_at_selection.add(record.ts_code)

    st_symbols = set(selection_st_symbols)
    ranking = _parse_daily_basic_ranking(
        selection_daily_basic,
        selection_date=selection_date,
    )
    candidates = [
        (market_value, symbol)
        for symbol, market_value in ranking.items()
        if symbol in listed_at_selection and symbol not in st_symbols
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) < top_n:
        raise ValueError(
            f"only {len(candidates)} non-ST selection-date members are available for top_n={top_n}"
        )
    return tuple(symbol for _, symbol in candidates[:top_n])


def _parse_export_rows(
    path: Path,
    *,
    expected_symbol: str,
    selection_date: date,
    as_of: date,
) -> tuple[NormalizedDailyBar, ...]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"full-market source must be a regular file: {path.name}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("symbol") != expected_symbol:
        raise ValueError(f"full-market source symbol mismatch: {path.name}")
    rows = document.get("data")
    if not isinstance(rows, list):
        raise ValueError(f"full-market source data must be a list: {path.name}")
    parsed: list[NormalizedDailyBar] = []
    seen_dates: set[date] = set()
    for row_number, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"full-market row {row_number} must be an object")
        trade_date = date.fromisoformat(str(row["trade_date"]))
        if trade_date < selection_date or trade_date > as_of:
            continue
        if trade_date in seen_dates:
            raise ValueError(f"duplicate full-market date: {expected_symbol} {trade_date}")
        seen_dates.add(trade_date)
        parsed.append(
            NormalizedDailyBar(
                symbol=expected_symbol,
                trade_date=trade_date,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                amount=row["amount"],
            )
        )
    result = tuple(sorted(parsed))
    if not result or result[0].trade_date != selection_date:
        raise ValueError(f"selected member lacks a bar on selection_date: {expected_symbol}")
    return result


def prepare_full_market_import(
    *,
    source_root: Path,
    selection_daily_basic: Path,
    security_master: Sequence[StockBasicRecord],
    selection_st_symbols: Iterable[str],
    current_st_symbols: Iterable[str],
    security_master_rejections: Iterable[str],
    selection_date: date,
    as_of: date,
    generated_at: datetime,
    top_n: int,
    supplement_loader: Callable[[str, date, date], Sequence[NormalizedDailyBar]],
    trading_days: tuple[date, ...],
    suspension_keys: frozenset[tuple[str, date]],
    authoritative_suspension_artifacts: tuple[AuxiliaryArtifact, ...] = (),
) -> FullMarketImportResult:
    """Prepare immutable bytes and a current PIT universe without writing."""
    source_root = Path(source_root)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("full-market source root must be a regular directory")
    if as_of < selection_date:
        raise ValueError("as_of cannot precede selection_date")
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    selection_st_symbols = tuple(selection_st_symbols)
    current_st_symbols = tuple(current_st_symbols)
    security_master_rejections = tuple(security_master_rejections)

    selected = select_fixed_universe(
        security_master=security_master,
        selection_daily_basic=selection_daily_basic,
        selection_st_symbols=selection_st_symbols,
        selection_date=selection_date,
        top_n=top_n,
    )
    records: list[NormalizedDailyBar] = []
    source_hashes: dict[str, str] = {}
    gap_supplement_hashes: dict[str, str] = {}
    last_dates: dict[str, date] = {}
    supplemented: list[str] = []
    for symbol in selected:
        path = source_root / f"{symbol.split('.', 1)[0]}.json"
        if path.exists():
            rows = _parse_export_rows(
                path,
                expected_symbol=symbol,
                selection_date=selection_date,
                as_of=as_of,
            )
            source_hashes[symbol] = _sha256(path)
        else:
            rows = tuple(supplement_loader(symbol, selection_date, as_of))
            if not rows or rows[0].trade_date != selection_date:
                raise ValueError(f"supplement lacks selection-date history: {symbol}")
            if any(row.symbol != symbol or row.trade_date > as_of for row in rows):
                raise ValueError(f"supplement is not bound to request: {symbol}")
            source_hashes[symbol] = hashlib.sha256(
                canonical_json_bytes([row.document for row in rows])
            ).hexdigest()
            supplemented.append(symbol)
        records.extend(rows)
        last_dates[symbol] = rows[-1].trade_date

    lifecycle_by_symbol = {record.ts_code: record for record in security_master}
    missing_list_dates = sorted(
        symbol for symbol in selected if lifecycle_by_symbol[symbol].list_date is None
    )
    if missing_list_dates:
        raise ValueError(f"security master lacks list dates: {missing_list_dates}")
    memberships = tuple(
        UniverseMembership(symbol=symbol, valid_from=selection_date, valid_to=None)
        for symbol in selected
    )
    lifecycles = tuple(
        SecurityLifecycle(
            symbol=symbol,
            listed_on=lifecycle_by_symbol[symbol].list_date,
            delisted_on=lifecycle_by_symbol[symbol].delist_date,
        )
        for symbol in selected
    )

    def audit_records():
        return audit_historical_coverage(
            audit_id=f"bootstrap-top{top_n}-{selection_date.isoformat()}-{as_of.isoformat()}",
            universe_policy_id=f"fixed-top{top_n}-market-cap",
            universe_policy_version=UNIVERSE_POLICY_VERSION,
            window_start=selection_date,
            window_end=as_of,
            generated_at=generated_at.astimezone(UTC),
            trading_days=trading_days,
            memberships=memberships,
            lifecycles=lifecycles,
            bar_keys=frozenset((record.symbol, record.trade_date) for record in records),
            suspension_keys=suspension_keys,
            expected_member_count=None,
            provenance_warnings=("NON_OFFICIAL_MIXED_BOOTSTRAP_SOURCE",),
        )

    coverage_audit = audit_records()
    missing_by_symbol: dict[str, set[date]] = {}
    for gap in coverage_audit.missing_member_days:
        missing_by_symbol.setdefault(gap.symbol, set()).add(gap.trade_date)
    for symbol, missing_dates in sorted(missing_by_symbol.items()):
        supplied = tuple(supplement_loader(symbol, min(missing_dates), max(missing_dates)))
        additions = tuple(row for row in supplied if row.trade_date in missing_dates)
        supplied_dates = {row.trade_date for row in additions if row.symbol == symbol}
        if supplied_dates != missing_dates or len(additions) != len(missing_dates):
            continue
        records.extend(additions)
        last_dates[symbol] = max(last_dates[symbol], max(missing_dates))
        supplemented.append(symbol)
        gap_supplement_hashes[symbol] = hashlib.sha256(
            canonical_json_bytes([row.document for row in sorted(additions)])
        ).hexdigest()
    if missing_by_symbol:
        coverage_audit = audit_records()
    if not coverage_audit.passed:
        samples = ",".join(
            f"{gap.symbol}:{gap.trade_date.isoformat()}"
            for gap in coverage_audit.missing_member_days[:10]
        )
        raise ValueError(
            "bootstrap member-day coverage failed: "
            + ",".join(coverage_audit.reason_codes)
            + f" missing={len(coverage_audit.missing_member_days)} sample={samples}"
        )

    receipt = {
        "receipt_id": "full-market-bootstrap-import/v2",
        "selection_date": selection_date.isoformat(),
        "as_of": as_of.isoformat(),
        "top_n": top_n,
        "selection_daily_basic_sha256": _sha256(selection_daily_basic),
        "selected_symbols": list(selected),
        "selection_st_symbols": sorted(set(selection_st_symbols)),
        "current_st_symbols": sorted(set(current_st_symbols)),
        "security_master_rejections": sorted(set(security_master_rejections)),
        "source_file_sha256": source_hashes,
        "gap_supplement_sha256": gap_supplement_hashes,
        "supplemented_symbols": sorted(set(supplemented)),
        "provenance": [
            "historical full-market export originally acquired through mootdx",
            (
                "subsequent daily refreshes and missing delisted histories "
                "from teajoin-compatible HTTPS"
            ),
        ],
    }
    prepared = prepare_normalized_dataset(
        records,
        as_of=as_of,
        generated_at=generated_at.astimezone(UTC),
        source_identity=DatasetSourceIdentity(
            base_url_host=SOURCE_HOST,
            is_official_vendor=False,
            manifest_source=NON_OFFICIAL_SOURCE,
        ),
        source_version=SOURCE_VERSION,
        dataset_family_id=DATASET_FAMILY_ID,
        normalization_version=NORMALIZATION_VERSION,
        auxiliary_artifacts=(
            AuxiliaryArtifact(
                relative_path="raw/import-receipt.json",
                content=canonical_json_bytes(receipt),
            ),
            AuxiliaryArtifact(
                relative_path="raw/coverage-audit.json",
                content=canonical_json_bytes(coverage_audit.document),
            ),
            *authoritative_suspension_artifacts,
        ),
    )

    current_st = set(current_st_symbols)
    members: list[dict[str, Any]] = []
    for symbol in selected:
        reasons = ["FIXED_TOP_MARKET_CAP_AT_SELECTION", "NON_ST_AT_SELECTION"]
        eligible = last_dates[symbol] == as_of and symbol not in current_st
        if last_dates[symbol] == as_of:
            reasons.append("DATA_CURRENT")
        else:
            reasons.append("NO_BAR_ON_AS_OF")
        if symbol in current_st:
            reasons.append("CURRENT_ST_EXCLUDED")
        members.append(
            {
                "symbol": symbol,
                "valid_from": selection_date.isoformat(),
                "valid_to": None,
                "eligible": eligible,
                "reason_codes": sorted(reasons),
            }
        )
    policy = f"fixed-top{top_n}-market-cap"
    universe = {
        "contract_id": "universe",
        "schema_version": "2.0.0",
        "universe_id": f"fixed-top{top_n}-pit/v1",
        "universe_policy_id": policy,
        "universe_policy_version": UNIVERSE_POLICY_VERSION,
        "as_of": as_of.isoformat(),
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "source": "full-market-bootstrap-import/v2",
        "source_version": "v1",
        "quality_status": "pass",
        "members": members,
    }
    return FullMarketImportResult(
        prepared=prepared,
        universe=universe,
        selected_symbols=selected,
        supplemented_symbols=tuple(sorted(set(supplemented))),
    )


def fetch_stock_st_symbols(transport: HttpJsonTransport, *, trade_date: date) -> set[str]:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise ValueError("TUSHARE_TOKEN is not set")
    response = transport.send(
        TransportRequest(
            api_name="stock_st",
            params=(("trade_date", trade_date.strftime("%Y%m%d")),),
            fields=("ts_code", "name", "trade_date", "type", "type_name"),
            offset=0,
            limit=1000,
        ),
        token=token,
    )
    if len(response.items) >= 1000:
        raise ValueError("stock_st response reached the page limit")
    indexes = {field: response.fields.index(field) for field in response.fields}
    required = {"ts_code", "trade_date"}
    if not required.issubset(indexes):
        raise ValueError("stock_st response is missing required fields")
    expected = trade_date.strftime("%Y%m%d")
    symbols: set[str] = set()
    for row in response.items:
        if str(row[indexes["trade_date"]]) != expected:
            raise ValueError("stock_st response escaped requested date")
        symbol = str(row[indexes["ts_code"]])
        if symbol in symbols:
            raise ValueError(f"duplicate stock_st symbol: {symbol}")
        symbols.add(symbol)
    return symbols


def load_security_master(
    client: TushareClient,
) -> tuple[tuple[StockBasicRecord, ...], tuple[str, ...]]:
    records: list[StockBasicRecord] = []
    rejections: list[str] = []
    for status in ("L", "D", "P"):
        batch = client.fetch_stock_basic_reconciled(list_status=status)
        records.extend(batch.accepted)
        rejections.extend(
            f"{status}:{item.reason}:{item.row_fingerprint}" for item in batch.rejected
        )
    return tuple(records), tuple(sorted(rejections))


def _supplement_loader(
    client: TushareClient,
) -> Callable[[str, date, date], Sequence[NormalizedDailyBar]]:
    def load(symbol: str, start: date, end: date) -> Sequence[NormalizedDailyBar]:
        raw: tuple[DailyBarRecord, ...] = client.fetch_daily(
            ts_code=symbol,
            start_date=start,
            end_date=end,
        )
        return normalize_daily_bars(
            raw,
            requested_symbols=(symbol,),
            window_start=start,
            window_end=end,
        )

    return load


def publish_full_market_import(
    *,
    publication_root: Path,
    universe_out: Path,
    result: FullMarketImportResult,
) -> PublishedDatasetPaths:
    paths = publish_normalized_dataset(
        publication_root=publication_root,
        prepared=result.prepared,
    )
    universe_out = Path(universe_out)
    if universe_out.exists() or universe_out.is_symlink():
        raise FileExistsError(f"universe output already exists: {universe_out}")
    universe_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = universe_out.with_name(f".{universe_out.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(result.universe) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, universe_out)
    finally:
        temporary.unlink(missing_ok=True)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a fixed-universe full-market dataset")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--selection-daily-basic", required=True, type=Path)
    parser.add_argument("--publication-root", required=True, type=Path)
    parser.add_argument("--universe-out", required=True, type=Path)
    parser.add_argument("--selection-date", required=True, type=date.fromisoformat)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--top-n", type=int, default=800)
    parser.add_argument(
        "--authoritative-suspension-evidence",
        action="append",
        default=[],
        type=Path,
        help="human-reviewed SSE evidence JSON with a hash-bound sibling source file",
    )
    args = parser.parse_args(argv)

    generated_at = datetime.fromisoformat(args.generated_at.replace("Z", "+00:00"))
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated-at must be timezone-aware")
    transport = HttpJsonTransport()
    client = TushareClient(transport)
    security_master, security_master_rejections = load_security_master(client)
    selection_st_symbols = fetch_stock_st_symbols(transport, trade_date=args.selection_date)
    selected_symbols = select_fixed_universe(
        security_master=security_master,
        selection_daily_basic=args.selection_daily_basic,
        selection_st_symbols=selection_st_symbols,
        selection_date=args.selection_date,
        top_n=args.top_n,
    )
    calendar = client.fetch_trade_cal(
        exchange="SSE", start_date=args.selection_date, end_date=args.as_of
    )
    trading_days = tuple(record.cal_date for record in calendar if record.is_open)
    suspension_records = client.fetch_suspend_d(
        ts_codes=selected_symbols,
        start_date=args.selection_date,
        end_date=args.as_of,
    )
    authoritative_suspensions = load_authoritative_suspension_evidence(
        tuple(args.authoritative_suspension_evidence)
    )
    result = prepare_full_market_import(
        source_root=args.source_root,
        selection_daily_basic=args.selection_daily_basic,
        security_master=security_master,
        selection_st_symbols=selection_st_symbols,
        current_st_symbols=fetch_stock_st_symbols(transport, trade_date=args.as_of),
        security_master_rejections=security_master_rejections,
        selection_date=args.selection_date,
        as_of=args.as_of,
        generated_at=generated_at,
        top_n=args.top_n,
        supplement_loader=_supplement_loader(client),
        trading_days=trading_days,
        suspension_keys=(
            frozenset(
                (record.ts_code, record.trade_date)
                for record in suspension_records
                if record.suspend_type == "S"
            )
            | authoritative_suspensions.suspension_keys
        ),
        authoritative_suspension_artifacts=authoritative_suspensions.artifacts,
    )
    paths = publish_full_market_import(
        publication_root=args.publication_root,
        universe_out=args.universe_out,
        result=result,
    )
    print(
        canonical_json_bytes(
            {
                "dataset_id": paths.dataset_id,
                "dataset_dir": str(paths.dataset_dir),
                "manifest_path": str(paths.manifest_path),
                "universe_path": str(args.universe_out),
                "selected_symbols": len(result.selected_symbols),
                "supplemented_symbols": list(result.supplemented_symbols),
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
