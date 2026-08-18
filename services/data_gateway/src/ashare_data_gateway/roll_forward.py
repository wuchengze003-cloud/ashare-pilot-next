"""Roll an immutable fixed-universe dataset forward by explicit trade date."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from .coverage import (
    SecurityLifecycle,
    UniverseMembership,
    audit_historical_coverage,
)
from .dataset_publication import (
    AuxiliaryArtifact,
    LoadedNormalizedDataset,
    PreparedNormalizedDataset,
    PublishedDatasetPaths,
    canonical_json_bytes,
    load_published_dataset,
    prepare_normalized_dataset,
    publish_normalized_dataset,
    source_identity_from_base_url,
)
from .full_market_import import fetch_stock_st_symbols, load_security_master
from .normalization import NormalizedDailyBar, normalize_daily_bars
from .suspension_evidence import load_authoritative_suspension_evidence
from .tushare_client import TushareClient
from .tushare_transport import HttpJsonTransport

SOURCE_VERSION = "fixed-universe-roll-forward/v2"
MINIMUM_AS_OF_COVERAGE = 0.90


def prepare_roll_forward(
    *,
    parent: LoadedNormalizedDataset,
    universe: Mapping[str, Any],
    incremental_records: Sequence[NormalizedDailyBar],
    current_st_symbols: Iterable[str],
    as_of: date,
    generated_at: datetime,
    source_base_url: str,
    trading_days: tuple[date, ...],
    suspension_keys: frozenset[tuple[str, date]],
    lifecycles: tuple[SecurityLifecycle, ...],
    authoritative_suspension_artifacts: tuple[AuxiliaryArtifact, ...] = (),
) -> tuple[PreparedNormalizedDataset, dict[str, Any]]:
    parent_manifest = parent.manifest
    parent_as_of = date.fromisoformat(str(parent_manifest["as_of"]))
    if as_of <= parent_as_of:
        raise ValueError("roll-forward as_of must follow the parent dataset")
    if str(universe.get("as_of")) != parent_as_of.isoformat():
        raise ValueError("parent universe as_of does not match the parent dataset")
    members = universe.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("parent universe members are unavailable")
    symbols = {str(member["symbol"]) for member in members}
    parent_symbols = {record.symbol for record in parent.records}
    if symbols != parent_symbols:
        raise ValueError("parent universe members do not match the parent dataset")

    seen: set[tuple[str, date]] = set()
    incremental: list[NormalizedDailyBar] = []
    for record in incremental_records:
        if record.symbol not in symbols:
            raise ValueError(f"incremental row is outside the fixed universe: {record.symbol}")
        if not parent_as_of < record.trade_date <= as_of:
            raise ValueError("incremental row escaped the roll-forward window")
        key = (record.symbol, record.trade_date)
        if key in seen:
            raise ValueError(f"duplicate incremental row: {key}")
        seen.add(key)
        incremental.append(record)

    as_of_symbols = {record.symbol for record in incremental if record.trade_date == as_of}
    coverage = len(as_of_symbols) / len(symbols)
    if coverage < MINIMUM_AS_OF_COVERAGE:
        raise ValueError(
            f"as_of coverage {coverage:.3f} is below {MINIMUM_AS_OF_COVERAGE:.2f}"
        )
    memberships = tuple(
        UniverseMembership(
            symbol=str(member["symbol"]),
            valid_from=date.fromisoformat(str(member["valid_from"])),
            valid_to=(
                date.fromisoformat(str(member["valid_to"]))
                if member.get("valid_to")
                else None
            ),
        )
        for member in members
    )
    coverage_audit = audit_historical_coverage(
        audit_id=f"roll-forward-{parent.dataset_id[-12:]}-{as_of.isoformat()}",
        universe_policy_id=str(universe["universe_policy_id"]),
        universe_policy_version=str(universe["universe_policy_version"]),
        window_start=parent_as_of + timedelta(days=1),
        window_end=as_of,
        generated_at=generated_at.astimezone(UTC),
        trading_days=trading_days,
        memberships=memberships,
        lifecycles=lifecycles,
        bar_keys=frozenset((record.symbol, record.trade_date) for record in incremental),
        suspension_keys=suspension_keys,
        expected_member_count=None,
    )
    if not coverage_audit.passed:
        samples = ",".join(
            f"{gap.symbol}:{gap.trade_date.isoformat()}"
            for gap in coverage_audit.missing_member_days[:10]
        )
        raise ValueError(
            "roll-forward member-day coverage failed: "
            + ",".join(coverage_audit.reason_codes)
            + f" missing={len(coverage_audit.missing_member_days)} sample={samples}"
        )
    parent_manifest_sha256 = hashlib.sha256(parent.manifest_bytes).hexdigest()
    receipt = {
        "receipt_id": SOURCE_VERSION,
        "parent_dataset_id": parent.dataset_id,
        "parent_manifest_sha256": parent_manifest_sha256,
        "window_start": (parent_as_of + timedelta(days=1)).isoformat(),
        "window_end": as_of.isoformat(),
        "member_count": len(symbols),
        "as_of_member_count": len(as_of_symbols),
        "as_of_coverage": round(coverage, 6),
        "incremental_row_count": len(incremental),
        "incremental_sha256": hashlib.sha256(
            canonical_json_bytes([record.document for record in sorted(incremental)])
        ).hexdigest(),
    }
    prepared = prepare_normalized_dataset(
        (*parent.records, *incremental),
        as_of=as_of,
        generated_at=generated_at.astimezone(UTC),
        source_identity=source_identity_from_base_url(source_base_url),
        source_version=SOURCE_VERSION,
        dataset_family_id=str(parent_manifest["dataset_family_id"]),
        normalization_version=str(parent_manifest["normalization_version"]),
        parent_manifest_sha256=parent_manifest_sha256,
        auxiliary_artifacts=(
            AuxiliaryArtifact(
                relative_path="raw/roll-forward-receipt.json",
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
    new_members: list[dict[str, Any]] = []
    for member in members:
        symbol = str(member["symbol"])
        eligible = symbol in as_of_symbols and symbol not in current_st
        reasons = ["FIXED_UNIVERSE_MEMBER"]
        reasons.append("DATA_CURRENT" if symbol in as_of_symbols else "NO_BAR_ON_AS_OF")
        if symbol in current_st:
            reasons.append("CURRENT_ST_EXCLUDED")
        new_members.append(
            {
                "symbol": symbol,
                "valid_from": member["valid_from"],
                "valid_to": member["valid_to"],
                "eligible": eligible,
                "reason_codes": sorted(reasons),
            }
        )
    new_universe = {
        "contract_id": "universe",
        "schema_version": "2.0.0",
        "universe_id": universe["universe_id"],
        "universe_policy_id": universe["universe_policy_id"],
        "universe_policy_version": universe["universe_policy_version"],
        "as_of": as_of.isoformat(),
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "source": SOURCE_VERSION,
        "source_version": "v1",
        "quality_status": "pass",
        "members": new_members,
    }
    return prepared, new_universe


def _write_universe(path: Path, document: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"universe output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(document))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_roll_forward(
    *,
    publication_root: Path,
    universe_out: Path,
    prepared: PreparedNormalizedDataset,
    universe: Mapping[str, Any],
) -> PublishedDatasetPaths:
    paths = publish_normalized_dataset(publication_root=publication_root, prepared=prepared)
    try:
        _write_universe(universe_out, universe)
    except BaseException:
        # The dataset remains a valid immutable publication. It is intentionally
        # not made current by any pointer, and the caller receives a hard failure.
        raise
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Roll a fixed universe dataset forward")
    parser.add_argument("--publication-root", required=True, type=Path)
    parser.add_argument("--parent-dataset-id", required=True)
    parser.add_argument("--parent-universe", required=True, type=Path)
    parser.add_argument("--universe-out", required=True, type=Path)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--generated-at", required=True)
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
    parent = load_published_dataset(
        publication_root=args.publication_root,
        dataset_id=args.parent_dataset_id,
    )
    universe = json.loads(args.parent_universe.read_text(encoding="utf-8"))
    if not isinstance(universe, dict):
        raise ValueError("parent universe must be an object")
    parent_as_of = date.fromisoformat(str(parent.manifest["as_of"]))
    transport = HttpJsonTransport()
    client = TushareClient(transport)
    calendar = client.fetch_trade_cal(
        exchange="SSE",
        start_date=parent_as_of + timedelta(days=1),
        end_date=args.as_of,
    )
    trading_days = tuple(record.cal_date for record in calendar if record.is_open)
    if args.as_of not in trading_days:
        raise ValueError("roll-forward as_of is not an open trading day")
    incremental: list[NormalizedDailyBar] = []
    symbols = tuple(sorted(str(member["symbol"]) for member in universe["members"]))
    for symbol in symbols:
        raw = client.fetch_daily(
            ts_code=symbol,
            start_date=parent_as_of + timedelta(days=1),
            end_date=args.as_of,
        )
        incremental.extend(
            normalize_daily_bars(
                raw,
                requested_symbols=(symbol,),
                window_start=parent_as_of + timedelta(days=1),
                window_end=args.as_of,
            )
        )
    suspension_records = client.fetch_suspend_d(
        ts_codes=symbols,
        start_date=parent_as_of + timedelta(days=1),
        end_date=args.as_of,
    )
    authoritative_suspensions = load_authoritative_suspension_evidence(
        tuple(args.authoritative_suspension_evidence)
    )
    security_master, security_master_rejections = load_security_master(client)
    if security_master_rejections:
        raise ValueError("security master contains rejected lifecycle rows")
    lifecycle_by_symbol = {}
    for record in security_master:
        if record.ts_code in lifecycle_by_symbol:
            raise ValueError(f"duplicate security lifecycle: {record.ts_code}")
        lifecycle_by_symbol[record.ts_code] = record
    missing_lifecycles = sorted(set(symbols) - lifecycle_by_symbol.keys())
    if missing_lifecycles:
        raise ValueError(f"security master lacks fixed members: {missing_lifecycles}")
    missing_list_dates = sorted(
        symbol for symbol in symbols if lifecycle_by_symbol[symbol].list_date is None
    )
    if missing_list_dates:
        raise ValueError(f"security master lacks list dates: {missing_list_dates}")
    prepared, new_universe = prepare_roll_forward(
        parent=parent,
        universe=universe,
        incremental_records=incremental,
        current_st_symbols=fetch_stock_st_symbols(transport, trade_date=args.as_of),
        as_of=args.as_of,
        generated_at=generated_at,
        source_base_url=transport.base_url,
        trading_days=trading_days,
        suspension_keys=(
            frozenset(
                (record.ts_code, record.trade_date)
                for record in suspension_records
                if record.suspend_type == "S"
            )
            | authoritative_suspensions.suspension_keys
        ),
        lifecycles=tuple(
            SecurityLifecycle(
                symbol=symbol,
                listed_on=lifecycle_by_symbol[symbol].list_date,
                delisted_on=lifecycle_by_symbol[symbol].delist_date,
            )
            for symbol in symbols
        ),
        authoritative_suspension_artifacts=authoritative_suspensions.artifacts,
    )
    paths = publish_roll_forward(
        publication_root=args.publication_root,
        universe_out=args.universe_out,
        prepared=prepared,
        universe=new_universe,
    )
    print(
        canonical_json_bytes(
            {
                "dataset_id": paths.dataset_id,
                "manifest_path": str(paths.manifest_path),
                "dataset_dir": str(paths.dataset_dir),
                "universe_path": str(args.universe_out),
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
