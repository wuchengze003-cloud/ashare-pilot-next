"""Validate contract schemas, examples, and cross-field semantics."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_semantics(contract_id: str, document: dict[str, Any], path: Path) -> None:
    if contract_id == "coverage-audit" and document.get("schema_version") == "2.0.0":
        expected_delisted = document["expected_delisted_member_days"]
        expected_delisted_count = document["expected_delisted_member_day_count"]
        if expected_delisted_count != len(expected_delisted):
            raise ValueError(
                f"{path}: expected-delisted count does not match member-day records"
            )

        classified_member_days = (
            document["bar_member_days"]
            + document["suspended_member_days"]
            + expected_delisted_count
            + len(document["missing_member_days"])
        )
        if classified_member_days != document["expected_member_days"]:
            raise ValueError(
                f"{path}: classified member-days do not reconcile to expected total"
            )

        expected_delisted_keys = {
            (item["symbol"], item["trade_date"]) for item in expected_delisted
        }
        missing_keys = {
            (item["symbol"], item["trade_date"])
            for item in document["missing_member_days"]
        }
        if expected_delisted_keys & missing_keys:
            raise ValueError(
                f"{path}: member-day cannot be both expected-delisted and missing"
            )

    if contract_id == "cost-model":
        source_ids = [source["source_id"] for source in document["sources"]]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"{path}: duplicate cost source_id")
        known_source_ids = set(source_ids)

        segment_ids: set[str] = set()
        by_market: dict[str, list[tuple[date, date | None, str]]] = {}
        for segment in document["segments"]:
            segment_id = segment["segment_id"]
            if segment_id in segment_ids:
                raise ValueError(f"{path}: duplicate cost segment_id {segment_id}")
            segment_ids.add(segment_id)

            effective_from = parse_date(segment["effective_from"])
            effective_to = (
                parse_date(segment["effective_to"]) if segment["effective_to"] else None
            )
            if effective_to is not None and effective_from > effective_to:
                raise ValueError(f"{path}: cost segment {segment_id} ends before it starts")

            unknown_sources = sorted(set(segment["source_ids"]) - known_source_ids)
            if unknown_sources:
                raise ValueError(
                    f"{path}: cost segment {segment_id} has unknown sources "
                    f"{unknown_sources}"
                )
            for market in segment["markets"]:
                by_market.setdefault(market, []).append(
                    (effective_from, effective_to, segment_id)
                )

        for market, periods in by_market.items():
            ordered = sorted(periods, key=lambda period: (period[0], period[2]))
            for index, (_effective_from, effective_to, segment_id) in enumerate(ordered):
                is_last = index == len(ordered) - 1
                if effective_to is None and not is_last:
                    raise ValueError(
                        f"{path}: open-ended cost segment {segment_id} is not last "
                        f"for {market}"
                    )
                if is_last:
                    if effective_to is not None:
                        raise ValueError(
                            f"{path}: last cost segment for {market} must be open-ended"
                        )
                    continue
                next_from = ordered[index + 1][0]
                if effective_to is None or effective_to.toordinal() + 1 != next_from.toordinal():
                    raise ValueError(
                        f"{path}: cost segments for {market} must be contiguous "
                        "and non-overlapping"
                    )

    if contract_id == "universe":
        as_of = parse_date(document["as_of"])
        symbols: set[str] = set()
        for member in document["members"]:
            symbol = member["symbol"]
            if symbol in symbols:
                raise ValueError(f"{path}: duplicate universe symbol {symbol}")
            symbols.add(symbol)
            valid_from = parse_date(member["valid_from"])
            valid_to = parse_date(member["valid_to"]) if member["valid_to"] else None
            if valid_from > as_of or (valid_to is not None and as_of > valid_to):
                raise ValueError(f"{path}: member validity does not cover as_of")

    if contract_id == "experiment-config":
        windows = document["windows"]
        development_end = parse_date(windows["development"]["end"])
        validation_start = parse_date(windows["validation"]["start"])
        validation_end = parse_date(windows["validation"]["end"])
        final_start = parse_date(windows["final"]["start"])
        if development_end >= validation_start or validation_end >= final_start:
            raise ValueError(f"{path}: experiment windows must be ordered and disjoint")
        for name, window in windows.items():
            if parse_date(window["start"]) > parse_date(window["end"]):
                raise ValueError(f"{path}: {name} starts after it ends")

    if contract_id == "production-signal":
        state = document["state"]
        targets = document["target_positions"]
        champion = document["champion"]
        champion_hash = document["contract_set"]["champion_sha256"]
        symbols = [item["symbol"] for item in targets]
        if len(symbols) != len(set(symbols)):
            raise ValueError(f"{path}: duplicate target symbol")
        if sum(item["target_weight"] for item in targets) > 1 + 1e-12:
            raise ValueError(f"{path}: target weights exceed 100%")
        if parse_date(document["latest_complete_date"]) > parse_date(document["as_of"]):
            raise ValueError(f"{path}: latest_complete_date cannot be later than as_of")
        generated_at = parse_datetime(document["generated_at"])
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError(f"{path}: generated_at must include a timezone")
        if generated_at.utcoffset().total_seconds() != 0:
            raise ValueError(f"{path}: generated_at must use UTC")
        if state == "FLAT" and targets:
            raise ValueError(f"{path}: FLAT requires an empty target")
        if state == "ACTIVE" and champion is None:
            raise ValueError(f"{path}: ACTIVE requires a champion")
        if state in {"HOLD", "REDUCE_ONLY"} and document["previous_signal_sha256"] is None:
            raise ValueError(f"{path}: {state} requires a previous signal")
        if champion is None and champion_hash is not None:
            raise ValueError(f"{path}: champion hash requires a champion")
        if champion is not None and champion_hash != champion["sha256"]:
            raise ValueError(f"{path}: champion hash does not match champion")

    if contract_id == "stage-health" and parse_datetime(
        document["started_at"]
    ) > parse_datetime(document["finished_at"]):
        raise ValueError(f"{path}: stage finishes before it starts")

    if contract_id == "web-state":
        generated_at = parse_datetime(document["generated_at"])
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError(f"{path}: generated_at must include a timezone")
        if generated_at.utcoffset().total_seconds() != 0:
            raise ValueError(f"{path}: generated_at must use UTC")
        as_of = parse_date(document["as_of"])
        latest_trade_date = document["data_source"]["latest_trade_date"]
        if latest_trade_date is not None and parse_date(latest_trade_date) > as_of:
            raise ValueError(f"{path}: latest_trade_date cannot be later than as_of")
        window = document["model"]["backtest_window"]
        if parse_date(window["start"]) > parse_date(window["end"]):
            raise ValueError(f"{path}: backtest window starts after it ends")
        if parse_date(document["model"]["training_cutoff"]) > parse_date(
            document["model"]["validation_cutoff"]
        ):
            raise ValueError(f"{path}: training cutoff cannot follow validation cutoff")
        ranks = [item["rank"] for item in document["rankings"]]
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError(f"{path}: ranking ranks must be a 1..N sequence")
        symbols = [item["symbol"] for item in document["rankings"]]
        if len(symbols) != len(set(symbols)):
            raise ValueError(f"{path}: duplicate ranking symbol")
        portfolio = document["portfolio"]
        position_value = sum(item["market_value"] for item in portfolio["positions"])
        if abs(position_value - portfolio["market_value"]) > 0.01:
            raise ValueError(f"{path}: position values do not reconcile to market_value")
        accounted = portfolio["cash"] + portfolio["market_value"]
        if abs(accounted - portfolio["total_assets"]) > 0.01:
            raise ValueError(f"{path}: cash plus market_value must equal total_assets")
        for item in portfolio["positions"]:
            if item["locked_shares"] > item["shares"]:
                raise ValueError(f"{path}: locked shares exceed held shares")

        def _split_bounds(raw: str) -> tuple[date, date]:
            start_text, _, end_text = raw.partition("/")
            return parse_date(start_text), parse_date(end_text)

        splits = document["performance"]["splits"]
        in_sample_end = _split_bounds(splits["in_sample"])[1]
        validation_start, validation_end = _split_bounds(splits["validation"])
        out_of_sample_start = _split_bounds(splits["out_of_sample"])[0]
        if not (in_sample_end < validation_start <= validation_end < out_of_sample_start):
            raise ValueError(f"{path}: splits must be ordered and disjoint")
        leak_checks = document["performance"]["leak_checks"]
        check_ids = [item["check_id"] for item in leak_checks]
        if len(check_ids) != len(set(check_ids)):
            raise ValueError(f"{path}: duplicate leak check_id")


def validate_registry() -> tuple[int, int]:
    registry = load_json(CONTRACTS / "registry.json")
    contract_versions: set[tuple[str, str]] = set()
    schema_paths: set[Path] = set()
    schema_count = 0
    example_count = 0

    for entry in registry["contracts"]:
        contract_id = entry["contract_id"]
        schema_version = entry["schema_version"]
        contract_version = (contract_id, schema_version)
        if contract_version in contract_versions:
            raise ValueError(
                f"duplicate contract version: {contract_id} {schema_version}"
            )
        contract_versions.add(contract_version)

        schema_path = CONTRACTS / entry["schema"]
        if schema_path in schema_paths:
            raise ValueError(f"schema registered twice: {schema_path}")
        schema_paths.add(schema_path)

        schema = load_json(schema_path)
        Draft202012Validator.check_schema(schema)
        schema_contract_id = schema.get("properties", {}).get("contract_id", {}).get("const")
        if schema_contract_id != contract_id:
            raise ValueError(f"{schema_path}: contract_id does not match registry")
        schema_contract_version = (
            schema.get("properties", {}).get("schema_version", {}).get("const")
        )
        if schema_contract_version != schema_version:
            raise ValueError(f"{schema_path}: schema_version does not match registry")
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        schema_count += 1

        for relative_example in entry["examples"]:
            example_path = CONTRACTS / relative_example
            document = load_json(example_path)
            validator.validate(document)
            if document.get("contract_id") != contract_id:
                raise ValueError(f"{example_path}: contract_id does not match registry")
            if document.get("schema_version") != schema_version:
                raise ValueError(f"{example_path}: schema_version does not match registry")
            validate_semantics(contract_id, document, example_path)
            example_count += 1

    registered = {path.resolve() for path in schema_paths}
    actual = {path.resolve() for path in (CONTRACTS / "schemas").glob("*.schema.json")}
    if registered != actual:
        missing = sorted(str(path.relative_to(ROOT)) for path in actual - registered)
        stale = sorted(str(path.relative_to(ROOT)) for path in registered - actual)
        raise ValueError(f"contract registry mismatch; unregistered={missing}, missing={stale}")

    return schema_count, example_count


def main() -> None:
    schema_count, example_count = validate_registry()
    print(f"PASS: validated {schema_count} schemas and {example_count} examples")


if __name__ == "__main__":
    main()
