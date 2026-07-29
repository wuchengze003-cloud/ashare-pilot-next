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
        symbols = [item["symbol"] for item in targets]
        if len(symbols) != len(set(symbols)):
            raise ValueError(f"{path}: duplicate target symbol")
        if sum(item["target_weight"] for item in targets) > 1 + 1e-12:
            raise ValueError(f"{path}: target weights exceed 100%")
        if state == "FLAT" and any(item["target_weight"] != 0 for item in targets):
            raise ValueError(f"{path}: FLAT requires zero target weights")
        if state == "ACTIVE" and document["champion"] is None:
            raise ValueError(f"{path}: ACTIVE requires a champion")
        if state == "HOLD" and document["previous_signal_sha256"] is None:
            raise ValueError(f"{path}: HOLD requires a previous signal")

    if contract_id == "stage-health" and parse_datetime(
        document["started_at"]
    ) > parse_datetime(document["finished_at"]):
        raise ValueError(f"{path}: stage finishes before it starts")


def validate_registry() -> tuple[int, int]:
    registry = load_json(CONTRACTS / "registry.json")
    contract_ids: set[str] = set()
    schema_paths: set[Path] = set()
    schema_count = 0
    example_count = 0

    for entry in registry["contracts"]:
        contract_id = entry["contract_id"]
        if contract_id in contract_ids:
            raise ValueError(f"duplicate contract_id: {contract_id}")
        contract_ids.add(contract_id)

        schema_path = CONTRACTS / entry["schema"]
        if schema_path in schema_paths:
            raise ValueError(f"schema registered twice: {schema_path}")
        schema_paths.add(schema_path)

        schema = load_json(schema_path)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        schema_count += 1

        for relative_example in entry["examples"]:
            example_path = CONTRACTS / relative_example
            document = load_json(example_path)
            validator.validate(document)
            if document.get("contract_id") != contract_id:
                raise ValueError(f"{example_path}: contract_id does not match registry")
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
