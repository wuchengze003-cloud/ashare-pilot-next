"""Public command for advancing a simulated account by one committed signal."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from ashare_signal_runner import (
    canonical_json_bytes,
    load_current_run,
    load_dataset_snapshot,
)
from jsonschema import Draft202012Validator, FormatChecker

from .engine import advance_account, build_market_day
from .storage import commit_account_state, load_current_account_state


def _load_object(path: Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return document


def _load_schemas(contracts_root: Path) -> dict[str, dict]:
    registry = _load_object(contracts_root / "registry.json")
    schemas: dict[str, dict] = {}
    for entry in registry["contracts"]:
        schemas[str(entry["contract_id"])] = _load_object(contracts_root / entry["schema"])
    return schemas


def _validate(document: dict, schema: dict) -> None:
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        raise ValueError("contract validation failed: " + errors[0].message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Advance one forward-only simulated account")
    parser.add_argument("--contracts-root", required=True)
    parser.add_argument("--signal-runs-root", required=True)
    parser.add_argument("--signal-head", required=True)
    parser.add_argument("--signal-as-of", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--previous-trade-date", required=True)
    parser.add_argument("--cost-model", required=True)
    parser.add_argument("--market-rules", required=True)
    parser.add_argument("--execution-policy", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--execution-date", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--initial-cash", required=True)
    args = parser.parse_args(argv)

    contracts_root = Path(args.contracts_root)
    schemas = _load_schemas(contracts_root)
    signal_artifacts = load_current_run(
        runs_root=Path(args.signal_runs_root),
        head_path=Path(args.signal_head),
        required_as_of=date.fromisoformat(args.signal_as_of),
        schemas=schemas,
    )
    if signal_artifacts is None:
        raise ValueError("no committed Production Signal is available")
    production_signal = dict(signal_artifacts.production_signal)
    execution_date = date.fromisoformat(args.execution_date)
    dataset_manifest = _load_object(Path(args.dataset_manifest))
    _validate(dataset_manifest, schemas["dataset-manifest"])
    dataset_snapshot = load_dataset_snapshot(
        dataset_manifest=dataset_manifest,
        dataset_root=Path(args.dataset_root),
        as_of=execution_date,
    )
    market_day = build_market_day(
        dataset_snapshot,
        previous_trade_date=date.fromisoformat(args.previous_trade_date),
        execution_date=execution_date,
    )
    cost_model = _load_object(Path(args.cost_model))
    market_rules = _load_object(Path(args.market_rules))
    execution_policy = _load_object(Path(args.execution_policy))
    _validate(market_day, schemas["simulated-market-day"])
    previous_state = load_current_account_state(
        runtime_root=Path(args.runtime_root),
        account_id=args.account_id,
    )
    if previous_state is not None:
        _validate(dict(previous_state), schemas["simulated-account-state"])
    result = advance_account(
        account_id=args.account_id,
        production_signal=production_signal,
        market_day=market_day,
        cost_model=cost_model,
        market_rules=market_rules,
        execution_policy=execution_policy,
        execution_date=execution_date,
        generated_at=datetime.fromisoformat(args.generated_at),
        initial_cash=Decimal(args.initial_cash),
        previous_state=previous_state,
    )
    document = dict(result.document)
    _validate(document, schemas["simulated-account-state"])
    commit_account_state(runtime_root=Path(args.runtime_root), document=document)
    print(canonical_json_bytes(document).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
