"""Public command: assemble the read-only web-state contract document.

Consumes the research report, the latest committed production signal,
and the dataset manifest. Validates against the web-state 1.0 schema
before writing. No wall-clock reads; generated_at is explicit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from .features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[4]


def _load_json_object(path: Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"document must be an object: {path}")
    return document


def load_web_state_schema() -> dict[str, Any]:
    registry = _load_json_object(ROOT / "contracts/registry.json")
    for entry in registry["contracts"]:
        if entry["contract_id"] == "web-state":
            return _load_json_object(ROOT / "contracts" / entry["schema"])
    raise ValueError("web-state contract is not registered")


def _read_latest_signal(runs_root: Path, head_path: Path) -> dict[str, Any] | None:
    head_path = Path(head_path)
    if not head_path.is_file():
        return None
    head = _load_json_object(head_path)
    signal_path = Path(runs_root) / str(head["run_id"]) / "production-signal.json"
    if not signal_path.is_file():
        return None
    return _load_json_object(signal_path)


def build_web_state(
    *,
    research_report: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    champion: Mapping[str, Any],
    latest_signal: Mapping[str, Any] | None,
    mode: str,
    is_official_vendor: bool,
    generated_at: str,
    extra_notices: tuple[str, ...] = (),
) -> dict[str, Any]:
    if mode not in {"live-fetch", "m1-replay", "synthetic"}:
        raise ValueError(f"unsupported web-state mode: {mode}")
    portfolio = research_report["portfolio"]
    files = dataset_manifest["files"]
    latest_trade_date = str(files[0]["max_trade_date"])
    as_of = str(research_report["as_of"])

    notices: list[str] = []
    if not is_official_vendor:
        notices.append("NON_OFFICIAL_VENDOR_ENDPOINT")
    if mode == "m1-replay":
        notices.append("REPLAY_OF_M1_ACCEPTED_DATASET")
    if latest_signal is not None and latest_signal.get("state") != "ACTIVE":
        notices.append(f"SIGNAL_STATE_{latest_signal.get('state')}")
    notices.extend(extra_notices)

    rankings = []
    for item in research_report["recommendations"]:
        rankings.append(
            {
                "symbol": item["symbol"],
                "rank": item["rank"],
                "score": item["score"],
                "recommendation": item["recommendation"],
                "confidence": item["confidence"],
                "price_band": item["price_band"],
                "factors": [dict(factor) for factor in research_report["feature_weights"]],
                "risk_notes": list(item["risk_notes"]),
            }
        )

    recent_actions = [
        {
            "trade_date": trade["trade_date"],
            "symbol": trade["symbol"],
            "side": trade["side"],
            "shares": int(trade["shares"]),
            "price": float(trade["price"]),
            "gross_amount": float(trade["gross_amount"]),
            "total_cost": float(trade["total_cost"]),
            "reason": str(trade["reason"]),
        }
        for trade in research_report["trades"]
        if int(trade["shares"]) > 0
    ][-20:]

    splits = {
        "in_sample": f"{files[0]['min_trade_date']}/{research_report['train_end']}",
        "validation": research_report["splits"]["validation"],
        "out_of_sample": research_report["splits"]["out_of_sample"],
    }

    sequence = int(latest_signal["sequence"]) if latest_signal is not None else 0
    document = {
        "contract_id": "web-state",
        "schema_version": "1.0.0",
        "state_id": f"pilot-live-{as_of}-seq{sequence}",
        "as_of": as_of,
        "generated_at": generated_at,
        "data_source": {
            "source_id": str(dataset_manifest["source"]),
            "is_official_vendor": is_official_vendor,
            "mode": mode,
            "latest_trade_date": latest_trade_date,
            "notices": notices,
        },
        "market": {
            "calendar_date": as_of,
            "is_open": False,
            "session_label": "UNKNOWN",
        },
        "model": {
            "strategy_id": str(champion["strategy_id"]),
            "strategy_version": str(champion["strategy_version"]),
            "adapter_sha256": str(champion["adapter_sha256"]),
            "training_cutoff": research_report["training_cutoff"],
            "validation_cutoff": research_report["validation_cutoff"],
            "backtest_window": {
                "start": research_report["test_start"],
                "end": research_report["test_end"],
            },
            "feature_names": list(FEATURE_NAMES),
            "validation_ic_mean": research_report["validation_ic_mean"],
            "signal_sequence": sequence,
            "signal_state": latest_signal["state"] if latest_signal else "UNAVAILABLE",
            "dataset_id": research_report["dataset_id"],
        },
        "rankings": rankings,
        "portfolio": {
            "initial_capital": float(portfolio["initial_capital"]),
            "cash": round(float(portfolio["cash"]), 2),
            "market_value": round(float(portfolio["market_value"]), 2),
            "total_assets": round(float(portfolio["total_assets"]), 2),
            "positions": [
                {
                    "symbol": position["symbol"],
                    "shares": int(position["shares"]),
                    "locked_shares": int(position["locked_shares"]),
                    "avg_cost": round(float(position["avg_cost"]), 4),
                    "last_price": float(position["last_price"]),
                    "market_value": round(float(position["market_value"]), 2),
                    "weight": float(position["weight"]),
                }
                for position in portfolio["positions"]
            ],
            "recent_actions": recent_actions,
        },
        "performance": {
            "nav_curve": [dict(point) for point in research_report["nav_curve"]],
            "metrics": {
                "total_return": float(research_report["metrics"]["total_return"]),
                "benchmark_total_return": float(
                    research_report["metrics"]["benchmark_total_return"]
                ),
                "max_drawdown": float(research_report["metrics"]["max_drawdown"]),
                "win_rate": float(research_report["metrics"]["win_rate"]),
                "turnover": float(research_report["metrics"]["turnover"]),
            },
            "splits": splits,
            "leak_checks": [dict(check) for check in research_report["leak_checks"]],
        },
    }
    schema = load_web_state_schema()
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble the web-state document")
    parser.add_argument("--research-report", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--head-path", required=True)
    parser.add_argument("--mode", required=True, choices=["live-fetch", "m1-replay", "synthetic"])
    parser.add_argument("--is-official-vendor", action="store_true")
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--notice", action="append", default=[])
    args = parser.parse_args(argv)

    document = build_web_state(
        research_report=_load_json_object(Path(args.research_report)),
        dataset_manifest=_load_json_object(Path(args.dataset_manifest)),
        champion=_load_json_object(Path(args.champion)),
        latest_signal=_read_latest_signal(Path(args.runs_root), Path(args.head_path)),
        mode=args.mode,
        is_official_vendor=args.is_official_vendor,
        generated_at=args.generated_at,
        extra_notices=tuple(args.notice),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.name}.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=True, indent=2) + "\n")
    os.replace(temporary, out_path)
    json.dump({"out": str(out_path), "state_id": document["state_id"]}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
