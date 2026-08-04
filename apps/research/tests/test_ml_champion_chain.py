"""End-to-end chain test: research promotion -> public signal command.

Research cannot import Signal Runner, so this test exercises the public
`ashare_signal_runner.pilot_run` command through a subprocess, exactly
like the ops orchestrator does.
"""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from ashare_quant_core import (
    DAILY_BAR_SCHEMA_ID,
    DAILY_BAR_SCHEMA_SHA256,
    DailyBar,
    DatasetSnapshot,
)
from ashare_research_app import promotion
from ashare_research_app.backtest import PilotConfig, run_walk_forward
from ashare_research_app.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[3]


def trading_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def synthetic_snapshot(*, days: int = 80) -> DatasetSnapshot:
    import random

    calendar = trading_days(date(2023, 6, 1), days)
    symbols = (
        "000001.SZ",
        "000333.SZ",
        "002594.SZ",
        "300750.SZ",
        "600036.SH",
        "600519.SH",
        "601318.SH",
        "688981.SH",
    )
    records: list[DailyBar] = []
    for symbol in symbols:
        generator = random.Random(f"11:{symbol}")
        price = 12.0 + generator.random() * 30
        for day in calendar:
            drift = generator.uniform(-0.03, 0.032)
            open_price = price
            close_price = max(1.0, price * (1 + drift))
            high = max(open_price, close_price) * (1 + generator.uniform(0, 0.01))
            low = min(open_price, close_price) * (1 - generator.uniform(0, 0.01))
            volume = generator.uniform(5e6, 5e7)
            records.append(
                DailyBar(
                    symbol=symbol,
                    trade_date=day,
                    open=round(open_price, 4),
                    high=round(high, 4),
                    low=round(low, 4),
                    close=round(close_price, 4),
                    volume=round(volume, 0),
                    amount=round(volume * (open_price + close_price) / 2, 0),
                )
            )
            price = close_price
    return DatasetSnapshot(
        dataset_id="synthetic-chain/v1",
        dataset_family_id="synthetic-chain/v1",
        manifest_sha256="b" * 64,
        as_of=calendar[-1],
        data_schema_id=DAILY_BAR_SCHEMA_ID,
        data_schema_sha256=DAILY_BAR_SCHEMA_SHA256,
        normalization_version="synthetic-normalizer/v1",
        records=tuple(records),
    )


def write_dataset_files(snapshot: DatasetSnapshot, dataset_dir: Path) -> dict:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "symbol": bar.symbol,
            "trade_date": bar.trade_date.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "amount": bar.amount,
        }
        for bar in snapshot.records
    ]
    content = (json.dumps(rows, ensure_ascii=True, indent=2) + "\n").encode()
    (dataset_dir / "daily-bars.json").write_bytes(content)
    manifest = {
        "contract_id": "dataset-manifest",
        "schema_version": "2.0.0",
        "dataset_id": f"synthetic-chain-{snapshot.as_of.isoformat()}",
        "dataset_family_id": "synthetic-chain/v1",
        "dataset_kind": "normalized",
        "data_schema_id": DAILY_BAR_SCHEMA_ID,
        "data_schema_sha256": DAILY_BAR_SCHEMA_SHA256,
        "normalization_version": "synthetic-normalizer/v1",
        "as_of": snapshot.as_of.isoformat(),
        "generated_at": f"{snapshot.as_of.isoformat()}T00:30:00Z",
        "source": "synthetic-chain",
        "source_version": "v1",
        "parent_manifest_sha256": None,
        "quality_status": "pass",
        "quality_reasons": [],
        "files": [
            {
                "path": "daily-bars.json",
                "sha256": hashlib.sha256(content).hexdigest(),
                "row_count": len(rows),
                "file_size_bytes": len(content),
                "min_trade_date": min(row["trade_date"] for row in rows),
                "max_trade_date": max(row["trade_date"] for row in rows),
            }
        ],
    }
    return manifest


def contract_documents() -> dict[str, dict]:
    cost_model = json.loads(
        (ROOT / "contracts/examples/cost-model.example.json").read_text(encoding="utf-8")
    )
    market_rules = {
        "contract_id": "market-rules",
        "schema_version": "1.0.0",
        "rules_id": "test-a-share-market/v1",
        "segments": [
            {
                "segment_id": f"{board}-normal",
                "board": board,
                "security_status": "normal",
                "valid_from": "2020-01-01",
                "valid_to": None,
                "lot_size": 100,
                "sell_t_plus": 1,
                "price_limit_pct": 0.2 if board in {"gem", "star"} else 0.1,
            }
            for board in ("main", "gem", "star")
        ],
    }
    execution_policy = json.loads(
        (ROOT / "contracts/examples/execution-policy.example.json").read_text(encoding="utf-8")
    )
    portfolio_risk = json.loads(
        (ROOT / "contracts/examples/portfolio-risk.example.json").read_text(encoding="utf-8")
    )
    portfolio_risk["max_positions"] = 6
    portfolio_risk["max_single_weight"] = 0.25
    return {
        "cost-model": cost_model,
        "market-rules": market_rules,
        "execution-policy": execution_policy,
        "portfolio-risk": portfolio_risk,
    }


def run_pilot_command(
    *,
    contracts_dir: Path,
    adapter_root: Path,
    dataset_root: Path,
    runs_root: Path,
    head_path: Path,
    as_of: date,
    generated_at: datetime,
    sequence_label: str,
) -> dict:
    git_sha = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ashare_signal_runner.pilot_run",
            "--contracts-dir",
            str(contracts_dir),
            "--adapter-root",
            str(adapter_root),
            "--dataset-root",
            str(dataset_root),
            "--runs-root",
            str(runs_root),
            "--head-path",
            str(head_path),
            "--as-of",
            as_of.isoformat(),
            "--generated-at",
            generated_at.isoformat().replace("+00:00", "Z"),
            "--signal-id",
            f"pilot-signal-{sequence_label}",
            "--run-id",
            f"pilot-run-{sequence_label}",
            "--git-sha",
            git_sha,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_promotion_feeds_live_signal_chain(tmp_path: Path) -> None:
    snapshot = synthetic_snapshot()
    dataset_dir = tmp_path / "dataset"
    manifest = write_dataset_files(snapshot, dataset_dir)
    documents = contract_documents()

    model, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )
    runtime_root = tmp_path / "runtime"
    promoted_at = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)
    paths = promotion.promote_baseline_model(
        repository_root=ROOT,
        runtime_root=runtime_root,
        model_bundle_bytes=model.bundle_bytes(),
        report=report,
        dataset_manifest=manifest,
        snapshot_symbols=tuple(sorted({bar.symbol for bar in snapshot.records})),
        as_of=snapshot.as_of.isoformat(),
        generated_at=promoted_at,
        top_k=4,
        per_weight=0.24,
        feature_names=FEATURE_NAMES,
    )

    signal = run_pilot_command(
        contracts_dir=paths.contracts_dir,
        adapter_root=paths.adapter_root,
        dataset_root=dataset_dir,
        runs_root=runtime_root / "runs",
        head_path=runtime_root / "current-signal-head.json",
        as_of=snapshot.as_of,
        generated_at=promoted_at + timedelta(hours=1),
        sequence_label="first",
    )

    assert signal["state"] == "ACTIVE"
    assert signal["sequence"] == 1
    assert signal["target_positions"]
    assert len(signal["target_positions"]) <= 4
    assert all(position["target_weight"] == 0.24 for position in signal["target_positions"])
    assert "CHAMPION_ACTIVE" in signal["reason_codes"]

    second = run_pilot_command(
        contracts_dir=paths.contracts_dir,
        adapter_root=paths.adapter_root,
        dataset_root=dataset_dir,
        runs_root=runtime_root / "runs",
        head_path=runtime_root / "current-signal-head.json",
        as_of=snapshot.as_of,
        generated_at=promoted_at + timedelta(hours=2),
        sequence_label="second",
    )
    assert second["state"] == "ACTIVE"
    assert second["sequence"] == 2
    assert second["target_positions"] == signal["target_positions"]


def test_promotion_report_binds_leak_checks(tmp_path: Path) -> None:
    snapshot = synthetic_snapshot()
    dataset_dir = tmp_path / "dataset"
    manifest = write_dataset_files(snapshot, dataset_dir)
    documents = contract_documents()
    model, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )
    paths = promotion.promote_baseline_model(
        repository_root=ROOT,
        runtime_root=tmp_path / "runtime",
        model_bundle_bytes=model.bundle_bytes(),
        report=report,
        dataset_manifest=manifest,
        snapshot_symbols=tuple(sorted({bar.symbol for bar in snapshot.records})),
        as_of=snapshot.as_of.isoformat(),
        generated_at=datetime(2026, 8, 4, 1, 0, tzinfo=UTC),
        top_k=4,
        per_weight=0.24,
        feature_names=FEATURE_NAMES,
    )
    champion = json.loads(paths.champion_path.read_text(encoding="utf-8"))
    promotion_report = json.loads(paths.promotion_report_path.read_text(encoding="utf-8"))

    assert champion["promotion_report_sha256"] == promotion.canonical_json_sha256(
        promotion_report
    )
    statuses = {check["check_id"]: check["status"] for check in promotion_report["leak_checks"]}
    assert statuses["temporal_split_monotonic"] == "pass"
    assert statuses["feature_pit_truncation_invariant"] == "pass"
    assert promotion_report["metrics"]["max_drawdown"] >= 0
    assert report.metrics["total_return"] == pytest.approx(
        promotion_report["metrics"]["total_return"], abs=1e-9
    )
    assert paths.model_bundle_path.is_file()
