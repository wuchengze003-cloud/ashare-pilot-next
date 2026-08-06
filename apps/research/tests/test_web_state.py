"""Producer tests for the web-state document assembly."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from ashare_research_app.backtest import PilotConfig, run_walk_forward
from ashare_research_app.features import FEATURE_NAMES
from ashare_research_app.pilot_research import build_research_report
from ashare_research_app.promotion import promote_baseline_model
from ashare_research_app.web_state import build_web_state, load_web_state_schema
from jsonschema import Draft202012Validator, FormatChecker
from test_ml_champion_chain import ROOT, contract_documents, synthetic_snapshot, write_dataset_files

from tools.validate_contracts import validate_semantics


def assembled_document(tmp_path: Path) -> dict:
    snapshot = synthetic_snapshot()
    manifest = write_dataset_files(snapshot, tmp_path / "dataset")
    documents = contract_documents()
    model, production_model, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )
    paths = promote_baseline_model(
        repository_root=ROOT,
        runtime_root=tmp_path / "runtime",
        model_bundle_bytes=production_model.bundle_bytes(),
        report=report,
        dataset_manifest=manifest,
        snapshot_symbols=tuple(sorted({bar.symbol for bar in snapshot.records})),
        as_of=snapshot.as_of.isoformat(),
        generated_at=datetime(2026, 8, 4, 1, 0, tzinfo=UTC),
        top_k=4,
        per_weight=0.24,
        feature_names=FEATURE_NAMES,
    )
    import json

    champion = json.loads(paths.champion_path.read_text(encoding="utf-8"))
    research_report = build_research_report(report, initial_capital=PilotConfig().initial_capital)
    research_report["as_of"] = snapshot.as_of.isoformat()
    research_report["splits"] = {
        "validation": (
            f"{report.validation_start.isoformat()}/{report.validation_end.isoformat()}"
        ),
        "out_of_sample": f"{report.test_start.isoformat()}/{report.test_end.isoformat()}",
    }
    return build_web_state(
        research_report=research_report,
        dataset_manifest=manifest,
        champion=champion,
        latest_signal=None,
        mode="synthetic",
        is_official_vendor=False,
        generated_at="2026-08-04T02:00:00Z",
        extra_notices=("SYNTHETIC_PRODUCER_TEST",),
    )


def test_web_state_document_passes_schema_and_semantics(tmp_path: Path) -> None:
    document = assembled_document(tmp_path)

    schema = load_web_state_schema()
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
    validate_semantics("web-state", document, Path("producer-test.json"))

    assert document["data_source"]["mode"] == "synthetic"
    assert "NON_OFFICIAL_VENDOR_ENDPOINT" in document["data_source"]["notices"]
    assert document["portfolio"]["initial_capital"] == 500000.0
    assert document["portfolio"]["total_assets"] > 0
    assert document["rankings"]
    assert all("rank" in item for item in document["rankings"])
    assert document["model"]["signal_state"] == "UNAVAILABLE"
    assert document["performance"]["leak_checks"]
    assert all(
        check["status"] == "pass" for check in document["performance"]["leak_checks"]
    )


def test_web_state_rejects_unknown_mode(tmp_path: Path) -> None:
    snapshot = synthetic_snapshot()
    manifest = write_dataset_files(snapshot, tmp_path / "dataset")
    documents = contract_documents()
    model, production_model, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )
    paths = promote_baseline_model(
        repository_root=ROOT,
        runtime_root=tmp_path / "runtime",
        model_bundle_bytes=production_model.bundle_bytes(),
        report=report,
        dataset_manifest=manifest,
        snapshot_symbols=tuple(sorted({bar.symbol for bar in snapshot.records})),
        as_of=snapshot.as_of.isoformat(),
        generated_at=datetime(2026, 8, 4, 1, 0, tzinfo=UTC),
        top_k=4,
        per_weight=0.24,
        feature_names=FEATURE_NAMES,
    )
    import json

    champion = json.loads(paths.champion_path.read_text(encoding="utf-8"))
    research_report = build_research_report(report, initial_capital=PilotConfig().initial_capital)
    research_report["as_of"] = snapshot.as_of.isoformat()
    research_report["splits"] = {
        "validation": (
            f"{report.validation_start.isoformat()}/{report.validation_end.isoformat()}"
        ),
        "out_of_sample": f"{report.test_start.isoformat()}/{report.test_end.isoformat()}",
    }

    with pytest.raises(ValueError, match="unsupported web-state mode"):
        build_web_state(
            research_report=research_report,
            dataset_manifest=manifest,
            champion=champion,
            latest_signal=None,
            mode="not-a-mode",
            is_official_vendor=False,
            generated_at="2026-08-04T02:00:00Z",
        )
