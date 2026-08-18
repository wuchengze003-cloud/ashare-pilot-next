from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from ashare_research_app import promotion
from ashare_research_app.activate import main as activation_main
from ashare_research_app.backtest import PilotConfig, run_walk_forward
from ashare_research_app.features import FEATURE_NAMES
from test_ml_champion_chain import (
    ROOT,
    contract_documents,
    permissive_promotion_gate,
    synthetic_snapshot,
    synthetic_universe,
    write_dataset_files,
)


def _candidate(tmp_path: Path):
    snapshot = synthetic_snapshot()
    manifest = write_dataset_files(snapshot, tmp_path / "dataset")
    documents = contract_documents()
    _evaluation, production, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )
    promoted_at = datetime(2026, 8, 4, 1, tzinfo=UTC)
    runtime_root = tmp_path / "runtime"
    paths = promotion.promote_baseline_model(
        repository_root=ROOT,
        runtime_root=runtime_root,
        model_bundle_bytes=production.bundle_bytes(),
        report=report,
        dataset_manifest=manifest,
        snapshot_symbols=tuple(sorted({bar.symbol for bar in snapshot.records})),
        universe_document=synthetic_universe(snapshot, manifest, promoted_at),
        as_of=snapshot.as_of.isoformat(),
        generated_at=promoted_at,
        top_k=4,
        per_weight=0.24,
        feature_names=FEATURE_NAMES,
        promotion_gate=permissive_promotion_gate(),
    )
    return paths, runtime_root, promoted_at


def test_candidate_is_not_active_until_explicit_activation(tmp_path: Path) -> None:
    paths, runtime_root, promoted_at = _candidate(tmp_path)
    assert promotion.load_active_champion(runtime_root) is None

    pointer = promotion.activate_champion(
        runtime_root=runtime_root,
        champion_id=paths.champion_id,
        activated_at=promoted_at,
        approval_id="test-human-approval",
    )

    assert pointer["champion_id"] == paths.champion_id
    assert promotion.load_active_champion(runtime_root) == pointer


def test_activation_rejects_path_escape_and_tampered_gate(tmp_path: Path) -> None:
    paths, runtime_root, promoted_at = _candidate(tmp_path)
    with pytest.raises(promotion.ChampionPackageError, match="champion id is invalid"):
        promotion.activate_champion(
            runtime_root=runtime_root,
            champion_id="../../outside",
            activated_at=promoted_at,
            approval_id="test-human-approval",
        )

    report_path = paths.package_dir / "promotion-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["promotion_gate_evaluation"]["status"] = "fail"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(promotion.ChampionPackageError, match="promotion report hash"):
        promotion.activate_champion(
            runtime_root=runtime_root,
            champion_id=paths.champion_id,
            activated_at=promoted_at,
            approval_id="test-human-approval",
        )


def test_activation_does_not_flip_pointer_when_receipt_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, runtime_root, promoted_at = _candidate(tmp_path)

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated receipt storage failure")

    monkeypatch.setattr(promotion.os, "replace", fail_replace)
    with pytest.raises(OSError, match="receipt storage"):
        promotion.activate_champion(
            runtime_root=runtime_root,
            champion_id=paths.champion_id,
            activated_at=promoted_at,
            approval_id="test-human-approval",
        )
    assert not (runtime_root / "active-champion.json").exists()


def test_activation_cli_binds_human_approval_into_pointer_and_receipt(
    tmp_path: Path,
) -> None:
    paths, runtime_root, promoted_at = _candidate(tmp_path)
    result = activation_main(
        [
            "--runtime-root",
            str(runtime_root),
            "--champion-id",
            paths.champion_id,
            "--expected-champion-sha256",
            paths.champion_sha256,
            "--approval-id",
            "review-20260818",
            "--activated-at",
            promoted_at.isoformat(),
        ]
    )

    assert result == 0
    pointer = promotion.load_active_champion(runtime_root)
    assert pointer is not None
    assert pointer["approval_id"] == "review-20260818"
    receipt_path = (
        runtime_root
        / "activation-receipts"
        / f"{pointer['activation_id']}.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["activation_id"] == pointer["activation_id"]
    assert receipt["pointer_sha256"] == hashlib.sha256(
        promotion.canonical_json_bytes(pointer) + b"\n"
    ).hexdigest()
