"""Champion lifecycle tests: frozen champion, activation gating, fail-closed.

These exercise the Research/Promotion vs Live Inference separation at the
CLI boundary (subprocess), mirroring how ops/live_demo orchestrates it.
"""

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ashare_research_app import promotion
from ashare_research_app.backtest import PilotConfig, run_walk_forward
from ashare_research_app.features import FEATURE_NAMES
from test_ml_champion_chain import ROOT, contract_documents, synthetic_snapshot, write_dataset_files


def promote_and_activate(tmp_path, *, generated_at, activate=True):
    snapshot = synthetic_snapshot()
    dataset_dir = tmp_path / "dataset"
    manifest = write_dataset_files(snapshot, dataset_dir)
    documents = contract_documents()
    _model, production_model, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )
    runtime_root = tmp_path / "runtime"
    paths = promotion.promote_baseline_model(
        repository_root=ROOT,
        runtime_root=runtime_root,
        model_bundle_bytes=production_model.bundle_bytes(),
        report=report,
        dataset_manifest=manifest,
        snapshot_symbols=tuple(sorted({bar.symbol for bar in snapshot.records})),
        as_of=snapshot.as_of.isoformat(),
        generated_at=generated_at,
        top_k=4,
        per_weight=0.24,
        feature_names=FEATURE_NAMES,
    )
    if activate:
        promotion.activate_champion(
            runtime_root=runtime_root,
            champion_id=paths.champion_id,
            activated_at=generated_at,
        )
    return snapshot, dataset_dir, manifest, paths, runtime_root


def run_pilot_runtime_root(*, runtime_root, dataset_root, as_of, generated_at, label):
    git_sha = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ashare_signal_runner.pilot_run",
            "--runtime-root",
            str(runtime_root),
            "--dataset-root",
            str(dataset_root),
            "--runs-root",
            str(runtime_root / "runs"),
            "--head-path",
            str(runtime_root / "current-signal-head.json"),
            "--as-of",
            as_of.isoformat(),
            "--generated-at",
            generated_at.isoformat().replace("+00:00", "Z"),
            "--signal-id",
            f"pilot-signal-{label}",
            "--run-id",
            f"pilot-run-{label}",
            "--git-sha",
            git_sha,
        ],
        capture_output=True,
        text=True,
    )


def test_no_active_champion_fails_closed(tmp_path: Path) -> None:
    snapshot = synthetic_snapshot()
    dataset_dir = tmp_path / "dataset"
    write_dataset_files(snapshot, dataset_dir)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True)

    result = run_pilot_runtime_root(
        runtime_root=runtime_root,
        dataset_root=dataset_dir,
        as_of=snapshot.as_of,
        generated_at=datetime(2026, 8, 4, 2, 0, tzinfo=UTC),
        label="nochamp",
    )
    assert result.returncode == 4
    assert json.loads(result.stdout)["error"] == "NO_ACTIVE_CHAMPION"


def test_frozen_champion_stable_across_consecutive_inference(tmp_path: Path) -> None:
    promoted_at = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)
    snapshot, dataset_dir, _manifest, paths, runtime_root = promote_and_activate(
        tmp_path, generated_at=promoted_at
    )

    first = run_pilot_runtime_root(
        runtime_root=runtime_root,
        dataset_root=dataset_dir,
        as_of=snapshot.as_of,
        generated_at=promoted_at + timedelta(hours=1),
        label="one",
    )
    second = run_pilot_runtime_root(
        runtime_root=runtime_root,
        dataset_root=dataset_dir,
        as_of=snapshot.as_of,
        generated_at=promoted_at + timedelta(hours=2),
        label="two",
    )
    assert first.returncode == 0 and second.returncode == 0
    a = json.loads(first.stdout)
    b = json.loads(second.stdout)
    assert a["champion_id"] == b["champion_id"] == paths.champion_id
    assert a["champion_sha256"] == b["champion_sha256"] == paths.champion_sha256
    # Only the signal sequence may change; champion identity is frozen.
    assert a["sequence"] == 1 and b["sequence"] == 2


def test_live_path_source_has_no_training_or_promotion() -> None:
    frozen = (ROOT / "apps/research/src/ashare_research_app/frozen_inference.py").read_text()
    pilot_run = (
        ROOT / "apps/signal_runner/src/ashare_signal_runner/pilot_run.py"
    ).read_text()
    for source in (frozen, pilot_run):
        assert "run_walk_forward" not in source
        assert ".fit(" not in source
        assert "promote_baseline_model" not in source
        assert "activate_champion" not in source
    live_demo = (ROOT / "ops/live_demo.py").read_text()
    # The refresh cycle must not invoke research promotion; only bootstrap may.
    cycle_body = live_demo[live_demo.index("def run_cycle"):]
    assert "pilot_research" not in cycle_body


def test_activation_gating_live_uses_previous_champion_until_activated(tmp_path: Path) -> None:
    promoted_a = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)
    snapshot, dataset_dir, _manifest, paths_a, runtime_root = promote_and_activate(
        tmp_path, generated_at=promoted_a
    )

    # Promote a second champion (different promoted_at => different id) but do NOT activate.
    promoted_b = promoted_a + timedelta(days=1)
    documents = contract_documents()
    _m, production_b, report_b = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )
    paths_b = promotion.promote_baseline_model(
        repository_root=ROOT,
        runtime_root=runtime_root,
        model_bundle_bytes=production_b.bundle_bytes(),
        report=report_b,
        dataset_manifest=_manifest,
        snapshot_symbols=tuple(sorted({bar.symbol for bar in snapshot.records})),
        as_of=snapshot.as_of.isoformat(),
        generated_at=promoted_b,
        top_k=4,
        per_weight=0.24,
        feature_names=FEATURE_NAMES,
    )
    assert paths_b.champion_id != paths_a.champion_id

    # Before activation, live inference must still use champion A.
    before = run_pilot_runtime_root(
        runtime_root=runtime_root,
        dataset_root=dataset_dir,
        as_of=snapshot.as_of,
        generated_at=promoted_b + timedelta(hours=1),
        label="before",
    )
    assert json.loads(before.stdout)["champion_id"] == paths_a.champion_id

    promotion.activate_champion(
        runtime_root=runtime_root,
        champion_id=paths_b.champion_id,
        activated_at=promoted_b + timedelta(hours=2),
    )
    after = run_pilot_runtime_root(
        runtime_root=runtime_root,
        dataset_root=dataset_dir,
        as_of=snapshot.as_of,
        generated_at=promoted_b + timedelta(hours=3),
        label="after",
    )
    assert json.loads(after.stdout)["champion_id"] == paths_b.champion_id


def test_promotion_conflict_does_not_clobber_existing_package(tmp_path: Path) -> None:
    promoted_at = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)
    snapshot, _dataset_dir, manifest, paths, runtime_root = promote_and_activate(
        tmp_path, generated_at=promoted_at
    )
    documents = contract_documents()
    _m, production, report = run_walk_forward(
        snapshot,
        cost_model_doc=documents["cost-model"],
        market_rules_doc=documents["market-rules"],
        execution_policy_doc=documents["execution-policy"],
        portfolio_risk_doc=documents["portfolio-risk"],
        config=PilotConfig(top_k=4, per_weight=0.24),
    )
    # Forge a same-id package with different content by writing a conflicting file.
    conflicting = paths.package_dir / "model" / "model.bundle"
    original = conflicting.read_bytes()

    import pytest

    # Re-promote with identical inputs is idempotent (no error).
    again = promotion.promote_baseline_model(
        repository_root=ROOT,
        runtime_root=runtime_root,
        model_bundle_bytes=production.bundle_bytes(),
        report=report,
        dataset_manifest=manifest,
        snapshot_symbols=tuple(sorted({bar.symbol for bar in snapshot.records})),
        as_of=snapshot.as_of.isoformat(),
        generated_at=promoted_at,
        top_k=4,
        per_weight=0.24,
        feature_names=FEATURE_NAMES,
    )
    assert again.champion_id == paths.champion_id

    # Corrupt the package, then a re-promote with same id but different content fails.
    conflicting.write_bytes(original + b"tamper")
    with pytest.raises(promotion.ChampionPackageError):
        promotion.promote_baseline_model(
            repository_root=ROOT,
            runtime_root=runtime_root,
            model_bundle_bytes=production.bundle_bytes(),
            report=report,
            dataset_manifest=manifest,
            snapshot_symbols=tuple(sorted({bar.symbol for bar in snapshot.records})),
            as_of=snapshot.as_of.isoformat(),
            generated_at=promoted_at,
            top_k=4,
            per_weight=0.24,
            feature_names=FEATURE_NAMES,
        )
    # Active pointer is untouched by the failed creation.
    pointer = json.loads((runtime_root / "active-champion.json").read_text())
    assert pointer["champion_id"] == paths.champion_id


def test_verify_signal_binding_detects_stale_dataset_champion_and_as_of() -> None:
    from ashare_research_app.web_state import verify_signal_binding

    signal = {
        "as_of": "2023-10-31",
        "contract_set": {
            "dataset_snapshot_sha256": "d" * 64,
            "champion_sha256": "c" * 64,
        },
    }
    # Matching: no reasons.
    assert verify_signal_binding(
        latest_signal=signal,
        as_of="2023-10-31",
        current_snapshot_sha256="d" * 64,
        active_champion_sha256="c" * 64,
    ) == []
    # Stale as_of.
    assert "AS_OF_MISMATCH" in verify_signal_binding(
        latest_signal=signal,
        as_of="2023-11-01",
        current_snapshot_sha256="d" * 64,
        active_champion_sha256="c" * 64,
    )
    # Stale dataset snapshot.
    assert "SNAPSHOT_MISMATCH" in verify_signal_binding(
        latest_signal=signal,
        as_of="2023-10-31",
        current_snapshot_sha256="e" * 64,
        active_champion_sha256="c" * 64,
    )
    # Stale champion.
    assert "CHAMPION_MISMATCH" in verify_signal_binding(
        latest_signal=signal,
        as_of="2023-10-31",
        current_snapshot_sha256="d" * 64,
        active_champion_sha256="f" * 64,
    )
    # Missing signal.
    assert verify_signal_binding(
        latest_signal=None,
        as_of="2023-10-31",
        current_snapshot_sha256="d" * 64,
        active_champion_sha256="c" * 64,
    ) == ["NO_SIGNAL"]
