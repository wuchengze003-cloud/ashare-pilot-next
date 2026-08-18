from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from ashare_web.release import build_release, canonical_json_bytes
from ashare_web.server import make_handler, validate_release_dir

ROOT = Path(__file__).resolve().parents[3]


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    signal = json.loads(
        (ROOT / "contracts/examples/production-signal.active.example.json").read_text()
    )
    signal_bytes = canonical_json_bytes(signal)
    signal_path = tmp_path / "signal.json"
    signal_path.write_bytes(signal_bytes)
    account = json.loads(
        (ROOT / "contracts/examples/simulated-account-state-v2.example.json").read_text()
    )
    account["source_signal"] = {
        "signal_id": signal["signal_id"],
        "sequence": signal["sequence"],
        "state": signal["state"],
        "as_of": signal["as_of"],
        "sha256": hashlib.sha256(signal_bytes).hexdigest(),
    }
    account_path = tmp_path / "account.json"
    account_path.write_bytes(canonical_json_bytes(account))
    return signal_path, account_path


def test_build_release_cross_binds_explicit_contracts(tmp_path: Path) -> None:
    signal_path, account_path = _inputs(tmp_path)
    target = build_release(
        contracts_root=ROOT / "contracts",
        static_root=ROOT / "apps/web/static",
        signal_path=signal_path,
        account_state_path=account_path,
        releases_root=tmp_path / "releases",
        generated_at=datetime(2026, 8, 18, 12, tzinfo=UTC),
    )
    assert target.name.startswith("web-")
    assert {path.name for path in target.iterdir()} == {
        "index.html", "app.css", "app.js", "production-signal.json",
        "simulated-account-state.json", "release-manifest.json",
    }
    manifest = json.loads((target / "release-manifest.json").read_bytes())
    assert manifest["signal_sha256"] == hashlib.sha256(signal_path.read_bytes()).hexdigest()
    assert manifest["account_state_sha256"] == hashlib.sha256(account_path.read_bytes()).hexdigest()
    assert "innerHTML" not in (target / "app.js").read_text(encoding="utf-8")


def test_build_release_rejects_account_from_another_signal(tmp_path: Path) -> None:
    signal_path, account_path = _inputs(tmp_path)
    account = json.loads(account_path.read_bytes())
    account["source_signal"]["sha256"] = "f" * 64
    account_path.write_bytes(canonical_json_bytes(account))
    with pytest.raises(ValueError, match="not bound"):
        build_release(
            contracts_root=ROOT / "contracts",
            static_root=ROOT / "apps/web/static",
            signal_path=signal_path,
            account_state_path=account_path,
            releases_root=tmp_path / "releases",
            generated_at=datetime(2026, 8, 18, 12, tzinfo=UTC),
        )
    assert not (tmp_path / "releases").exists()


def test_server_rechecks_every_release_byte_before_serving(tmp_path: Path) -> None:
    signal_path, account_path = _inputs(tmp_path)
    target = build_release(
        contracts_root=ROOT / "contracts",
        static_root=ROOT / "apps/web/static",
        signal_path=signal_path,
        account_state_path=account_path,
        releases_root=tmp_path / "releases",
        generated_at=datetime(2026, 8, 18, 12, tzinfo=UTC),
    )

    assert validate_release_dir(target) == target.resolve()
    assert make_handler(target)
    (target / "app.js").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact (size|hash) mismatch"):
        make_handler(target)
