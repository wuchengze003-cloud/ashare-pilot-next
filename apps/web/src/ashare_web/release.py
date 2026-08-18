"""Build an immutable static release from explicit production contracts.

The builder never scans runtime directories and never calculates financial
values.  It validates and cross-binds one Production Signal and the simulated
account state produced from that exact signal, then copies their canonical
bytes beside fixed presentation assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ashare_signal_runner import load_current_run
from jsonschema import Draft202012Validator, FormatChecker

STATIC_FILES = ("index.html", "app.css", "app.js")
DATA_FILES = ("production-signal.json", "simulated-account-state.json")


def canonical_json_bytes(document: dict[str, Any]) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"contract must be a regular file: {path}")
    content = path.read_bytes()
    try:
        document = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"contract is not valid JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"contract must be a JSON object: {path}")
    if content != canonical_json_bytes(document):
        raise ValueError(f"contract bytes are not canonical: {path}")
    return content, document


def _schemas(contracts_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    registry = json.loads((contracts_root / "registry.json").read_text(encoding="utf-8"))
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in registry["contracts"]:
        key = (str(entry["contract_id"]), str(entry["schema_version"]))
        result[key] = json.loads(
            (contracts_root / str(entry["schema"])).read_text(encoding="utf-8")
        )
    return result


def _validate(document: dict[str, Any], schemas: dict[tuple[str, str], dict]) -> None:
    key = (str(document.get("contract_id")), str(document.get("schema_version")))
    schema = schemas.get(key)
    if schema is None:
        raise ValueError(f"unregistered contract: {key[0]} {key[1]}")
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        raise ValueError(f"{key[0]} contract validation failed: {errors[0].message}")


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def build_release(
    *,
    contracts_root: Path,
    static_root: Path,
    signal_path: Path,
    account_state_path: Path,
    releases_root: Path,
    generated_at: datetime,
) -> Path:
    """Publish one content-addressed static release and return its directory."""
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    schemas = _schemas(Path(contracts_root))
    signal_bytes, signal = _read_object(Path(signal_path))
    account_bytes, account = _read_object(Path(account_state_path))
    _validate(signal, schemas)
    _validate(account, schemas)
    if signal["contract_id"] != "production-signal":
        raise ValueError("signal input is not a Production Signal")
    if account["contract_id"] != "simulated-account-state":
        raise ValueError("account input is not a simulated account state")

    signal_sha256 = hashlib.sha256(signal_bytes).hexdigest()
    source_signal = account["source_signal"]
    expected = {
        "signal_id": signal["signal_id"],
        "sequence": signal["sequence"],
        "state": signal["state"],
        "as_of": signal["as_of"],
        "sha256": signal_sha256,
    }
    if source_signal != expected:
        raise ValueError("simulated account is not bound to the supplied Production Signal")

    static_root = Path(static_root)
    if static_root.is_symlink() or not static_root.is_dir():
        raise ValueError("static asset root must be a regular directory")
    static_bytes: dict[str, bytes] = {}
    for name in STATIC_FILES:
        path = static_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"static asset is unavailable: {name}")
        static_bytes[name] = path.read_bytes()

    identity = {
        "signal_sha256": signal_sha256,
        "account_state_sha256": hashlib.sha256(account_bytes).hexdigest(),
        "assets": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in sorted(static_bytes.items())
        },
    }
    release_id = "web-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:20]
    generated_text = generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    manifest: dict[str, Any] = {
        "contract_id": "static-web-release",
        "schema_version": "1.0.0",
        "release_id": release_id,
        "generated_at": generated_text,
        "as_of": signal["as_of"],
        "signal_id": signal["signal_id"],
        "signal_sha256": signal_sha256,
        "account_state_id": account["state_id"],
        "account_state_sha256": identity["account_state_sha256"],
        "files": [],
    }
    file_contents = {
        **static_bytes,
        "production-signal.json": signal_bytes,
        "simulated-account-state.json": account_bytes,
    }
    manifest["files"] = [
        {
            "path": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "file_size_bytes": len(content),
        }
        for name, content in sorted(file_contents.items())
    ]
    _validate(manifest, schemas)
    manifest_bytes = canonical_json_bytes(manifest)

    releases_root = Path(releases_root)
    if releases_root.is_symlink():
        raise ValueError("release root cannot be a symlink")
    releases_root.mkdir(parents=True, exist_ok=True)
    target = releases_root / release_id
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"static release already exists: {release_id}")
    staging = Path(tempfile.mkdtemp(prefix=f".{release_id}.tmp-", dir=releases_root))
    try:
        for name, content in file_contents.items():
            _write_new(staging / name, content)
        _write_new(staging / "release-manifest.json", manifest_bytes)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an immutable read-only Web release")
    parser.add_argument("--contracts-root", required=True, type=Path)
    parser.add_argument("--static-root", required=True, type=Path)
    parser.add_argument("--signal-runs-root", required=True, type=Path)
    parser.add_argument("--signal-head", required=True, type=Path)
    parser.add_argument("--signal-as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--account-state", required=True, type=Path)
    parser.add_argument("--releases-root", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args(argv)
    generated_at = datetime.fromisoformat(args.generated_at.replace("Z", "+00:00"))
    versioned = _schemas(args.contracts_root)
    schemas = {contract_id: schema for (contract_id, _version), schema in versioned.items()}
    artifacts = load_current_run(
        runs_root=args.signal_runs_root,
        head_path=args.signal_head,
        required_as_of=args.signal_as_of,
        schemas=schemas,
    )
    if artifacts is None:
        raise ValueError("no committed Production Signal is available")
    with tempfile.TemporaryDirectory(prefix="ashare-web-signal-") as temporary_dir:
        signal_path = Path(temporary_dir) / "production-signal.json"
        signal_path.write_bytes(artifacts.production_signal_bytes)
        target = build_release(
            contracts_root=args.contracts_root,
            static_root=args.static_root,
            signal_path=signal_path,
            account_state_path=args.account_state,
            releases_root=args.releases_root,
            generated_at=generated_at,
        )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
