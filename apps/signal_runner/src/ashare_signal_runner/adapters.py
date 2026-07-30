"""Load only content-addressed strategy adapters approved by a Champion."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from ashare_quant_core import Strategy
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from .runner import canonical_json_sha256

ADAPTER_ID_PATTERN = re.compile(r"^[a-z0-9-]+/v[0-9]+$")


class AdapterVerificationError(ValueError):
    """Raised before untrusted adapter code is executed."""


@dataclass(frozen=True)
class VerifiedAdapter:
    strategy: Strategy
    adapter_id: str
    package_sha256: str
    manifest_sha256: str
    code_sha256: str
    config_sha256: str


def _safe_package_file(*, package_root: Path, filename: str) -> Path:
    candidate = package_root / filename
    if candidate.is_symlink():
        raise AdapterVerificationError(f"adapter file cannot be a symlink: {filename}")
    resolved_root = package_root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise AdapterVerificationError(f"adapter file escapes package root: {filename}")
    return resolved


def _package_sha256(manifest: Mapping[str, Any]) -> str:
    identity = {
        "adapter_id": manifest["adapter_id"],
        "strategy_id": manifest["strategy_id"],
        "strategy_version": manifest["strategy_version"],
        "entrypoint": manifest["entrypoint"],
        "code_sha256": manifest["code_sha256"],
        "config_sha256": manifest["config_sha256"],
    }
    return canonical_json_sha256(identity)


def load_verified_adapter(
    *,
    champion: Mapping[str, Any],
    adapter_root: Path,
    schema: Mapping[str, Any],
) -> VerifiedAdapter:
    """Verify package identity and execute only the verified adapter bytes."""
    adapter_id = str(champion["adapter_id"])
    if not ADAPTER_ID_PATTERN.fullmatch(adapter_id):
        raise AdapterVerificationError("Champion adapter_id is invalid")
    package_root = adapter_root.joinpath(*adapter_id.split("/"))
    try:
        manifest_path = _safe_package_file(
            package_root=package_root,
            filename="adapter.json",
        )
        manifest_bytes = manifest_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise AdapterVerificationError(f"adapter manifest is unavailable: {adapter_id}") from exc
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterVerificationError("adapter manifest is not valid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise AdapterVerificationError("adapter manifest must be an object")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).validate(manifest)
    except (ValidationError, ValueError) as exc:
        raise AdapterVerificationError("adapter manifest does not satisfy its contract") from exc

    fixed_contract_set = champion["fixed_contract_set"]
    expected_bindings = {
        "adapter_id": champion["adapter_id"],
        "strategy_id": champion["strategy_id"],
        "strategy_version": champion["strategy_version"],
        "package_sha256": champion["adapter_sha256"],
        "code_sha256": fixed_contract_set["code_sha256"],
        "config_sha256": fixed_contract_set["config_sha256"],
    }
    for field_name, expected in expected_bindings.items():
        if manifest[field_name] != expected:
            raise AdapterVerificationError(
                f"adapter {field_name} does not match Champion"
            )
    if _package_sha256(manifest) != manifest["package_sha256"]:
        raise AdapterVerificationError("adapter package_sha256 is invalid")

    try:
        code_path = _safe_package_file(
            package_root=package_root,
            filename=str(manifest["code_file"]),
        )
        config_path = _safe_package_file(
            package_root=package_root,
            filename=str(manifest["config_file"]),
        )
        code_bytes = code_path.read_bytes()
        config_bytes = config_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise AdapterVerificationError("adapter package files are unavailable") from exc
    if hashlib.sha256(code_bytes).hexdigest() != manifest["code_sha256"]:
        raise AdapterVerificationError("adapter code hash mismatch")
    if hashlib.sha256(config_bytes).hexdigest() != manifest["config_sha256"]:
        raise AdapterVerificationError("adapter config hash mismatch")
    try:
        config = json.loads(config_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterVerificationError("adapter config is not valid JSON") from exc
    if not isinstance(config, Mapping):
        raise AdapterVerificationError("adapter config must be an object")

    module = ModuleType(f"_ashare_adapter_{manifest['package_sha256']}")
    module.__file__ = str(code_path)
    compiled = compile(code_bytes, str(code_path), "exec")
    exec(compiled, module.__dict__)
    factory = module.__dict__.get("build_strategy")
    if not callable(factory):
        raise RuntimeError("verified adapter does not export build_strategy")
    strategy = factory(config)
    if getattr(strategy, "strategy_id", None) != manifest["strategy_id"]:
        raise RuntimeError("verified adapter returned the wrong strategy_id")
    if getattr(strategy, "strategy_version", None) != manifest["strategy_version"]:
        raise RuntimeError("verified adapter returned the wrong strategy_version")
    if not callable(getattr(strategy, "target_positions", None)):
        raise RuntimeError("verified adapter did not return a Strategy")

    return VerifiedAdapter(
        strategy=strategy,
        adapter_id=adapter_id,
        package_sha256=str(manifest["package_sha256"]),
        manifest_sha256=canonical_json_sha256(manifest),
        code_sha256=str(manifest["code_sha256"]),
        config_sha256=str(manifest["config_sha256"]),
    )
