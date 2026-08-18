"""Canonical JSON helpers shared by Research artifacts."""

from __future__ import annotations

import hashlib
import json


def canonical_json_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()
