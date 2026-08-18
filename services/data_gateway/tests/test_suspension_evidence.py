from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from ashare_data_gateway.suspension_evidence import (
    load_authoritative_suspension_evidence,
)


def _write_evidence(tmp_path: Path, *, source_url: str) -> Path:
    source = tmp_path / "notice.pdf"
    source.write_bytes(b"exchange notice bytes")
    evidence = tmp_path / "notice.json"
    evidence.write_text(
        json.dumps(
            {
                "evidence_id": "exchange-suspension-evidence/v1",
                "symbol": "688005.SH",
                "trade_date": "2026-01-16",
                "status": "S",
                "source_url": source_url,
                "source_path": source.name,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_title": "关于股票继续停牌的公告",
                "published_on": "2026-01-16",
            }
        ),
        encoding="utf-8",
    )
    return evidence


def test_loads_hash_bound_sse_evidence(tmp_path: Path) -> None:
    evidence = _write_evidence(
        tmp_path,
        source_url="https://star.sse.com.cn/disclosure/notice.pdf",
    )
    loaded = load_authoritative_suspension_evidence((evidence,))
    assert loaded.suspension_keys == {("688005.SH", date(2026, 1, 16))}
    assert len(loaded.artifacts) == 2


def test_rejects_non_exchange_url_and_tampered_source(tmp_path: Path) -> None:
    evidence = _write_evidence(tmp_path, source_url="https://example.com/notice.pdf")
    with pytest.raises(ValueError, match="official SSE"):
        load_authoritative_suspension_evidence((evidence,))

    document = json.loads(evidence.read_text(encoding="utf-8"))
    document["source_url"] = "https://www.sse.com.cn/notice.pdf"
    evidence.write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / "notice.pdf").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source_sha256 mismatch"):
        load_authoritative_suspension_evidence((evidence,))
