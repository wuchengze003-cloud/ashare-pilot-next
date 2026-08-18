"""Load immutable, human-reviewed exchange suspension evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from .dataset_publication import AuxiliaryArtifact, canonical_json_bytes

ALLOWED_EXCHANGE_HOSTS = frozenset({"sse.com.cn", "www.sse.com.cn", "star.sse.com.cn"})


@dataclass(frozen=True, order=True)
class AuthoritativeSuspension:
    symbol: str
    trade_date: date
    source_url: str
    source_sha256: str
    source_title: str
    published_on: date


@dataclass(frozen=True)
class LoadedSuspensionEvidence:
    records: tuple[AuthoritativeSuspension, ...]
    artifacts: tuple[AuxiliaryArtifact, ...]

    @property
    def suspension_keys(self) -> frozenset[tuple[str, date]]:
        return frozenset((record.symbol, record.trade_date) for record in self.records)


def load_authoritative_suspension_evidence(
    evidence_paths: tuple[Path, ...],
) -> LoadedSuspensionEvidence:
    """Validate exchange evidence and bind both its metadata and source bytes.

    Each JSON document points to a local copy of the exchange announcement. The
    announcement bytes are hash checked and published inside the immutable
    dataset. This path is intentionally explicit: vendor ``R`` records never
    become suspension evidence on their own.
    """
    records: list[AuthoritativeSuspension] = []
    artifacts: list[AuxiliaryArtifact] = []
    seen_keys: set[tuple[str, date]] = set()
    for raw_path in sorted(Path(path) for path in evidence_paths):
        if raw_path.is_symlink() or not raw_path.is_file():
            raise ValueError(f"suspension evidence must be a regular file: {raw_path}")
        document = json.loads(raw_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("suspension evidence must be an object")
        if document.get("evidence_id") != "exchange-suspension-evidence/v1":
            raise ValueError("unsupported suspension evidence_id")
        source_path_value = document.get("source_path")
        if not isinstance(source_path_value, str) or not source_path_value:
            raise ValueError("suspension evidence source_path is required")
        source_path = raw_path.parent / source_path_value
        if (
            source_path_value != Path(source_path_value).name
            or source_path.is_symlink()
            or not source_path.is_file()
        ):
            raise ValueError("suspension evidence source_path must name a sibling file")
        source_bytes = source_path.read_bytes()
        actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if document.get("source_sha256") != actual_sha256:
            raise ValueError("suspension evidence source_sha256 mismatch")
        source_url = str(document.get("source_url", ""))
        parsed_url = urlparse(source_url)
        if parsed_url.scheme != "https" or parsed_url.hostname not in ALLOWED_EXCHANGE_HOSTS:
            raise ValueError("suspension evidence must use an official SSE HTTPS URL")
        if document.get("status") != "S":
            raise ValueError("authoritative evidence may only declare suspension status S")
        source_title = document.get("source_title")
        if not isinstance(source_title, str) or not source_title.strip():
            raise ValueError("suspension evidence source_title is required")
        record = AuthoritativeSuspension(
            symbol=str(document["symbol"]),
            trade_date=date.fromisoformat(str(document["trade_date"])),
            source_url=source_url,
            source_sha256=actual_sha256,
            source_title=source_title.strip(),
            published_on=date.fromisoformat(str(document["published_on"])),
        )
        key = (record.symbol, record.trade_date)
        if key in seen_keys:
            raise ValueError(f"duplicate authoritative suspension evidence: {key}")
        if record.published_on > record.trade_date:
            raise ValueError("suspension evidence was published after the covered date")
        seen_keys.add(key)
        records.append(record)
        prefix = f"raw/authoritative-suspensions/{record.symbol}-{record.trade_date.isoformat()}"
        artifacts.extend(
            (
                AuxiliaryArtifact(
                    relative_path=f"{prefix}.json",
                    content=canonical_json_bytes(
                        {
                            "evidence_id": "exchange-suspension-evidence/v1",
                            "symbol": record.symbol,
                            "trade_date": record.trade_date.isoformat(),
                            "status": "S",
                            "source_url": record.source_url,
                            "source_sha256": record.source_sha256,
                            "source_title": record.source_title,
                            "published_on": record.published_on.isoformat(),
                        }
                    ),
                ),
                AuxiliaryArtifact(
                    relative_path=f"{prefix}{source_path.suffix.lower()}",
                    content=source_bytes,
                ),
            )
        )
    return LoadedSuspensionEvidence(
        records=tuple(sorted(records)),
        artifacts=tuple(artifacts),
    )
