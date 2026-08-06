"""Public command: acquire pilot daily bars and publish an immutable dataset.

Reads credentials ONLY from environment variables (TUSHARE_TOKEN and
optional TUSHARE_HTTP_URL). Never echoes token values. Includes a
--probe mode that reports vendor capability and latency without writing
anything, so callers can degrade gracefully.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

from .acquisition import CountingTransport
from .dataset_publication import (
    AuxiliaryArtifact,
    prepare_normalized_dataset,
    publish_normalized_dataset,
)
from .normalization import normalize_daily_bars
from .tushare_client import TOKEN_ENV_VAR, TokenMissingError, TushareClient
from .tushare_transport import (
    URL_ENV_VAR,
    HttpJsonTransport,
    TushareAPIError,
    TushareTransientError,
)

PILOT_DATASET_FAMILY_ID = "pilot-daily/v1"
EXIT_OK = 0
EXIT_TOKEN_MISSING = 2
EXIT_VENDOR_FAILURE = 3


def _sanitize_text(value: str) -> str:
    token = os.environ.get(TOKEN_ENV_VAR, "")
    sanitized = value.replace(token, "[REDACTED]") if token else value
    return sanitized[:400]


def _build_client() -> tuple[TushareClient, CountingTransport]:
    transport = HttpJsonTransport()
    counting = CountingTransport(transport, monotonic_fn=time.monotonic)
    return TushareClient(counting), counting


def probe_vendor() -> dict[str, object]:
    """Probe forwarding-service capabilities without publishing anything."""
    client, counting = _build_client()
    probe_symbol = "000001.SZ"
    probe_end = date(2026, 8, 3)
    capabilities: dict[str, object] = {}

    started = time.monotonic()
    try:
        bars = client.fetch_daily(
            ts_code=probe_symbol,
            start_date=date(2026, 7, 25),
            end_date=probe_end,
        )
        capabilities["daily"] = {
            "available": True,
            "rows": len(bars),
            "latency_seconds": round(time.monotonic() - started, 3),
        }
    except (TushareAPIError, TushareTransientError) as exc:
        capabilities["daily"] = {"available": False, "error": _sanitize_text(str(exc))}

    return {
        "base_url_host": counting.base_url_host,
        "is_official_vendor": counting.is_official_vendor,
        "token_env_var": TOKEN_ENV_VAR,
        "url_env_var": URL_ENV_VAR,
        "probe_window_end": probe_end.isoformat(),
        "capabilities": capabilities,
    }


def acquire_and_publish(
    *,
    symbols: tuple[str, ...],
    window_start: date,
    window_end: date,
    publication_root: Path,
    generated_at: datetime,
) -> dict[str, object]:
    client, counting = _build_client()
    raw_records = []
    for symbol in symbols:
        raw_records.extend(
            client.fetch_daily(ts_code=symbol, start_date=window_start, end_date=window_end)
        )
    normalized = normalize_daily_bars(
        raw_records,
        requested_symbols=symbols,
        window_start=window_start,
        window_end=window_end,
    )
    if not normalized:
        raise ValueError("pilot acquisition produced no normalized rows")

    evidence_payload = json.dumps(
        {
            "telemetry": counting.snapshot().to_dict(),
            "raw_responses": [item.to_dict() for item in counting.raw_snapshot()],
        },
        ensure_ascii=True,
        sort_keys=True,
    ).encode()
    prepared = prepare_normalized_dataset(
        normalized,
        as_of=window_end,
        generated_at=generated_at,
        source_identity=counting.source_identity,
        source_version="pilot-fetch/v1",
        dataset_family_id=PILOT_DATASET_FAMILY_ID,
        auxiliary_artifacts=(
            AuxiliaryArtifact(relative_path="raw/vendor-responses.json", content=evidence_payload),
        ),
    )
    paths = publish_normalized_dataset(
        publication_root=publication_root,
        prepared=prepared,
    )
    manifest = prepared.manifest
    files = manifest["files"]
    assert isinstance(files, list)
    first_file = files[0]
    assert isinstance(first_file, dict)
    return {
        "dataset_id": paths.dataset_id,
        "manifest_path": str(paths.manifest_path),
        "dataset_dir": str(paths.dataset_dir),
        "publication_root": str(publication_root),
        "source": manifest["source"],
        "is_official_vendor": counting.is_official_vendor,
        "base_url_host": counting.base_url_host,
        "latest_trade_date": first_file["max_trade_date"],
        "row_count": first_file["row_count"],
        "provenance_warnings": list(counting.provenance_warnings),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pilot daily acquisition")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--window-start", default="")
    parser.add_argument("--window-end", default="")
    parser.add_argument("--publication-root", default="")
    parser.add_argument("--generated-at", default="")
    args = parser.parse_args(argv)

    try:
        if args.probe:
            json.dump(probe_vendor(), sys.stdout, ensure_ascii=True)
            sys.stdout.write("\n")
            return EXIT_OK
        if not (args.symbols and args.window_start and args.window_end):
            raise ValueError("--symbols, --window-start and --window-end are required")
        generated_at = datetime.fromisoformat(args.generated_at.replace("Z", "+00:00"))
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=UTC)
        result = acquire_and_publish(
            symbols=tuple(item.strip() for item in args.symbols.split(",") if item.strip()),
            window_start=date.fromisoformat(args.window_start),
            window_end=date.fromisoformat(args.window_end),
            publication_root=Path(args.publication_root),
            generated_at=generated_at,
        )
        json.dump(result, sys.stdout, ensure_ascii=True)
        sys.stdout.write("\n")
        return EXIT_OK
    except TokenMissingError:
        json.dump(
            {
                "error": "TOKEN_MISSING",
                "detail": f"{TOKEN_ENV_VAR} environment variable is not set",
            },
            sys.stdout,
            ensure_ascii=True,
        )
        sys.stdout.write("\n")
        return EXIT_TOKEN_MISSING
    except (TushareAPIError, TushareTransientError) as exc:
        json.dump(
            {"error": "VENDOR_FAILURE", "detail": _sanitize_text(str(exc))},
            sys.stdout,
            ensure_ascii=True,
        )
        sys.stdout.write("\n")
        return EXIT_VENDOR_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
