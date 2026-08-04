"""Immutable normalized dataset publication tests; all fixtures are synthetic."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from ashare_data_gateway import dataset_publication
from ashare_data_gateway.dataset_publication import (
    AUXILIARY_INDEX_FILENAME,
    DATA_SCHEMA_DESCRIPTOR,
    DATA_SCHEMA_SHA256,
    NON_OFFICIAL_SOURCE,
    OFFICIAL_SOURCE,
    AuxiliaryArtifact,
    DatasetSourceIdentity,
    load_published_auxiliary_artifacts,
    load_published_dataset,
    prepare_normalized_dataset,
    publish_normalized_dataset,
    source_identity_from_base_url,
    validate_prepared_dataset,
)
from ashare_data_gateway.normalization import (
    canonical_json_bytes,
    normalize_daily_bars,
)
from ashare_data_gateway.tushare_models import DailyBarRecord
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[3]
AS_OF = date(2023, 10, 31)
GENERATED_AT = datetime(2026, 8, 4, 8, 0, tzinfo=UTC)


def leaked_traceback_locals(exc_tb: object, canary: str) -> list[str]:
    test_file = Path(__file__).resolve()
    leaked: list[str] = []
    tb = exc_tb
    while tb is not None:
        frame = tb.tb_frame
        if Path(frame.f_code.co_filename).resolve() != test_file:
            for name, value in frame.f_locals.items():
                if canary in repr(value):
                    leaked.append(f"{frame.f_code.co_name}.{name}")
        tb = tb.tb_next
    return leaked


def normalized_rows(*, reverse: bool = False):
    rows = [
        DailyBarRecord(
            ts_code="000001.SZ",
            trade_date=date(2023, 6, 1),
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.5,
            vol=100.0,
            amount=100.0,
        ),
        DailyBarRecord(
            ts_code="600519.SH",
            trade_date=date(2023, 10, 31),
            open=100.0,
            high=102.0,
            low=98.0,
            close=101.0,
            vol=20.0,
            amount=200.0,
        ),
    ]
    if reverse:
        rows.reverse()
    return normalize_daily_bars(
        rows,
        requested_symbols=("000001.SZ", "600519.SH"),
        window_start=date(2023, 6, 1),
        window_end=AS_OF,
    )


def prepare(
    *,
    reverse: bool = False,
    generated_at: datetime = GENERATED_AT,
    base_url: str = "https://api.tushare.pro",
    quality_status: str = "pass",
    quality_reasons: tuple[str, ...] = (),
):
    return prepare_normalized_dataset(
        normalized_rows(reverse=reverse),
        as_of=AS_OF,
        generated_at=generated_at,
        source_identity=source_identity_from_base_url(base_url),
        source_version="tushare-pro-api/v1",
        quality_status=quality_status,
        quality_reasons=quality_reasons,
    )


def test_descriptor_hash_matches_frozen_contract() -> None:
    descriptor_path = ROOT / "contracts/data_schemas/normalized-daily-bar-v1.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))

    assert descriptor == DATA_SCHEMA_DESCRIPTOR
    assert hashlib.sha256(canonical_json_bytes(descriptor)).hexdigest() == DATA_SCHEMA_SHA256
    assert DATA_SCHEMA_SHA256 == "f7c3a922e4ee8834bcea26870e689f99f90d2093c215ccd0baf72387567a5caf"


def test_manifest_is_schema_valid_and_self_consistent() -> None:
    prepared = prepare()
    manifest = prepared.manifest
    schema = json.loads(
        (ROOT / "contracts/schemas/dataset-manifest.schema.json").read_text(encoding="utf-8")
    )

    Draft202012Validator(schema, format_checker=FormatChecker()).validate(manifest)
    assert manifest["dataset_id"] == prepared.dataset_id
    assert manifest["files"][0]["sha256"] == hashlib.sha256(prepared.data_bytes).hexdigest()
    assert manifest["files"][0]["row_count"] == 2
    assert manifest["files"][0]["file_size_bytes"] == len(prepared.data_bytes)
    assert validate_prepared_dataset(prepared) == normalized_rows()


def test_input_order_and_generated_at_do_not_change_content_identity() -> None:
    first = prepare()
    second = prepare(
        reverse=True,
        generated_at=datetime(2026, 8, 4, 9, 30, tzinfo=UTC),
    )

    assert first.data_bytes == second.data_bytes
    assert first.manifest["files"] == second.manifest["files"]
    assert first.dataset_id == second.dataset_id
    assert first.manifest["generated_at"] != second.manifest["generated_at"]
    assert first.manifest_bytes != second.manifest_bytes


def test_source_identity_distinguishes_official_and_proxy_endpoints() -> None:
    official_identity = source_identity_from_base_url(
        "https://API.TUSHARE.PRO:443/vendor/path?ignored=true"
    )
    proxy_identity = source_identity_from_base_url(
        "https://user:password@Proxy.Example.COM:8443/tushare?token=ignored#fragment"
    )
    official = prepare(base_url="https://api.tushare.pro")
    proxy = prepare(base_url="https://proxy.example.com/tushare")

    assert official_identity.base_url_host == "api.tushare.pro"
    assert official_identity.is_official_vendor is True
    assert official_identity.manifest_source == OFFICIAL_SOURCE
    assert proxy_identity.base_url_host == "proxy.example.com"
    assert proxy_identity.is_official_vendor is False
    assert proxy_identity.manifest_source == NON_OFFICIAL_SOURCE
    assert official.manifest["source"] == OFFICIAL_SOURCE
    assert proxy.manifest["source"] == NON_OFFICIAL_SOURCE
    assert official.dataset_id != proxy.dataset_id
    assert "proxy.example.com" not in proxy.manifest_bytes.decode("utf-8")
    assert "password" not in proxy.manifest_bytes.decode("utf-8")


def test_prepare_api_accepts_only_sanitized_source_identity() -> None:
    parameters = inspect.signature(prepare_normalized_dataset).parameters
    assert "source_identity" in parameters
    assert "base_url" not in parameters

    with pytest.raises(TypeError, match="source_identity must be DatasetSourceIdentity"):
        prepare_normalized_dataset(
            normalized_rows(),
            as_of=AS_OF,
            generated_at=GENERATED_AT,
            source_identity="https://api.tushare.pro",  # type: ignore[arg-type]
            source_version="tushare-pro-api/v1",
        )


@pytest.mark.parametrize(
    "identity",
    [
        DatasetSourceIdentity("api.tushare.pro", True, OFFICIAL_SOURCE),
        DatasetSourceIdentity("proxy.example.com", False, NON_OFFICIAL_SOURCE),
    ],
)
def test_valid_source_identity_is_frozen(identity: DatasetSourceIdentity) -> None:
    with pytest.raises(AttributeError):
        identity.base_url_host = "changed.example.com"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "base_url_host": "API.TUSHARE.PRO",
            "is_official_vendor": True,
            "manifest_source": OFFICIAL_SOURCE,
        },
        {
            "base_url_host": "proxy.example.com:443",
            "is_official_vendor": False,
            "manifest_source": NON_OFFICIAL_SOURCE,
        },
        {
            "base_url_host": "proxy.example.com",
            "is_official_vendor": True,
            "manifest_source": OFFICIAL_SOURCE,
        },
        {
            "base_url_host": "proxy.example.com",
            "is_official_vendor": False,
            "manifest_source": OFFICIAL_SOURCE,
        },
    ],
)
def test_inconsistent_or_unsanitized_source_identity_is_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        DatasetSourceIdentity(**kwargs)  # type: ignore[arg-type]


def test_source_url_credentials_do_not_survive_into_later_tracebacks() -> None:
    url_canary = "publisher-url-credential-canary"
    base_url = (
        f"https://user:{url_canary}@Proxy.Example.COM:8443/vendor"
        f"?token={url_canary}#{url_canary}"
    )
    identity = source_identity_from_base_url(base_url)

    with pytest.raises(ValueError, match="safe POSIX relative path") as exc_info:
        prepare_normalized_dataset(
            normalized_rows(),
            as_of=AS_OF,
            generated_at=GENERATED_AT,
            source_identity=identity,
            source_version="tushare-pro-api/v1",
            relative_data_path="../unsafe.json",
        )

    assert identity == DatasetSourceIdentity(
        base_url_host="proxy.example.com",
        is_official_vendor=False,
        manifest_source=NON_OFFICIAL_SOURCE,
    )
    assert leaked_traceback_locals(exc_info.value.__traceback__, url_canary) == []


def test_source_parser_cleans_url_before_failure_traceback() -> None:
    url_canary = "parser-url-credential-canary"
    invalid_url = (
        f"ftp://user:{url_canary}@Proxy.Example.COM/vendor"
        f"?token={url_canary}#{url_canary}"
    )

    with pytest.raises(ValueError, match="must use http or https") as exc_info:
        source_identity_from_base_url(invalid_url)

    assert leaked_traceback_locals(exc_info.value.__traceback__, url_canary) == []


@pytest.mark.parametrize(
    "relative_path",
    ["/daily-bars.json", "../daily-bars.json", "daily/../bars.json", "daily\\bars.json"],
)
def test_unsafe_manifest_data_paths_are_rejected(relative_path: str) -> None:
    with pytest.raises(ValueError, match="safe POSIX relative path"):
        prepare_normalized_dataset(
            normalized_rows(),
            as_of=AS_OF,
            generated_at=GENERATED_AT,
            source_identity=source_identity_from_base_url("https://api.tushare.pro"),
            source_version="tushare-pro-api/v1",
            relative_data_path=relative_path,
        )


def test_publish_moves_data_then_publishes_manifest_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare()
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(dataset_publication.os, "replace", recording_replace)

    paths = publish_normalized_dataset(
        publication_root=tmp_path,
        prepared=prepared,
    )

    assert [destination for _, destination in replacements] == [
        paths.dataset_dir,
        paths.manifest_path,
    ]
    assert paths.dataset_dir.is_dir()
    assert paths.manifest_path.is_file()
    assert paths.manifest_path.read_bytes() == prepared.manifest_bytes
    loaded = load_published_dataset(
        publication_root=tmp_path,
        dataset_id=prepared.dataset_id,
    )
    assert loaded.records == normalized_rows()
    assert loaded.data_bytes == prepared.data_bytes


def test_raw_and_adj_factor_evidence_stays_outside_normalized_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auxiliary = (
        AuxiliaryArtifact(
            relative_path="raw/daily/000001.SZ.json",
            content=b'{"vendor":"raw-daily"}',
        ),
        AuxiliaryArtifact(
            relative_path="adj_factor/000001.SZ.json",
            content=b'[{"adj_factor":1.0}]',
        ),
    )
    prepared = prepare_normalized_dataset(
        normalized_rows(),
        as_of=AS_OF,
        generated_at=GENERATED_AT,
        source_identity=source_identity_from_base_url("https://api.tushare.pro"),
        source_version="tushare-pro-api/v1",
        auxiliary_artifacts=reversed(auxiliary),
    )
    without_auxiliary = prepare()
    replacements: list[Path] = []
    real_replace = os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        replacements.append(Path(destination))
        real_replace(source, destination)

    monkeypatch.setattr(dataset_publication.os, "replace", recording_replace)

    paths = publish_normalized_dataset(publication_root=tmp_path, prepared=prepared)

    assert prepared.dataset_id == without_auxiliary.dataset_id
    assert prepared.manifest_bytes == without_auxiliary.manifest_bytes
    assert paths.auxiliary_dir == tmp_path / "auxiliary" / prepared.dataset_id
    assert paths.auxiliary_dir is not None
    assert replacements == [
        paths.auxiliary_dir,
        paths.dataset_dir,
        paths.manifest_path,
    ]
    assert (paths.auxiliary_dir / AUXILIARY_INDEX_FILENAME).is_file()
    assert prepared.manifest["files"] == [
        {
            "path": "daily-bars.json",
            "sha256": hashlib.sha256(prepared.data_bytes).hexdigest(),
            "row_count": 2,
            "file_size_bytes": len(prepared.data_bytes),
            "min_trade_date": "2023-06-01",
            "max_trade_date": "2023-10-31",
        }
    ]
    assert not any("raw" in item["path"] for item in prepared.manifest["files"])
    assert not any("adj_factor" in item["path"] for item in prepared.manifest["files"])
    assert (
        load_published_dataset(
            publication_root=tmp_path,
            dataset_id=prepared.dataset_id,
        ).records
        == normalized_rows()
    )
    assert load_published_auxiliary_artifacts(
        publication_root=tmp_path,
        dataset_id=prepared.dataset_id,
    ) == tuple(sorted(auxiliary, key=lambda item: item.relative_path))


def test_tampered_prepared_bytes_fail_before_publication(tmp_path: Path) -> None:
    prepared = prepare()
    tampered = replace(
        prepared,
        data_bytes=prepared.data_bytes.replace(b'"close":10.5', b'"close":10.6'),
    )

    with pytest.raises(ValueError, match="hash"):
        publish_normalized_dataset(publication_root=tmp_path, prepared=tampered)

    assert not (tmp_path / "datasets" / prepared.dataset_id).exists()
    assert not (tmp_path / "manifests" / f"{prepared.dataset_id}.json").exists()


def test_loader_rejects_post_publication_tampering(tmp_path: Path) -> None:
    prepared = prepare()
    paths = publish_normalized_dataset(publication_root=tmp_path, prepared=prepared)
    data_path = paths.dataset_dir / prepared.relative_data_path
    data_path.write_bytes(prepared.data_bytes.replace(b'"close":10.5', b'"close":10.6'))

    with pytest.raises(ValueError, match="hash"):
        load_published_dataset(
            publication_root=tmp_path,
            dataset_id=prepared.dataset_id,
        )


def test_manifest_publish_failure_removes_new_data_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare()
    real_replace = os.replace
    replace_count = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("synthetic manifest commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(dataset_publication.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="synthetic manifest commit failure"):
        publish_normalized_dataset(publication_root=tmp_path, prepared=prepared)

    assert replace_count == 2
    assert not (tmp_path / "datasets" / prepared.dataset_id).exists()
    assert not (tmp_path / "manifests" / f"{prepared.dataset_id}.json").exists()
    assert not list((tmp_path / "datasets").glob("*.tmp-*"))
    assert not list((tmp_path / "manifests").glob("*.tmp"))


def test_manifest_failure_also_removes_moved_auxiliary_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_normalized_dataset(
        normalized_rows(),
        as_of=AS_OF,
        generated_at=GENERATED_AT,
        source_identity=source_identity_from_base_url("https://api.tushare.pro"),
        source_version="tushare-pro-api/v1",
        auxiliary_artifacts=(
            AuxiliaryArtifact("raw/vendor-responses.json", b"[]"),
            AuxiliaryArtifact("adj_factor/records.json", b"[]"),
        ),
    )
    real_replace = os.replace
    replace_count = 0

    def fail_manifest_replace(source: Path, destination: Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 3:
            raise OSError("synthetic manifest commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(dataset_publication.os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="synthetic manifest commit failure"):
        publish_normalized_dataset(publication_root=tmp_path, prepared=prepared)

    assert not (tmp_path / "datasets" / prepared.dataset_id).exists()
    assert not (tmp_path / "auxiliary" / prepared.dataset_id).exists()
    assert not (tmp_path / "manifests" / f"{prepared.dataset_id}.json").exists()


def test_existing_publication_is_never_overwritten(tmp_path: Path) -> None:
    prepared = prepare()
    paths = publish_normalized_dataset(publication_root=tmp_path, prepared=prepared)
    original_data = (paths.dataset_dir / prepared.relative_data_path).read_bytes()
    original_manifest = paths.manifest_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        publish_normalized_dataset(publication_root=tmp_path, prepared=prepared)

    assert (paths.dataset_dir / prepared.relative_data_path).read_bytes() == original_data
    assert paths.manifest_path.read_bytes() == original_manifest


def test_orphan_auxiliary_path_cannot_be_silently_attached(tmp_path: Path) -> None:
    prepared = prepare()
    orphan = tmp_path / "auxiliary" / prepared.dataset_id
    orphan.mkdir(parents=True)
    (orphan / "unexpected.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="auxiliary evidence already exists"):
        publish_normalized_dataset(publication_root=tmp_path, prepared=prepared)

    assert not (tmp_path / "datasets" / prepared.dataset_id).exists()
    assert not (tmp_path / "manifests" / f"{prepared.dataset_id}.json").exists()


def test_failed_quality_dataset_is_not_publishable(tmp_path: Path) -> None:
    prepared = prepare(
        quality_status="fail",
        quality_reasons=("VWAP_OUTSIDE_OHLC_BAND",),
    )

    with pytest.raises(ValueError, match="failed-quality"):
        publish_normalized_dataset(publication_root=tmp_path, prepared=prepared)

    assert not (tmp_path / "datasets").exists()
    assert not (tmp_path / "manifests").exists()
