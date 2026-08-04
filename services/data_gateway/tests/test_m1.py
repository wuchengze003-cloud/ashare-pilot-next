"""End-to-end synthetic tests for the frozen M1 orchestration boundary."""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path

import ashare_data_gateway.m1 as m1_module
import pytest
from ashare_data_gateway.dataset_publication import (
    load_published_auxiliary_artifacts,
    load_published_dataset,
)
from ashare_data_gateway.m1 import (
    AUDIT_SYMBOLS,
    DATASET_AS_OF,
    DELISTED_SYMBOLS,
    EXPECTED_DELIST_DATES,
    LONG_SUSPENDED_SYMBOLS,
    NORMAL_SYMBOLS,
    M1RunSpec,
    run_m1,
)
from ashare_data_gateway.tushare_client import TokenMissingError
from ashare_data_gateway.tushare_transport import (
    TransportRequest,
    TransportResponse,
)

_CANARY = "<<m1-token-placeholder>>"
_OPEN_DAYS = (
    date(2023, 6, 1),
    date(2023, 6, 6),
    date(2023, 8, 4),
    date(2023, 10, 26),
    date(2023, 10, 31),
)


class StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


class FixtureM1Transport:
    """Request-aware transport; no network and no copied production data."""

    base_url = "https://proxy.example.com/tushare"

    def __init__(
        self,
        *,
        reverse_rows: bool = False,
        reverse_fields: bool = False,
        drop_normal_bar: bool = False,
        invalid_vwap: bool = False,
        nonfinite_daily: bool = False,
    ) -> None:
        self.reverse_rows = reverse_rows
        self.reverse_fields = reverse_fields
        self.drop_normal_bar = drop_normal_bar
        self.invalid_vwap = invalid_vwap
        self.nonfinite_daily = nonfinite_daily
        self.calls = 0

    @staticmethod
    def _params(request: TransportRequest) -> dict[str, str]:
        return dict(request.params)

    def _ordered(self, rows: list[list[object]]) -> list[list[object]]:
        return list(reversed(rows)) if self.reverse_rows else rows

    @staticmethod
    def _stock_row(symbol: str, status: str) -> list[object]:
        delist = EXPECTED_DELIST_DATES.get(symbol)
        return [
            symbol,
            symbol.split(".")[0],
            f"fixture-{symbol}",
            None,
            None,
            "主板",
            "20000101",
            delist.strftime("%Y%m%d") if delist else None,
            status,
        ]

    @staticmethod
    def _daily_dates(symbol: str) -> tuple[date, ...]:
        if symbol in LONG_SUSPENDED_SYMBOLS:
            return ()
        if symbol in NORMAL_SYMBOLS:
            return _OPEN_DAYS
        delist = EXPECTED_DELIST_DATES[symbol]
        return tuple(day for day in _OPEN_DAYS if day < delist)

    def send(self, request: TransportRequest, *, token: str) -> TransportResponse:
        assert token == _CANARY
        self.calls += 1
        params = self._params(request)
        rows: list[list[object]]

        if request.api_name == "trade_cal":
            rows = []
            current = date(2023, 6, 1)
            while current <= DATASET_AS_OF:
                rows.append(
                    [
                        "SSE",
                        current.strftime("%Y%m%d"),
                        1 if current in _OPEN_DAYS else 0,
                        None,
                    ]
                )
                current = current.fromordinal(current.toordinal() + 1)
        elif request.api_name == "stock_basic":
            status = params["list_status"]
            if status == "L":
                rows = [
                    self._stock_row(symbol, status)
                    for symbol in NORMAL_SYMBOLS + LONG_SUSPENDED_SYMBOLS
                ]
            elif status == "D":
                rows = [self._stock_row(symbol, status) for symbol in DELISTED_SYMBOLS]
                rows.extend(
                    [
                        [
                            "T600018.SH",
                            "T600018",
                            "fixture-rejected-one",
                            None,
                            None,
                            "主板",
                            "20000719",
                            "20061020",
                            "D",
                        ],
                        [
                            "TS0018.SH",
                            "TS0018",
                            "fixture-rejected-two",
                            None,
                            None,
                            None,
                            "20000719",
                            None,
                            "D",
                        ],
                    ]
                )
            else:
                rows = []
        elif request.api_name == "daily":
            symbol = params["ts_code"]
            dates = self._daily_dates(symbol)
            if self.drop_normal_bar and symbol == NORMAL_SYMBOLS[0]:
                dates = tuple(day for day in dates if day != _OPEN_DAYS[0])
            rows = []
            for day in dates:
                amount = (
                    1000.0
                    if self.invalid_vwap
                    and symbol == NORMAL_SYMBOLS[0]
                    and day == _OPEN_DAYS[0]
                    else 100.0
                )
                rows.append(
                    [
                        symbol,
                        day.strftime("%Y%m%d"),
                        (
                            math.nan
                            if self.nonfinite_daily
                            and symbol == NORMAL_SYMBOLS[0]
                            and day == _OPEN_DAYS[0]
                            else 10.0
                        ),
                        10.0,
                        10.0,
                        10.0,
                        100.0,
                        amount,
                    ]
                )
        elif request.api_name == "adj_factor":
            symbol = params["ts_code"]
            rows = [
                [symbol, day.strftime("%Y%m%d"), 1.0]
                for day in _OPEN_DAYS
            ]
        elif request.api_name == "suspend_d":
            rows = [
                [symbol, day.strftime("%Y%m%d"), None, "S"]
                for symbol in LONG_SUSPENDED_SYMBOLS
                for day in _OPEN_DAYS
            ]
            rows.append([NORMAL_SYMBOLS[0], "20230601", None, "R"])
        else:
            raise AssertionError(f"unexpected API: {request.api_name}")

        response_fields = request.fields
        if self.reverse_fields:
            response_fields = tuple(reversed(response_fields))
            rows = [list(reversed(row)) for row in rows]
        return TransportResponse(
            code=0,
            msg="",
            fields=response_fields,
            items=tuple(tuple(row) for row in self._ordered(rows)),
        )


def _read_json(path: Path) -> dict[str, object]:
    document = json.loads(path.read_bytes())
    assert isinstance(document, dict)
    return document


def _traceback_string_leaks(
    traceback: object,
    canary: str,
) -> list[tuple[str, str]]:
    leaks: list[tuple[str, str]] = []
    current = traceback
    while current is not None:
        frame = current.tb_frame
        if Path(frame.f_code.co_filename).resolve() != Path(__file__).resolve():
            for name, value in frame.f_locals.items():
                if isinstance(value, str) and canary in value:
                    leaks.append((frame.f_code.co_name, name))
        current = current.tb_next
    return leaks


def _spec(root: Path, *, run_id: str = "m1-fixture-run") -> M1RunSpec:
    return M1RunSpec(
        output_root=root,
        run_id=run_id,
        generated_at=datetime(2026, 8, 4, 8, 0, tzinfo=UTC),
    )


def test_m1_publishes_deterministic_dataset_and_auditable_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", _CANARY)
    output_root = tmp_path / "shared"
    first_transport = FixtureM1Transport()
    second_transport = FixtureM1Transport(reverse_rows=True, reverse_fields=True)

    first = run_m1(
        _spec(output_root, run_id="m1-fixture-first"),
        transport=first_transport,
        monotonic_fn=StepClock(),
        sleep_fn=lambda _delay: None,
    )
    second = run_m1(
        _spec(output_root, run_id="m1-fixture-second"),
        transport=second_transport,
        monotonic_fn=StepClock(),
        sleep_fn=lambda _delay: None,
    )

    assert len(AUDIT_SYMBOLS) == 20
    assert first.dataset_id == second.dataset_id
    assert first.normalized_row_count == second.normalized_row_count == 81
    assert first.coverage_expected_member_days == 94
    assert first.http_attempts == 45
    assert (first.dataset_dir / "daily-bars.json").read_bytes() == (
        second.dataset_dir / "daily-bars.json"
    ).read_bytes()

    loaded = load_published_dataset(
        publication_root=first.run_dir,
        dataset_id=first.dataset_id,
    )
    assert len(loaded.records) == 81
    assert loaded.manifest["source"] == "tushare-via-non-official-endpoint"

    coverage = _read_json(first.run_dir / "coverage-audit.json")
    assert coverage["passed"] is True
    assert coverage["bar_member_days"] == 81
    assert coverage["suspended_member_days"] == 10
    assert coverage["expected_delisted_member_day_count"] == 3
    assert coverage["missing_member_days"] == []
    assert coverage["silent_skip_symbols"] == []
    assert coverage["provenance_warnings"] == ["NON_OFFICIAL_VENDOR_ENDPOINT"]

    telemetry = _read_json(first.run_dir / "acquisition-telemetry.json")
    assert telemetry["base_url_host"] == "proxy.example.com"
    assert telemetry["is_official_vendor"] is False
    assert telemetry["logical_calls"] == 45
    assert telemetry["pages"] == 45
    assert telemetry["http_attempts"] == 45
    assert telemetry["http_successes"] == 45
    assert telemetry["http_failures"] == 0
    alignment = telemetry["daily_adj_alignment"]
    assert alignment["daily_only_keys"] == 0
    assert alignment["adj_factor_only_keys"] == 19
    assert alignment["matched_keys"] == 81
    assert alignment["daily_only_samples"] == []
    assert len(alignment["adj_factor_only_samples"]) == 19
    assert all(
        set(sample) == {"symbol", "trade_date"}
        for sample in alignment["adj_factor_only_samples"]
    )

    stock = _read_json(first.run_dir / "stock-basic-reconciliation.json")
    assert stock["L"]["raw_rows"] == 17
    assert stock["D"]["raw_rows"] == 5
    assert stock["D"]["accepted_rows"] == 3
    assert stock["D"]["rejected_rows"] == 2
    assert stock["P"]["raw_rows"] == 0

    auxiliary = {
        artifact.relative_path: artifact.content
        for artifact in load_published_auxiliary_artifacts(
            publication_root=first.run_dir,
            dataset_id=first.dataset_id,
        )
    }
    assert set(auxiliary) == {
        "adj_factor/records.json",
        "raw/vendor-responses.json",
    }
    assert _CANARY.encode() not in auxiliary["raw/vendor-responses.json"]
    raw = json.loads(auxiliary["raw/vendor-responses.json"])
    assert len(raw["responses"]) == 45

    receipt = _read_json(first.run_dir / "run-receipt.json")
    assert receipt["status"] == "PASS"
    assert receipt["dataset_id"] == first.dataset_id
    assert len(receipt["evidence_files"]) == 4
    assert len(receipt["auxiliary_files"]) == 2

    calls_before_rejected_repeat = first_transport.calls
    with pytest.raises(FileExistsError, match="run already exists"):
        run_m1(
            _spec(output_root, run_id="m1-fixture-first"),
            transport=first_transport,
            monotonic_fn=StepClock(),
            sleep_fn=lambda _delay: None,
        )
    assert first_transport.calls == calls_before_rejected_repeat


def test_m1_coverage_failure_does_not_publish_partial_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", _CANARY)
    transport = FixtureM1Transport(drop_normal_bar=True)

    with pytest.raises(ValueError, match="coverage audit failed"):
        run_m1(
            _spec(tmp_path),
            transport=transport,
            monotonic_fn=StepClock(),
            sleep_fn=lambda _delay: None,
        )

    assert not (tmp_path / "runs" / "m1-fixture-run").exists()
    assert not list((tmp_path / "runs").glob(".m1-fixture-run.tmp-*"))
    assert not list((tmp_path / "runs").glob("*/manifests/*.json"))
    failure = tmp_path / "failures" / "m1-fixture-run"
    receipt = _read_json(failure / "failure-receipt.json")
    assert receipt["status"] == "FAIL"
    assert receipt["reason_code"] == "VALIDATION_FAILED"
    assert _read_json(failure / "coverage-audit.json")["passed"] is False
    assert _read_json(failure / "acquisition-telemetry.json")["http_attempts"] == 45
    assert not (failure / "manifests").exists()


def test_m1_normalization_failure_retains_structured_issue_and_no_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", _CANARY)
    transport = FixtureM1Transport(invalid_vwap=True)

    with pytest.raises(ValueError, match="VWAP_OUTSIDE_OHLC_BAND"):
        run_m1(
            _spec(tmp_path),
            transport=transport,
            monotonic_fn=StepClock(),
            sleep_fn=lambda _delay: None,
        )

    assert not (tmp_path / "runs" / "m1-fixture-run").exists()
    assert not list((tmp_path / "runs").glob("*/manifests/*.json"))
    failure = tmp_path / "failures" / "m1-fixture-run"
    receipt = _read_json(failure / "failure-receipt.json")
    quality = _read_json(failure / "quality-summary.json")
    telemetry = _read_json(failure / "acquisition-telemetry.json")
    assert receipt["reason_code"] == "VWAP_OUTSIDE_OHLC_BAND"
    assert quality["normalization_issue_count"] == 1
    assert quality["normalization_issues"][0]["reason_code"] == (
        "VWAP_OUTSIDE_OHLC_BAND"
    )
    assert telemetry["http_attempts"] == 45
    assert telemetry["daily_adj_alignment"]["matched_keys"] == 81
    assert (failure / "raw-responses.json").is_file()


def test_m1_nonfinite_vendor_value_cannot_destroy_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", _CANARY)

    with pytest.raises(ValueError, match="non-finite"):
        run_m1(
            _spec(tmp_path),
            transport=FixtureM1Transport(nonfinite_daily=True),
            monotonic_fn=StepClock(),
            sleep_fn=lambda _delay: None,
        )

    failure = tmp_path / "failures" / "m1-fixture-run"
    receipt = _read_json(failure / "failure-receipt.json")
    raw = _read_json(failure / "raw-responses.json")
    assert receipt["status"] == "FAIL"
    assert receipt["error_type"] == "ValueError"
    assert "NaN" in json.dumps(raw, allow_nan=False)
    assert not list((tmp_path / "runs").glob("*/manifests/*.json"))


def test_m1_staged_validation_failure_leaves_only_failed_run_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", _CANARY)
    monkeypatch.setattr(
        m1_module,
        "_validate_staged_m1_run",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected staging fault")),
    )

    with pytest.raises(RuntimeError, match="injected staging fault"):
        run_m1(
            _spec(tmp_path),
            transport=FixtureM1Transport(),
            monotonic_fn=StepClock(),
            sleep_fn=lambda _delay: None,
        )

    assert not (tmp_path / "runs" / "m1-fixture-run").exists()
    assert not list((tmp_path / "runs").glob("*/manifests/*.json"))
    assert not list((tmp_path / "runs").glob(".m1-fixture-run.tmp-*"))
    failure = tmp_path / "failures" / "m1-fixture-run"
    assert _read_json(failure / "failure-receipt.json")["status"] == "FAIL"
    assert not (failure / "manifests").exists()


def test_m1_post_commit_fsync_failure_never_creates_a_second_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", _CANARY)
    original_fsync = m1_module._fsync_directory

    def fail_after_final_rename(path: Path) -> None:
        final_run = tmp_path / "runs" / "m1-fixture-run"
        if path == final_run.parent and final_run.is_dir():
            raise OSError("injected post-commit fsync fault")
        original_fsync(path)

    monkeypatch.setattr(m1_module, "_fsync_directory", fail_after_final_rename)
    with pytest.raises(OSError, match="post-commit fsync fault"):
        run_m1(
            _spec(tmp_path),
            transport=FixtureM1Transport(),
            monotonic_fn=StepClock(),
            sleep_fn=lambda _delay: None,
        )

    final_run = tmp_path / "runs" / "m1-fixture-run"
    assert _read_json(final_run / "run-receipt.json")["status"] == "PASS"
    assert not (tmp_path / "failures" / "m1-fixture-run").exists()


def test_m1_proxy_url_credentials_do_not_escape_into_traceback_or_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url_canary = "m1-url-credential-canary"

    class CredentialUrlTransport(FixtureM1Transport):
        base_url = (
            f"https://{url_canary}:password@proxy.example.com/vendor"
            f"?token={url_canary}#fragment"
        )

    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    with pytest.raises(TokenMissingError) as exc_info:
        run_m1(
            _spec(tmp_path),
            transport=CredentialUrlTransport(),
            monotonic_fn=StepClock(),
            sleep_fn=lambda _delay: None,
        )

    assert _traceback_string_leaks(exc_info.value.__traceback__, url_canary) == []
    failure = tmp_path / "failures" / "m1-fixture-run"
    persisted = b"".join(path.read_bytes() for path in failure.iterdir() if path.is_file())
    assert url_canary.encode() not in persisted
    telemetry = _read_json(failure / "acquisition-telemetry.json")
    assert telemetry["base_url_host"] == "proxy.example.com"
