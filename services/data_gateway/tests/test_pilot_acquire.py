"""Tests for the pilot acquisition command using a fake vendor transport."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from ashare_data_gateway import load_published_dataset, pilot_acquire
from ashare_data_gateway.tushare_client import TOKEN_ENV_VAR
from ashare_data_gateway.tushare_transport import TransportResponse

DAILY_FIELDS = ("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount")
TRADING_DAYS = (
    "20230601",
    "20230602",
    "20230605",
    "20230606",
    "20230607",
    "20260727",
    "20260728",
    "20260729",
    "20260730",
    "20260731",
)


class FakeVendorTransport:
    """Deterministic vendor double with a resolvable base URL."""

    def __init__(self, *, base_url: str) -> None:
        self.base_url = base_url
        self.sends = 0

    def send(self, request, *, token: str) -> TransportResponse:
        self.sends += 1
        assert token
        params = dict(request.params)
        symbol = params["ts_code"]
        start_bound = params.get("start_date", "")
        end_bound = params.get("end_date", "")
        rows = []
        for index, trade_date in enumerate(TRADING_DAYS):
            if trade_date < start_bound or trade_date > end_bound:
                continue
            price = 10.0 + index * 0.1 + (0.5 if symbol.endswith("SH") else 0.0)
            volume_lots = 10000.0 + index * 100
            amount_thousand_yuan = volume_lots * 100 * price / 1000
            rows.append(
                (
                    symbol,
                    trade_date,
                    price,
                    price,
                    price,
                    price,
                    volume_lots,
                    round(amount_thousand_yuan, 3),
                )
            )
        return TransportResponse(
            code=0,
            msg="",
            fields=DAILY_FIELDS,
            items=tuple(rows),
        )


@pytest.fixture()
def fake_vendor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(TOKEN_ENV_VAR, "test-token")
    transport = FakeVendorTransport(base_url="https://forwarding.example")

    def factory() -> FakeVendorTransport:
        return transport

    monkeypatch.setattr(pilot_acquire, "HttpJsonTransport", factory)
    return transport


def test_acquire_publishes_immutable_dataset(fake_vendor, tmp_path: Path) -> None:
    result = pilot_acquire.acquire_and_publish(
        symbols=("000001.SZ", "600519.SH"),
        window_start=date(2023, 6, 1),
        window_end=date(2023, 6, 7),
        publication_root=tmp_path / "publication",
        generated_at=datetime(2026, 8, 4, 2, 0, tzinfo=UTC),
    )

    assert result["row_count"] == 10
    assert result["latest_trade_date"] == "2023-06-07"
    assert result["source"] == "tushare-via-non-official-endpoint"
    assert result["is_official_vendor"] is False
    assert "NON_OFFICIAL_VENDOR_ENDPOINT" in result["provenance_warnings"]

    loaded = load_published_dataset(
        publication_root=tmp_path / "publication",
        dataset_id=str(result["dataset_id"]),
    )
    assert len(loaded.records) == 10
    manifest = json.loads(Path(str(result["manifest_path"])).read_text(encoding="utf-8"))
    assert manifest["dataset_family_id"] == pilot_acquire.PILOT_DATASET_FAMILY_ID
    auxiliary_root = tmp_path / "publication" / "auxiliary"
    assert any(auxiliary_root.rglob("vendor-responses.json"))


def test_acquire_is_content_deterministic(fake_vendor, tmp_path: Path) -> None:
    arguments = {
        "symbols": ("000001.SZ", "600519.SH"),
        "window_start": date(2023, 6, 1),
        "window_end": date(2023, 6, 7),
        "generated_at": datetime(2026, 8, 4, 2, 0, tzinfo=UTC),
    }
    first = pilot_acquire.acquire_and_publish(
        publication_root=tmp_path / "one", **arguments
    )
    second = pilot_acquire.acquire_and_publish(
        publication_root=tmp_path / "two", **arguments
    )
    assert first["dataset_id"] == second["dataset_id"]


def test_probe_reports_capabilities_without_writing(fake_vendor, tmp_path: Path) -> None:
    monkey_result = pilot_acquire.probe_vendor()
    assert monkey_result["capabilities"]["daily"]["available"] is True
    assert monkey_result["base_url_host"] == "forwarding.example"
    assert not list(tmp_path.iterdir())


def test_missing_token_exits_with_dedicated_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    exit_code = pilot_acquire.main(["--probe"])
    assert exit_code == pilot_acquire.EXIT_TOKEN_MISSING
