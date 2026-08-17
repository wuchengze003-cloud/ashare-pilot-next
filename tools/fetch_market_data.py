"""Fetch A-share market data from a Tushare-compatible HTTP proxy.

Credentials and endpoint are read from the environment or a project-root
``.env`` file (both git-ignored):

  TUSHARE_TOKEN      API key (required)
  TUSHARE_HTTP_URL   base URL (default: https://teajoin.com)

Supported endpoints (Tushare Pro protocol):

  ``daily``        daily bars (unadjusted OHLCV)
  ``stk_mins``     minute bars (freq: 1min/5min/15min/30min/60min)
  ``daily_basic``  daily fundamentals (market cap, PE, PB, turnover)

Calls are rate-limited (>=0.2s) and retried with backoff. Output lands under
``runtime/`` (git-ignored). This module is data acquisition only — it performs
no financial semantics, which remain the sole authority of ``quant_core``.

CLI examples::

  python tools/fetch_market_data.py daily 600519.SH 20260801 20260814
  python tools/fetch_market_data.py mins 600519.SH 5min 2026-08-11 2026-08-14
  python tools/fetch_market_data.py daily_basic 20260814
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib import error, request

DEFAULT_URL = "https://teajoin.com"
MIN_INTERVAL = 0.2
MAX_RETRIES = 3
TIMEOUT = 30

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Populate os.environ from <project>/.env without overriding set values."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("TUSHARE_TOKEN is not set (see .env)")
    return token


def _base_url() -> str:
    return os.environ.get("TUSHARE_HTTP_URL", "").strip() or DEFAULT_URL


class DataApi:
    """Minimal rate-limited client for the Tushare-compatible proxy."""

    def __init__(self, token: str | None = None, base_url: str | None = None) -> None:
        self.token = token or _token()
        self.base_url = (base_url or _base_url()).rstrip("/")
        self._last_call = 0.0

    def call(self, api_name: str, params: dict) -> dict:
        self._throttle()
        body = json.dumps(
            {"api_name": api_name, "token": self.token, "params": params}
        ).encode("utf-8")
        req = request.Request(
            self.base_url, data=body, headers={"Content-Type": "application/json"}
        )
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                with request.urlopen(req, timeout=TIMEOUT) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except error.URLError as exc:
                last_err = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"{api_name} failed after {MAX_RETRIES} attempts: {last_err}")

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._last_call + MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    @staticmethod
    def _items(payload: dict) -> list[dict]:
        if payload.get("code") != 0:
            raise RuntimeError(f"api error: {payload.get('msg')!r}")
        data = payload.get("data", {})
        fields = data.get("fields", [])
        return [dict(zip(fields, row, strict=False)) for row in data.get("items", [])]

    def daily(self, ts_code: str, start_date: str, end_date: str) -> list[dict]:
        return self._items(
            self.call(
                "daily",
                {"ts_code": ts_code, "start_date": start_date, "end_date": end_date},
            )
        )

    def minutes(
        self, ts_code: str, freq: str, start: str, end: str
    ) -> list[dict]:
        return self._items(
            self.call(
                "stk_mins",
                {
                    "ts_code": ts_code,
                    "freq": freq,
                    "start_date": start,
                    "end_date": end,
                },
            )
        )

    def daily_basic(self, trade_date: str) -> list[dict]:
        return self._items(self.call("daily_basic", {"trade_date": trade_date}))


def _save(items: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(items)} rows -> {out}")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("daily", help="daily bars")
    d.add_argument("ts_code")
    d.add_argument("start_date")
    d.add_argument("end_date")

    m = sub.add_parser("mins", help="minute bars")
    m.add_argument("ts_code")
    m.add_argument("freq", choices=["1min", "5min", "15min", "30min", "60min"])
    m.add_argument("start")
    m.add_argument("end")

    b = sub.add_parser("daily_basic", help="daily fundamentals for one date")
    b.add_argument("trade_date")

    args = parser.parse_args()
    api = DataApi()

    if args.cmd == "daily":
        rows = api.daily(args.ts_code, args.start_date, args.end_date)
        out = PROJECT_ROOT / "runtime" / "proxy" / "daily" / f"{args.ts_code}.json"
    elif args.cmd == "mins":
        rows = api.minutes(args.ts_code, args.freq, args.start, args.end)
        out = (
            PROJECT_ROOT
            / "runtime"
            / "proxy"
            / "mins"
            / f"{args.ts_code}_{args.freq}.json"
        )
    else:
        rows = api.daily_basic(args.trade_date)
        out = (
            PROJECT_ROOT
            / "runtime"
            / "proxy"
            / "daily_basic"
            / f"{args.trade_date}.json"
        )

    _save(rows, out)


if __name__ == "__main__":
    main()
