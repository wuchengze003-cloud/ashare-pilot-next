"""Independent update system — a low-coupling orchestrator for the dashboard.

Splits the update chain into three self-documenting tiers so any agent can pick
it up by reading this file + docs/UPDATE_RUNBOOK.md. This script only calls the
existing modules (fetch / champion / stock_profile / render); it owns no
financial semantics and never touches strategy internals.

Usage:
  python tools/update.py intraday   # intraday: refresh prices + rerender (~1 min)
  python tools/update.py full       # EOD: collect -> retrain -> profile -> render
  python tools/update.py render     # rerender only (no fetch, no retrain)

Data flow:
  fetch_market_data.py / collect_alt_data.py  -> runtime/ raw data
  ashare_research_app.champion      -> runtime/dashboard/champion.json (retrain)
  ashare_research_app.stock_profile -> runtime/stock-profiles/*.json (research card)
  tools/render_dashboard.py         -> runtime/dashboard/*.html (static render)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def render() -> None:
    """Rerender all static pages from champion.json + stock-profiles."""
    _run([sys.executable, str(ROOT / "tools/render_dashboard.py")])


def intraday() -> None:
    """Refresh the latest close for current holdings, then rerender.

    Only mutates ``current_holdings[*].last_close`` — it does NOT retrain the
    model, regenerate signals, or touch backtest results. The price source is
    the tushare ``daily`` endpoint (latest close). For true intraday ticks the
    tdx/westock connectors are needed; the script falls back to the most recent
    daily close which is the honest approximation outside trading hours.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    from fetch_market_data import DataApi, load_env  # noqa: E402

    doc_path = ROOT / "runtime/dashboard/champion.json"
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    holdings = doc.get("current_holdings", [])
    if not holdings:
        print("当前无持仓，跳过行情刷新")
        render()
        return

    load_env()
    api = DataApi()
    today = date.today()
    start = (today - timedelta(days=12)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    n_ok = 0
    for h in holdings:
        sym = h.get("symbol")
        try:
            rows = api.daily(sym, start, end)
            if rows:
                h["last_close"] = round(float(rows[-1]["close"]), 2)
                n_ok += 1
                print(f"  {sym} {h.get('name')} 现价 -> {h['last_close']}")
            else:
                print(f"  {sym} 无行情")
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym} 拉行情失败: {exc}")

    doc_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"刷新 {n_ok}/{len(holdings)} 只持仓现价，已写回 champion.json")
    render()


def full() -> None:
    """End-of-day full update: collect alt data -> retrain -> profile -> render."""
    today = date.today().strftime("%Y%m%d")

    # 1. collect alt-data up to the latest trade date (breakpoint-resume)
    os.environ["ALT_END_DATE"] = today
    _run([sys.executable, str(ROOT / "tools/collect_alt_data.py")])

    # 2. retrain the rolling model + regenerate champion.json (~7 min)
    _run([sys.executable, "-m", "ashare_research_app.champion", str(ROOT)])

    # 3. regenerate per-stock research cards
    _run([sys.executable, "-m", "ashare_research_app.stock_profile", str(ROOT)])

    # 4. render all static pages
    render()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    mode = sys.argv[1]
    if mode == "intraday":
        intraday()
    elif mode == "full":
        full()
    elif mode == "render":
        render()
    else:
        print(f"未知模式: {mode!r}（可用: intraday / full / render）", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
