"""A股另类数据采集脚本（teajoin.com / Tushare 兼容协议）。

按接口串行采集，断点续传（已存在且非空的文件跳过），失败记录到
``runtime/alt-data/_failed.txt``。保存格式：{"fields": [...], "items": [[...]]}。

用法:
  python tools/collect_alt_data.py moneyflow daily_basic top_list top_inst \
      hsgt_top10 moneyflow_hsgt margin_detail cyq_perf stk_holdernumber

支持按日接口（trade_date 循环）与逐只接口（ts_code 循环）两类，脚本自动判别。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from fetch_market_data import DataApi, load_env  # noqa: E402

ALT_DIR = ROOT / "runtime" / "alt-data"
FAILED_FILE = ALT_DIR / "_failed.txt"

START_DATE = "20230101"
# 结束日期可用环境变量 ALT_END_DATE 覆盖（收盘后完整更新时由 update.py 设为最新交易日）
END_DATE = os.environ.get("ALT_END_DATE", "20260814")

# 按日接口：参数键为 trade_date
BY_DATE_APIS = {
    "moneyflow", "top_list", "top_inst", "hsgt_top10",
    "moneyflow_hsgt", "margin_detail", "daily_basic",
}
# 逐只接口：参数键为 ts_code + 区间
BY_TS_APIS = {"cyq_perf", "stk_holdernumber"}


def load_symbols() -> list[str]:
    meta = json.loads((ROOT / "runtime" / "full-market" / "symbols.json").read_text("utf-8"))
    return [s["symbol"] for s in meta["symbols"]]


def get_trade_dates(api: DataApi) -> list[str]:
    r = api.call("trade_cal", {"exchange": "SSE", "start_date": START_DATE, "end_date": END_DATE})
    data = r.get("data", {})
    fields = data.get("fields", [])
    ci = fields.index("cal_date")
    oi = fields.index("is_open")
    return [row[ci] for row in data.get("items", []) if row[oi] == 1]


def save(subdir: str, key: str, fields: list, items: list) -> bool:
    """写入紧凑 JSON；已存在且非空则跳过返回 False。"""
    d = ALT_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key}.json"
    if path.exists() and path.stat().st_size > 0:
        return False
    path.write_text(
        json.dumps({"fields": fields, "items": items}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return True


def log_fail(api_name: str, params: dict, reason: str) -> None:
    with open(FAILED_FILE, "a", encoding="utf-8") as f:
        f.write(f"{api_name} | {json.dumps(params, ensure_ascii=False)} | {reason}\n")


def fetch_by_date(api: DataApi, api_name: str, dates: list[str]) -> tuple[int, int]:
    subdir = api_name
    total_rows = 0
    failed = 0
    n = len(dates)
    for i, d in enumerate(dates):
        path = ALT_DIR / subdir / f"{d}.json"
        if path.exists() and path.stat().st_size > 0:
            continue
        params = {"trade_date": d}
        done = False
        for attempt in range(3):
            try:
                r = api.call(api_name, params)
                if r.get("code") == 0:
                    data = r.get("data", {})
                    fields = data.get("fields", [])
                    items = data.get("items", [])
                    save(subdir, d, fields, items)
                    total_rows += len(items)
                    done = True
                    break
                else:
                    log_fail(api_name, params, f"code={r.get('code')} msg={r.get('msg')!r}")
                    failed += 1
                    done = True
                    break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    log_fail(api_name, params, str(e))
                    failed += 1
                time.sleep(1.5 * (attempt + 1))
        if done:
            time.sleep(0.05)  # 兜底间隔，配合 DataApi 内置 0.2s
        if (i + 1) % 100 == 0:
            print(f"[{api_name}] {i + 1}/{n}  total_rows={total_rows}  failed={failed}", flush=True)
    print(f"[{api_name}] DONE  total_rows={total_rows}  failed={failed}", flush=True)
    return total_rows, failed


def fetch_by_ts(api: DataApi, api_name: str, symbols: list[str]) -> tuple[int, int]:
    subdir = api_name
    total_rows = 0
    failed = 0
    n = len(symbols)
    for i, sym in enumerate(symbols):
        path = ALT_DIR / subdir / f"{sym}.json"
        if path.exists() and path.stat().st_size > 0:
            continue
        params = {"ts_code": sym, "start_date": START_DATE, "end_date": END_DATE}
        done = False
        for attempt in range(3):
            try:
                r = api.call(api_name, params)
                if r.get("code") == 0:
                    data = r.get("data", {})
                    fields = data.get("fields", [])
                    items = data.get("items", [])
                    save(subdir, sym, fields, items)
                    total_rows += len(items)
                    done = True
                    break
                else:
                    log_fail(api_name, params, f"code={r.get('code')} msg={r.get('msg')!r}")
                    failed += 1
                    done = True
                    break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    log_fail(api_name, params, str(e))
                    failed += 1
                time.sleep(1.5 * (attempt + 1))
        if done:
            time.sleep(0.05)
        if (i + 1) % 500 == 0:
            print(f"[{api_name}] {i + 1}/{n}  total_rows={total_rows}  failed={failed}", flush=True)
    print(f"[{api_name}] DONE  total_rows={total_rows}  failed={failed}", flush=True)
    return total_rows, failed


def main() -> None:
    load_env()
    apis = sys.argv[1:]
    if not apis:
        apis = sorted(BY_DATE_APIS | BY_TS_APIS)
    for a in apis:
        if a not in BY_DATE_APIS and a not in BY_TS_APIS:
            print(f"未知接口: {a}", file=sys.stderr)
            sys.exit(2)

    api = DataApi()
    if any(a in BY_DATE_APIS for a in apis):
        dates = get_trade_dates(api)
        print(
            f"交易日数(2023-01-01~2026-08-14): {len(dates)}  首={dates[0]} 末={dates[-1]}",
            flush=True,
        )

    symbols: list[str] = []
    if any(a in BY_TS_APIS for a in apis):
        symbols = load_symbols()
        print(f"股票数: {len(symbols)}", flush=True)

    summary = {}
    for a in apis:
        t0 = time.time()
        if a in BY_DATE_APIS:
            total, failed = fetch_by_date(api, a, dates)
        else:
            total, failed = fetch_by_ts(api, a, symbols)
        summary[a] = {"rows": total, "failed": failed, "secs": round(time.time() - t0, 1)}
        print(f"== {a}: rows={total} failed={failed} {summary[a]['secs']}s", flush=True)

    print("\n=== 汇总 ===", flush=True)
    for a, s in summary.items():
        print(f"{a}: rows={s['rows']} failed={s['failed']} secs={s['secs']}", flush=True)


if __name__ == "__main__":
    main()
