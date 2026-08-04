"""One-command live pilot orchestrator.

Ops role only: invokes public application commands as subprocesses,
serves nothing itself, and contains no financial semantics. May read
the wall clock because Ops is outside the deterministic core.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CN_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")
MARKET_SESSIONS = (
    ("PRE_OPEN", (9, 15), (9, 30)),
    ("OPEN", (9, 30), (11, 30)),
    ("MIDDAY_BREAK", (11, 30), (13, 0)),
    ("OPEN", (13, 0), (15, 0)),
)


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, cwd=str(REPO_ROOT))


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=True, indent=2) + "\n")
    temporary.replace(path)


def _fail(runtime_web: Path, error: str, detail: str) -> None:
    _write_json(
        runtime_web / "refresh-error.json",
        {
            "error": error,
            "detail": detail[-500:],
            "failed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )


def _clear_failure(runtime_web: Path) -> None:
    failure = runtime_web / "refresh-error.json"
    if failure.is_file():
        failure.unlink()


def discover_m1_dataset() -> tuple[Path, Path]:
    """Locate the accepted M1 run: manifest path and dataset directory."""
    candidates = sorted((REPO_ROOT / "runtime").glob("m1-*/runs/*/run-receipt.json"))
    for receipt_path in reversed(candidates):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if receipt.get("status") != "PASS":
            continue
        run_dir = receipt_path.parent
        manifests = sorted((run_dir / "manifests").glob("*.json"))
        if not manifests:
            continue
        manifest_path = manifests[-1]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_dir = run_dir / "datasets" / str(manifest["dataset_id"])
        if (dataset_dir / "daily-bars.json").is_file():
            return manifest_path, dataset_dir
    raise FileNotFoundError("no accepted M1 dataset found under runtime/")


def market_session(now_cn: datetime, calendar_days: set[str]) -> dict:
    calendar_date = now_cn.date().isoformat()
    if calendar_date not in calendar_days:
        return {"calendar_date": calendar_date, "is_open": False, "session_label": "HOLIDAY"}
    minutes = now_cn.hour * 60 + now_cn.minute
    for label, (start_h, start_m), (end_h, end_m) in MARKET_SESSIONS:
        if start_h * 60 + start_m <= minutes < end_h * 60 + end_m:
            return {
                "calendar_date": calendar_date,
                "is_open": label == "OPEN",
                "session_label": label,
            }
    return {"calendar_date": calendar_date, "is_open": False, "session_label": "CLOSED"}


def refresh_market_sidecar(runtime_web: Path) -> None:
    calendar_path = runtime_web / "calendar.json"
    calendar_days: set[str] = set()
    if calendar_path.is_file():
        try:
            calendar_days = set(json.loads(calendar_path.read_text(encoding="utf-8")))
        except ValueError:
            calendar_days = set()
    document = market_session(datetime.now(CN_TZ), calendar_days)
    document["computed_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    _write_json(runtime_web / "market.json", document)


def run_cycle(args, runtime_pilot: Path, runtime_web: Path) -> bool:
    generated_at = datetime.now(UTC)
    generated_text = generated_at.isoformat().replace("+00:00", "Z")
    stamp = generated_at.strftime("%Y%m%dT%H%M%S")
    python = sys.executable

    if args.mode == "live":
        symbols = args.symbols
        window_end = datetime.now(CN_TZ).date()
        window_start = window_end - timedelta(days=args.lookback_days)
        acquire = _run(
            [
                python,
                "-m",
                "ashare_data_gateway.pilot_acquire",
                "--symbols",
                symbols,
                "--window-start",
                window_start.isoformat(),
                "--window-end",
                window_end.isoformat(),
                "--publication-root",
                str(runtime_pilot / "publication"),
                "--generated-at",
                generated_text,
            ]
        )
        if acquire.returncode != 0 or not acquire.stdout.strip():
            try:
                payload = json.loads(acquire.stdout or "{}")
            except ValueError:
                payload = {}
            _fail(
                runtime_web,
                str(payload.get("error", "ACQUIRE_FAILED")),
                str(payload.get("detail", acquire.stderr[-300:])),
            )
            return False
        acquired = json.loads(acquire.stdout)
        manifest_path = Path(str(acquired["manifest_path"]))
        dataset_root = Path(str(acquired["dataset_dir"]))
        is_official = bool(acquired["is_official_vendor"])
        mode = "live-fetch"
    else:
        manifest_path, dataset_root = discover_m1_dataset()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        is_official = manifest.get("source") == "tushare-official"
        mode = "m1-replay"

    research = _run(
        [
            python,
            "-m",
            "ashare_research_app.pilot_research",
            "--dataset-manifest",
            str(manifest_path),
            "--dataset-root",
            str(dataset_root),
            "--repository-root",
            str(REPO_ROOT),
            "--runtime-root",
            str(runtime_pilot),
            "--generated-at",
            generated_text,
            "--top-k",
            str(args.top_k),
            "--per-weight",
            str(args.per_weight),
            "--initial-capital",
            str(args.initial_capital),
        ]
    )
    if research.returncode != 0 or not research.stdout.strip():
        _fail(runtime_web, "RESEARCH_FAILED", research.stderr[-500:])
        return False
    research_result = json.loads(research.stdout)
    as_of = str(research_result["as_of"])

    git_sha = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]).stdout.strip()
    signal_run = _run(
        [
            python,
            "-m",
            "ashare_signal_runner.pilot_run",
            "--contracts-dir",
            str(research_result["contracts_dir"]),
            "--adapter-root",
            str(research_result["adapter_root"]),
            "--dataset-root",
            str(dataset_root),
            "--runs-root",
            str(runtime_pilot / "runs"),
            "--head-path",
            str(runtime_pilot / "current-signal-head.json"),
            "--as-of",
            as_of,
            "--generated-at",
            generated_text,
            "--signal-id",
            f"pilot-signal-{stamp}",
            "--run-id",
            f"pilot-run-{stamp}",
            "--git-sha",
            git_sha,
        ]
    )
    if signal_run.returncode != 0:
        _fail(runtime_web, "SIGNAL_DEGRADED", signal_run.stderr[-500:])

    state_arguments = [
        python,
        "-m",
        "ashare_research_app.web_state",
        "--research-report",
        str(runtime_pilot / "research-report.json"),
        "--dataset-manifest",
        str(manifest_path),
        "--champion",
        str(research_result["champion_path"]),
        "--runs-root",
        str(runtime_pilot / "runs"),
        "--head-path",
        str(runtime_pilot / "current-signal-head.json"),
        "--mode",
        mode,
        "--generated-at",
        generated_text,
        "--out",
        str(runtime_web / "live-state.json"),
    ]
    if is_official:
        state_arguments.append("--is-official-vendor")
    state_result = _run(state_arguments)
    if state_result.returncode != 0:
        _fail(runtime_web, "WEB_STATE_FAILED", state_result.stderr[-500:])
        return False

    report = json.loads((runtime_pilot / "research-report.json").read_text(encoding="utf-8"))
    calendar_days = sorted({str(point["trade_date"]) for point in report["nav_curve"]} | {as_of})
    calendar_path = runtime_web / "calendar.json"
    temporary = calendar_path.with_name(f".{calendar_path.name}.tmp")
    temporary.write_text(json.dumps(calendar_days, ensure_ascii=True) + "\n")
    temporary.replace(calendar_path)
    _clear_failure(runtime_web)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ML live pilot demo")
    parser.add_argument("--mode", choices=["m1-replay", "live"], default="m1-replay")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--refresh-seconds", type=int, default=300)
    parser.add_argument("--runtime-root", default="runtime/pilot")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--per-weight", type=float, default=0.2)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument(
        "--symbols",
        default=(
            "000001.SZ,000333.SZ,000858.SZ,002415.SZ,002594.SZ,300014.SZ,300059.SZ,"
            "300750.SZ,600036.SH,600519.SH,601318.SH,601398.SH,688008.SH,688111.SH,"
            "688981.SH,600466.SH,000732.SZ,000666.SZ,600530.SZ,002564.SZ"
        ),
    )
    parser.add_argument("--lookback-days", type=int, default=240)
    args = parser.parse_args()

    runtime_pilot = REPO_ROOT / args.runtime_root
    runtime_web = runtime_pilot / "web"
    runtime_web.mkdir(parents=True, exist_ok=True)

    print(f"[ops] running initial cycle (mode={args.mode}) ...")
    run_cycle(args, runtime_pilot, runtime_web)
    refresh_market_sidecar(runtime_web)

    server = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "apps/web/server.py"),
            "--static-dir",
            str(REPO_ROOT / "apps/web/static"),
            "--state-file",
            str(runtime_web / "live-state.json"),
            "--market-file",
            str(runtime_web / "market.json"),
            "--error-file",
            str(runtime_web / "refresh-error.json"),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
        cwd=str(REPO_ROOT),
    )
    print(f"[ops] page: http://{args.host}:{args.port}")
    last_cycle = time.monotonic()
    last_market = 0.0
    try:
        while server.poll() is None:
            time.sleep(5)
            now = time.monotonic()
            if now - last_market >= 30:
                refresh_market_sidecar(runtime_web)
                last_market = now
            if args.mode == "live" and now - last_cycle >= args.refresh_seconds:
                print("[ops] refreshing live cycle ...")
                run_cycle(args, runtime_pilot, runtime_web)
                last_cycle = now
    except KeyboardInterrupt:
        pass
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
