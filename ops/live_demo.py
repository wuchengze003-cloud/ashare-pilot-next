"""One-command pilot orchestrator with separated lifecycles.

Bootstrap (startup only, synthetic/m1-replay modes): create the demo
dataset and, when no active champion exists, run one explicit Research
promotion + activation. Live mode never trains or promotes.

Every refresh cycle is inference-only: acquire/read dataset, frozen
inference scoring, production signal from the active frozen champion,
and web state with binding verification. Failures keep the last known
good page and record an explicit DEGRADED/STALE cycle status; the
failure marker is cleared only when a cycle fully succeeds.
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


def _read_json(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


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


def _last_good_from(previous: dict | None) -> dict:
    if previous is not None and previous.get("cycle_status") == "CURRENT":
        return {
            "last_good_cycle_id": previous.get("cycle_id"),
            "last_good_signal_id": previous.get("last_good_signal_id"),
            "last_good_as_of": previous.get("last_good_as_of"),
        }
    return {
        "last_good_cycle_id": None,
        "last_good_signal_id": None,
        "last_good_as_of": None,
    }


def fail_cycle(
    runtime_web: Path,
    *,
    cycle_id: str,
    generated_text: str,
    error: str,
    detail: str = "",
) -> None:
    status_path = runtime_web / "cycle-status.json"
    previous = _read_json(status_path)
    _write_json(
        status_path,
        {
            "status_id": "cycle-status/v1",
            "cycle_id": cycle_id,
            "cycle_status": "DEGRADED",
            "errors": [error] + ([detail[-300:]] if detail else []),
            **_last_good_from(previous),
            "updated_at": generated_text,
        },
    )


def run_research_activate(
    args: argparse.Namespace,
    runtime_pilot: Path,
    dataset: dict,
    generated_text: str,
) -> bool:
    """Explicit model-change action; only called from bootstrap."""
    result = _run(
        [
            sys.executable,
            "-m",
            "ashare_research_app.pilot_research",
            "--dataset-manifest",
            str(dataset["manifest"]),
            "--dataset-root",
            str(dataset["root"]),
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
            "--activate",
        ]
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def bootstrap(
    args: argparse.Namespace,
    runtime_pilot: Path,
    runtime_web: Path,
    generated_text: str,
) -> dict | None:
    pointer = runtime_pilot / "active-champion.json"
    if args.mode == "synthetic":
        synthetic = _run(
            [
                sys.executable,
                "-m",
                "ashare_research_app.synthetic_demo",
                "--out-dir",
                str(runtime_pilot / "synthetic"),
            ]
        )
        if synthetic.returncode != 0 or not synthetic.stdout.strip():
            fail_cycle(
                runtime_web,
                cycle_id="bootstrap",
                generated_text=generated_text,
                error="SYNTHETIC_FAILED",
                detail=synthetic.stderr,
            )
            raise SystemExit(1)
        generated = json.loads(synthetic.stdout)
        dataset = {
            "manifest": Path(generated["manifest_path"]),
            "root": Path(generated["dataset_dir"]),
            "as_of": generated["as_of"],
            "official": False,
            "web_mode": "synthetic",
        }
        if not pointer.is_file() and not run_research_activate(
            args, runtime_pilot, dataset, generated_text
        ):
            fail_cycle(
                runtime_web,
                cycle_id="bootstrap",
                generated_text=generated_text,
                error="PROMOTION_FAILED",
            )
            raise SystemExit(1)
        return dataset
    if args.mode == "m1-replay":
        manifest_path, dataset_dir = discover_m1_dataset()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset = {
            "manifest": manifest_path,
            "root": dataset_dir,
            "as_of": str(manifest["as_of"]),
            "official": manifest.get("source") == "tushare-official",
            "web_mode": "m1-replay",
        }
        if not pointer.is_file() and not run_research_activate(
            args, runtime_pilot, dataset, generated_text
        ):
            fail_cycle(
                runtime_web,
                cycle_id="bootstrap",
                generated_text=generated_text,
                error="PROMOTION_FAILED",
            )
            raise SystemExit(1)
        return dataset
    return None  # live mode: dataset acquired per cycle; never train here


def run_cycle(
    args: argparse.Namespace,
    runtime_pilot: Path,
    runtime_web: Path,
    dataset: dict | None,
    cycle_id: str,
    generated_text: str,
    stamp: str,
) -> bool:
    python = sys.executable

    if dataset is None:
        window_end = datetime.now(CN_TZ).date()
        window_start = window_end - timedelta(days=args.lookback_days)
        acquire = _run(
            [
                python,
                "-m",
                "ashare_data_gateway.pilot_acquire",
                "--symbols",
                args.symbols,
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
            fail_cycle(
                runtime_web,
                cycle_id=cycle_id,
                generated_text=generated_text,
                error=str(payload.get("error", "DATASET_FAILED")),
                detail=str(payload.get("detail", acquire.stderr)),
            )
            return False
        acquired = json.loads(acquire.stdout)
        manifest = json.loads(
            Path(str(acquired["manifest_path"])).read_text(encoding="utf-8")
        )
        dataset = {
            "manifest": Path(str(acquired["manifest_path"])),
            "root": Path(str(acquired["dataset_dir"])),
            "as_of": str(manifest["as_of"]),
            "official": bool(acquired["is_official_vendor"]),
            "web_mode": "live-fetch",
        }

    inference = _run(
        [
            python,
            "-m",
            "ashare_research_app.frozen_inference",
            "--runtime-root",
            str(runtime_pilot),
            "--dataset-manifest",
            str(dataset["manifest"]),
            "--dataset-root",
            str(dataset["root"]),
            "--top-k",
            str(args.top_k),
            "--out",
            str(runtime_web / "inference-report.json"),
        ]
    )
    if inference.returncode == 4:
        fail_cycle(
            runtime_web,
            cycle_id=cycle_id,
            generated_text=generated_text,
            error="NO_ACTIVE_CHAMPION",
        )
        return False
    if inference.returncode != 0:
        fail_cycle(
            runtime_web,
            cycle_id=cycle_id,
            generated_text=generated_text,
            error="INFERENCE_FAILED",
            detail=inference.stderr,
        )
        return False

    git_sha = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]).stdout.strip()
    signal_run = _run(
        [
            python,
            "-m",
            "ashare_signal_runner.pilot_run",
            "--runtime-root",
            str(runtime_pilot),
            "--dataset-root",
            str(dataset["root"]),
            "--runs-root",
            str(runtime_pilot / "runs"),
            "--head-path",
            str(runtime_pilot / "current-signal-head.json"),
            "--as-of",
            str(dataset["as_of"]),
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
    if signal_run.returncode == 4:
        fail_cycle(
            runtime_web,
            cycle_id=cycle_id,
            generated_text=generated_text,
            error="NO_ACTIVE_CHAMPION",
        )
        return False
    if signal_run.returncode != 0 or not signal_run.stdout.strip():
        fail_cycle(
            runtime_web,
            cycle_id=cycle_id,
            generated_text=generated_text,
            error="SIGNAL_FAILED",
            detail=signal_run.stderr,
        )
        return False

    pointer = _read_json(runtime_pilot / "active-champion.json")
    champion_path = (
        runtime_pilot / "champions" / str(pointer["champion_id"]) / "champion.json"
    )
    state_arguments = [
        python,
        "-m",
        "ashare_research_app.web_state",
        "--research-report",
        str(runtime_pilot / "research-report.json"),
        "--dataset-manifest",
        str(dataset["manifest"]),
        "--dataset-root",
        str(dataset["root"]),
        "--champion",
        str(champion_path),
        "--runs-root",
        str(runtime_pilot / "runs"),
        "--head-path",
        str(runtime_pilot / "current-signal-head.json"),
        "--mode",
        str(dataset["web_mode"]),
        "--generated-at",
        generated_text,
        "--cycle-id",
        cycle_id,
        "--active-pointer",
        str(runtime_pilot / "active-champion.json"),
        "--cycle-status-out",
        str(runtime_web / "cycle-status.json"),
        "--inference-report",
        str(runtime_web / "inference-report.json"),
        "--out",
        str(runtime_web / "live-state.json"),
    ]
    if dataset["official"]:
        state_arguments.append("--is-official-vendor")
    state_result = _run(state_arguments)
    if state_result.returncode != 0:
        if state_result.returncode != 3:
            fail_cycle(
                runtime_web,
                cycle_id=cycle_id,
                generated_text=generated_text,
                error="WEB_STATE_FAILED",
                detail=state_result.stderr,
            )
        return False

    failure = runtime_web / "refresh-error.json"
    if failure.is_file():
        failure.unlink()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ML live pilot demo")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "m1-replay", "live"],
        default="synthetic",
    )
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
            "688981.SH,600466.SH,000732.SZ,000666.SZ,600530.SH,002564.SZ"
        ),
    )
    parser.add_argument("--lookback-days", type=int, default=240)
    args = parser.parse_args()

    runtime_pilot = REPO_ROOT / args.runtime_root
    runtime_web = runtime_pilot / "web"
    runtime_web.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(UTC)
    generated_text = generated_at.isoformat().replace("+00:00", "Z")
    stamp = generated_at.strftime("%Y%m%dT%H%M%S")

    print(f"[ops] bootstrap (mode={args.mode}) ...")
    dataset = bootstrap(args, runtime_pilot, runtime_web, generated_text)
    if dataset is not None:
        report = _read_json(runtime_pilot / "research-report.json")
        if report is not None:
            calendar_days = sorted(
                {str(point["trade_date"]) for point in report["nav_curve"]}
                | {str(dataset["as_of"])}
            )
            (runtime_web / "calendar.json").write_text(
                json.dumps(calendar_days, ensure_ascii=True) + "\n"
            )

    cycle_id = f"live-{stamp}"
    print(f"[ops] cycle {cycle_id} ...")
    run_cycle(args, runtime_pilot, runtime_web, dataset, cycle_id, generated_text, stamp)
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
            "--cycle-status-file",
            str(runtime_web / "cycle-status.json"),
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
            if now - last_cycle >= args.refresh_seconds:
                cycle_generated = datetime.now(UTC)
                cycle_text = cycle_generated.isoformat().replace("+00:00", "Z")
                cycle_stamp = cycle_generated.strftime("%Y%m%dT%H%M%S")
                print(f"[ops] cycle live-{cycle_stamp} ...")
                run_cycle(
                    args,
                    runtime_pilot,
                    runtime_web,
                    dataset,
                    f"live-{cycle_stamp}",
                    cycle_text,
                    cycle_stamp,
                )
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
