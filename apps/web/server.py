"""Read-only web server for the pilot page.

Serves the static page and the latest versioned contract documents.
Performs no financial calculation: every number is rendered from the
web-state contract produced by Research. Optional market session and
refresh-error sidecars are plain JSON computed from explicit inputs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _read_json(path: Path) -> bytes | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        content = path.read_bytes()
        json.loads(content)
    except (OSError, ValueError):
        return None
    return content


def make_handler(static_dir: Path, state_file: Path, market_file: Path, error_file: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, content: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            route = self.path.split("?", 1)[0]
            if route in {"/", "/index.html"}:
                page = (static_dir / "index.html").read_bytes()
                self._send(200, page, "text/html; charset=utf-8")
            elif route == "/api/web-state":
                content = _read_json(state_file)
                if content is None:
                    self._send(404, b'{"error":"state unavailable"}', "application/json")
                else:
                    self._send(200, content, "application/json")
            elif route == "/api/market":
                content = _read_json(market_file)
                self._send(
                    200,
                    content if content is not None else b"{}",
                    "application/json",
                )
            elif route == "/api/refresh-error":
                content = _read_json(error_file)
                self._send(
                    200,
                    content if content is not None else b"{}",
                    "application/json",
                )
            else:
                self._send(404, b'{"error":"not found"}', "application/json")

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return Handler


def serve(*, static_dir: Path, state_file: Path, market_file: Path, error_file: Path,
          host: str, port: int) -> None:
    handler = make_handler(static_dir, state_file, market_file, error_file)
    server = ThreadingHTTPServer((host, port), handler)
    actual_port = server.server_address[1]
    print(f"pilot web listening on http://{host}:{actual_port}")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Pilot read-only web server")
    parser.add_argument("--static-dir", default=str(Path(__file__).parent / "static"))
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--market-file", required=True)
    parser.add_argument("--error-file", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(
        static_dir=Path(args.static_dir),
        state_file=Path(args.state_file),
        market_file=Path(args.market_file),
        error_file=Path(args.error_file),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
