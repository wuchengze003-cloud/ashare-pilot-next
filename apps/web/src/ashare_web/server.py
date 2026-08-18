"""Serve exactly one immutable static release."""

from __future__ import annotations

import argparse
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ALLOWED = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/production-signal.json": ("production-signal.json", "application/json"),
    "/simulated-account-state.json": ("simulated-account-state.json", "application/json"),
    "/release-manifest.json": ("release-manifest.json", "application/json"),
}


def make_handler(release_dir: Path):
    release_dir = Path(release_dir).resolve(strict=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            route = self.path.split("?", 1)[0]
            item = ALLOWED.get(route)
            if item is None:
                self.send_error(404)
                return
            name, content_type = item
            path = release_dir / name
            if path.is_symlink() or not path.is_file():
                self.send_error(404)
                return
            content = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve one immutable Web release")
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.release_dir))
    print(f"read-only Web release listening on http://{args.host}:{server.server_address[1]}")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
