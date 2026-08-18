"""Serve exactly one immutable static release."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .release import canonical_json_bytes

ALLOWED = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/production-signal.json": ("production-signal.json", "application/json"),
    "/simulated-account-state.json": ("simulated-account-state.json", "application/json"),
    "/release-manifest.json": ("release-manifest.json", "application/json"),
}


def validate_release_dir(release_dir: Path) -> Path:
    """Fail closed before serving a release whose published bytes changed."""
    release_dir = Path(release_dir)
    if release_dir.is_symlink() or not release_dir.is_dir():
        raise ValueError("release directory must be a regular directory")
    resolved = release_dir.resolve(strict=True)
    manifest_path = resolved / "release-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("release manifest must be a regular file")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("release manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest_bytes != canonical_json_bytes(manifest):
        raise ValueError("release manifest is not canonical")
    if (
        manifest.get("contract_id") != "static-web-release"
        or manifest.get("schema_version") != "1.0.0"
    ):
        raise ValueError("release manifest contract is unsupported")

    expected_names = {item[0] for item in ALLOWED.values()} - {"release-manifest.json"}
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("release manifest files must be an array")
    declared_names = [str(entry.get("path")) for entry in entries if isinstance(entry, dict)]
    if len(declared_names) != len(entries) or set(declared_names) != expected_names:
        raise ValueError("release manifest file set mismatch")
    if len(declared_names) != len(set(declared_names)):
        raise ValueError("release manifest contains duplicate files")
    for entry in entries:
        name = str(entry["path"])
        path = resolved / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"release artifact must be a regular file: {name}")
        content = path.read_bytes()
        if len(content) != entry.get("file_size_bytes"):
            raise ValueError(f"release artifact size mismatch: {name}")
        if hashlib.sha256(content).hexdigest() != entry.get("sha256"):
            raise ValueError(f"release artifact hash mismatch: {name}")
    if manifest.get("signal_sha256") != hashlib.sha256(
        (resolved / "production-signal.json").read_bytes()
    ).hexdigest():
        raise ValueError("release signal hash mismatch")
    if manifest.get("account_state_sha256") != hashlib.sha256(
        (resolved / "simulated-account-state.json").read_bytes()
    ).hexdigest():
        raise ValueError("release account-state hash mismatch")
    return resolved


def make_handler(release_dir: Path):
    release_dir = validate_release_dir(release_dir)

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
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; object-src 'none'")
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
