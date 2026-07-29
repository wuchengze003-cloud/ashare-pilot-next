"""Fail closed when source-tree or Python dependency boundaries are violated."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
FORBIDDEN_TREE_PARTS = {"runtime", "artifacts", "reports"}
FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".csv",
    ".db",
    ".feather",
    ".gz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".sqlite",
    ".sqlite3",
    ".tsv",
    ".xls",
    ".xlsx",
    ".zip",
}
CODE_SUFFIXES = {".py", ".toml", ".json", ".yaml", ".yml", ".sh", ".ts", ".tsx", ".js", ".jsx"}
WALL_CLOCK_TOKENS = ("datetime.now(", "date.today(", "time.time(")
MAX_TRACKED_FILE_BYTES = 1_000_000

AREA_RULES = {
    "packages/quant_core": {
        "ashare_research_app",
        "ashare_signal_runner",
        "ashare_data_gateway",
    },
    "apps/research": {
        "ashare_signal_runner",
        "ashare_data_gateway",
    },
    "apps/signal_runner": {
        "ashare_research_app",
        "ashare_data_gateway",
    },
    "services/data_gateway": {
        "ashare_quant_core",
        "ashare_research_app",
        "ashare_signal_runner",
    },
}


def source_files(root: Path = ROOT) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if (not path.is_file() and not path.is_symlink()) or any(
            part in IGNORED_PARTS for part in path.parts
        ):
            continue
        files.append(path)
    return files


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def check_repository(root: Path = ROOT) -> list[str]:
    violations: list[str] = []
    for path in source_files(root):
        relative = path.relative_to(root)
        relative_text = relative.as_posix()

        if path.is_symlink():
            violations.append(f"tracked symlink: {relative_text}")
            continue
        if any(part in FORBIDDEN_TREE_PARTS for part in relative.parts):
            violations.append(f"forbidden tracked tree: {relative_text}")
        if path.suffix in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden artifact type: {relative_text}")
        if path.stat().st_size > MAX_TRACKED_FILE_BYTES:
            violations.append(f"oversized tracked file: {relative_text}")

        if path.suffix in CODE_SUFFIXES:
            content = path.read_text(encoding="utf-8")
            personal_path_prefix = "/" + "Users/"
            if personal_path_prefix in content:
                violations.append(f"personal absolute path: {relative_text}")
            legacy_name = "ashare-pilot" + "-legacy"
            if legacy_name in content:
                violations.append(f"Legacy dependency: {relative_text}")

        if path.suffix == ".py":
            try:
                imports = imported_roots(path)
            except SyntaxError as exc:
                violations.append(f"invalid Python syntax in {relative_text}: {exc.msg}")
                imports = set()
            for area, forbidden in AREA_RULES.items():
                if relative_text.startswith(f"{area}/"):
                    overlap = sorted(imports & forbidden)
                    if overlap:
                        violations.append(
                            f"forbidden import in {relative_text}: {', '.join(overlap)}"
                        )

        if relative_text.startswith(("packages/quant_core/", "apps/signal_runner/")):
            content = path.read_text(encoding="utf-8")
            for token in WALL_CLOCK_TOKENS:
                if token in content:
                    violations.append(f"wall-clock token {token!r} in {relative_text}")

        if relative_text.startswith("apps/web/") and path.suffix in {".ts", ".tsx", ".js", ".jsx"}:
            lowered = path.read_text(encoding="utf-8").lower()
            for token in ("backtest", "optimizer", "targetweight", "position sizing"):
                if token in lowered:
                    violations.append(f"financial logic token {token!r} in {relative_text}")

    return sorted(set(violations))


def main() -> None:
    violations = check_repository()
    if violations:
        raise SystemExit("\n".join(violations))
    print(f"PASS: checked {len(source_files())} files and Python dependency boundaries")


if __name__ == "__main__":
    main()
