"""Fail closed when tracked-source or dependency boundaries are violated."""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".turbo",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "htmlcov",
    "node_modules",
}
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
CODE_SUFFIXES = {
    ".js",
    ".json",
    ".jsx",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
TEXT_SUFFIXES = CODE_SUFFIXES | {".md", ".txt"}
MAX_TRACKED_FILE_BYTES = 1_000_000

INTERNAL_IMPORTS = {
    "ashare_data_gateway",
    "ashare_quant_core",
    "ashare_research_app",
    "ashare_signal_runner",
}
INTERNAL_DISTRIBUTIONS = {
    "ashare-data-gateway",
    "ashare-quant-core",
    "ashare-research-app",
    "ashare-signal-runner",
}
ALLOWED_INTERNAL_IMPORTS = {
    "packages/quant_core": {"ashare_quant_core"},
    "services/data_gateway": {"ashare_data_gateway"},
    "apps/research": {"ashare_quant_core", "ashare_research_app"},
    "apps/signal_runner": {"ashare_quant_core", "ashare_signal_runner"},
    "apps/web": set(),
    "ops": set(),
    "tools": set(),
}
ALLOWED_INTERNAL_DISTRIBUTIONS = {
    ".": set(),
    "packages/quant_core": set(),
    "services/data_gateway": set(),
    "apps/research": {"ashare-quant-core"},
    "apps/signal_runner": {"ashare-quant-core"},
}
CURRENT_TIME_CALLS = {
    "arrow.now",
    "arrow.utcnow",
    "datetime.date.today",
    "datetime.datetime.now",
    "datetime.datetime.today",
    "datetime.datetime.utcnow",
    "pandas.Timestamp.now",
    "pandas.Timestamp.today",
    "pendulum.now",
    "pendulum.today",
    "time.time",
}
CRITICAL_TIME_AREAS = {
    "packages/quant_core",
    "services/data_gateway",
    "apps/research",
    "apps/signal_runner",
}


def _walk_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if (not path.is_file() and not path.is_symlink()) or any(
            part in IGNORED_PARTS for part in path.parts
        ):
            continue
        files.append(path)
    return files


def _git_candidate_files(root: Path) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [root / relative for relative in result.stdout.split("\0") if relative]


def source_files(root: Path = ROOT, *, use_git: bool | None = None) -> list[Path]:
    """Return tracked, staged, and unignored new files; tests may use a plain tree."""
    if use_git is None:
        use_git = (root / ".git").exists()
    files = _git_candidate_files(root) if use_git else _walk_files(root)
    return sorted(set(files))


def _read_text(path: Path, *, relative_text: str, violations: list[str]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        violations.append(f"non-UTF-8 source file: {relative_text}")
        return None


def _python_tree(
    path: Path,
    *,
    relative_text: str,
    violations: list[str],
) -> ast.AST | None:
    content = _read_text(path, relative_text=relative_text, violations=violations)
    if content is None:
        return None
    try:
        return ast.parse(content, filename=str(path))
    except SyntaxError as exc:
        violations.append(f"invalid Python syntax in {relative_text}: {exc.msg}")
        return None


def _area_for(relative_text: str) -> str | None:
    return next(
        (
            area
            for area in ALLOWED_INTERNAL_IMPORTS
            if relative_text == area or relative_text.startswith(f"{area}/")
        ),
        None,
    )


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _resolve_alias(name: str, aliases: Mapping[str, str]) -> str:
    first, separator, rest = name.partition(".")
    resolved = aliases.get(first, first)
    return f"{resolved}.{rest}" if separator else resolved


def _dynamic_internal_imports(tree: ast.AST) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function_name = _dotted_name(node.func)
        if function_name not in {"__import__", "importlib.import_module"}:
            continue
        module = node.args[0]
        if isinstance(module, ast.Constant) and isinstance(module.value, str):
            root = module.value.split(".", 1)[0]
            if root in INTERNAL_IMPORTS:
                findings.append((node.lineno, root))
    return findings


def _current_time_calls(tree: ast.AST) -> list[tuple[int, str]]:
    aliases = _import_aliases(tree)
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if name is None:
            continue
        resolved = _resolve_alias(name, aliases)
        if resolved in CURRENT_TIME_CALLS:
            findings.append((node.lineno, resolved))
            continue
        if resolved in {"numpy.datetime64", "pandas.Timestamp"} and node.args:
            value = node.args[0]
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and value.value.lower() in {"now", "today"}
            ):
                findings.append((node.lineno, f'{resolved}("{value.value.lower()}")'))
    return findings


def _normalized_distribution(requirement: str) -> str | None:
    match = re.match(r"^\s*([A-Za-z0-9_.-]+)", requirement)
    return match.group(1).lower().replace("_", "-") if match else None


def _dependency_strings(document: Mapping[str, Any]) -> Iterable[str]:
    project = document.get("project")
    if isinstance(project, Mapping):
        dependencies = project.get("dependencies", [])
        if isinstance(dependencies, list):
            yield from (item for item in dependencies if isinstance(item, str))
        optional = project.get("optional-dependencies", {})
        if isinstance(optional, Mapping):
            for items in optional.values():
                if isinstance(items, list):
                    yield from (item for item in items if isinstance(item, str))
    groups = document.get("dependency-groups", {})
    if isinstance(groups, Mapping):
        for items in groups.values():
            if isinstance(items, list):
                yield from (item for item in items if isinstance(item, str))


def _audit_pyproject(
    path: Path,
    *,
    root: Path,
    relative_text: str,
    violations: list[str],
) -> None:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        violations.append(f"invalid pyproject in {relative_text}: {exc}")
        return

    area = "." if path == root / "pyproject.toml" else path.parent.relative_to(root).as_posix()
    allowed = ALLOWED_INTERNAL_DISTRIBUTIONS.get(area)
    if allowed is None:
        violations.append(f"unregistered Python project area: {area}")
        allowed = set()
    for requirement in _dependency_strings(document):
        distribution = _normalized_distribution(requirement)
        if distribution in INTERNAL_DISTRIBUTIONS and distribution not in allowed:
            violations.append(
                f"forbidden project dependency in {relative_text}: {distribution}"
            )
        lowered = requirement.lower()
        if any(token in lowered for token in ("file://", "git+file:", "../")):
            violations.append(f"local project dependency in {relative_text}: {requirement}")

    uv = document.get("tool", {}).get("uv", {})
    sources = uv.get("sources", {}) if isinstance(uv, Mapping) else {}
    if isinstance(sources, Mapping):
        for distribution, source in sources.items():
            normalized = str(distribution).lower().replace("_", "-")
            if not isinstance(source, Mapping):
                continue
            if source.get("workspace") is True:
                if normalized not in INTERNAL_DISTRIBUTIONS:
                    violations.append(
                        f"unknown workspace dependency in {relative_text}: {distribution}"
                    )
                continue
            if any(key in source for key in ("path", "git")):
                violations.append(
                    f"non-workspace local or git source in {relative_text}: {distribution}"
                )


def _audit_lockfile(path: Path, *, root: Path, violations: list[str]) -> None:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        workspace = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        violations.append(f"invalid uv.lock: {exc}")
        return
    members = set(workspace.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", []))
    for package in document.get("package", []):
        source = package.get("source", {})
        if not isinstance(source, Mapping):
            continue
        editable = source.get("editable")
        if editable is not None and editable not in members:
            violations.append(f"unregistered editable source in uv.lock: {editable}")
        git_source = source.get("git")
        legacy_name = "ashare-pilot" + "-legacy"
        if isinstance(git_source, str) and (
            git_source.startswith("file:") or legacy_name in git_source
        ):
            violations.append(f"forbidden git source in uv.lock: {git_source}")


def _gitlink_violations(root: Path) -> list[str]:
    if not (root / ".git").exists():
        return []
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--stage", "-z"],
        check=True,
        capture_output=True,
        text=True,
    )
    violations: list[str] = []
    for record in result.stdout.split("\0"):
        if not record:
            continue
        metadata, path = record.split("\t", 1)
        mode = metadata.split(" ", 1)[0]
        if mode == "160000":
            violations.append(f"git submodule is forbidden: {path}")
    return violations


def check_repository(
    root: Path = ROOT,
    *,
    use_git: bool | None = None,
) -> list[str]:
    violations = _gitlink_violations(root) if use_git is not False else []
    files = source_files(root, use_git=use_git)
    for path in files:
        relative = path.relative_to(root)
        relative_text = relative.as_posix()

        if relative_text == ".gitmodules":
            violations.append("Git submodules are forbidden: .gitmodules")
        if path.is_symlink():
            violations.append(f"tracked symlink: {relative_text}")
            continue
        if not path.is_file():
            continue
        if any(part in FORBIDDEN_TREE_PARTS for part in relative.parts):
            violations.append(f"forbidden tracked tree: {relative_text}")
        if path.suffix in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden artifact type: {relative_text}")
        if path.stat().st_size > MAX_TRACKED_FILE_BYTES:
            violations.append(f"oversized tracked file: {relative_text}")

        content: str | None = None
        if path.suffix in TEXT_SUFFIXES:
            content = _read_text(
                path,
                relative_text=relative_text,
                violations=violations,
            )
            if content is not None:
                personal_prefixes = ("/" + "Users/", "/" + "home/", "/" + "root/")
                if any(prefix in content for prefix in personal_prefixes) or re.search(
                    r"(?i)\b[A-Z]:(?:\\+|/+)Users(?:\\+|/+)",
                    content,
                ):
                    violations.append(f"personal absolute path: {relative_text}")
                legacy_name = "ashare-pilot" + "-legacy"
                if legacy_name in content:
                    violations.append(f"Legacy dependency: {relative_text}")

        if path.name == "pyproject.toml":
            _audit_pyproject(
                path,
                root=root,
                relative_text=relative_text,
                violations=violations,
            )
        if relative_text == "uv.lock":
            _audit_lockfile(path, root=root, violations=violations)

        if path.suffix != ".py":
            continue
        tree = _python_tree(
            path,
            relative_text=relative_text,
            violations=violations,
        )
        if tree is None:
            continue
        area = _area_for(relative_text)
        if area is None:
            violations.append(f"unregistered Python source area: {relative_text}")
            continue
        illegal = sorted(
            (_imported_roots(tree) & INTERNAL_IMPORTS) - ALLOWED_INTERNAL_IMPORTS[area]
        )
        if illegal:
            violations.append(
                f"forbidden import in {relative_text}: {', '.join(illegal)}"
            )
        for line, module in _dynamic_internal_imports(tree):
            if module not in ALLOWED_INTERNAL_IMPORTS[area]:
                violations.append(
                    f"forbidden dynamic import in {relative_text}:{line}: {module}"
                )
        if area in CRITICAL_TIME_AREAS:
            for line, call in _current_time_calls(tree):
                violations.append(
                    f"wall-clock call in {relative_text}:{line}: {call}"
                )

    return sorted(set(violations))


def main() -> None:
    files = source_files()
    violations = check_repository()
    if violations:
        raise SystemExit("\n".join(violations))
    print(f"PASS: checked {len(files)} Git candidate files and dependency boundaries")


if __name__ == "__main__":
    main()
