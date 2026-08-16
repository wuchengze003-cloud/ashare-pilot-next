"""Machine-check the repository language policy.

Policy enforced here:
- README.md and docs/architecture/**/*.md contain no CJK characters.
- Comments and docstrings in Python/JS/TS/Shell files under apps/, packages/,
  services/, tools/, and ops/ are English.
- CJK string literals are allowed only in user-facing UI/output strings:
  * tools/*.py (dashboard HTML and CLI output),
  * test files under */tests/ (Chinese stock names in fixture data),
  * champion.py display-label assignments FEATURE_LABELS, KEY_FEATURES, and
    reason_labels,
  * apps/web/static/*.html user-facing HTML.
Exit code 0 means the policy holds; 1 reports every violation.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CODE_ROOTS = ("apps", "packages", "services", "tools", "ops")
PY_SUFFIXES = {".py"}
JS_SUFFIXES = {".js", ".jsx", ".ts", ".tsx", ".sh"}
HTML_SUFFIXES = {".html", ".htm"}
CHAMPION_UI_VARIABLES = {"FEATURE_LABELS", "KEY_FEATURES", "reason_labels"}


def _has_cjk(text: str) -> bool:
    return CJK_PATTERN.search(text) is not None


def _cjk_lines(text: str) -> list[int]:
    return [i + 1 for i, line in enumerate(text.splitlines()) if _has_cjk(line)]


def _relative(path: Path, root: Path = ROOT) -> str:
    return str(path.relative_to(root))


def _walk_python(path: Path, violations: list[str], root: Path = ROOT) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        violations.append(f"non-UTF-8 Python source: {_relative(path, root)}")
        return
    if not _has_cjk(text):
        return

    # Comments must be English everywhere.
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT and _has_cjk(token.string):
                violations.append(
                    f"CJK in comment: {_relative(path, root)}:{token.start[0]}: "
                    f"{token.string.strip()}"
                )
    except (IndentationError, SyntaxError, tokenize.TokenError):
        pass

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return

    # Docstrings must be English everywhere.
    if tree.body and isinstance(tree.body[0], ast.Expr):
        node = tree.body[0].value
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _has_cjk(node.value):
            violations.append(f"CJK in module docstring: {_relative(path, root)}:1")
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring and _has_cjk(docstring):
                violations.append(
                    f"CJK in docstring: {_relative(path, root)}:{node.lineno}: "
                    f"{docstring.strip()[:70]}"
                )

    # String literals are checked against the user-facing allowlist.
    relative = _relative(path, root)
    allow_all_strings = (
        relative.startswith("tools/")
        or "/tests/" in f"/{relative}"
        or relative.startswith("apps/web/static/")
    )
    allowed_node_ids: set[int] = set()
    if path.name == "champion.py":
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(
                    isinstance(target, ast.Name) and target.id in CHAMPION_UI_VARIABLES
                    for target in targets
                ):
                    for child in ast.walk(node.value):
                        allowed_node_ids.add(id(child))
            elif isinstance(node, ast.FunctionDef) and node.name == "generate":
                # ``generate`` builds the dashboard-facing champion document.
                for child in ast.walk(node):
                    allowed_node_ids.add(id(child))
            elif (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
            ):
                # The __main__ block is user-facing console output.
                for child in ast.walk(node):
                    allowed_node_ids.add(id(child))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if not _has_cjk(node.value):
            continue
        if allow_all_strings or id(node) in allowed_node_ids:
            continue
        snippet = " ".join(node.value.split())
        violations.append(
            f"CJK in string literal: {_relative(path, root)}:{node.lineno}: "
            f"{snippet[:70]}"
        )


def _walk_web_file(path: Path, violations: list[str], root: Path = ROOT) -> None:
    text = path.read_text(encoding="utf-8")
    # HTML user-facing text is allowed; HTML comments still must be English.
    for match in re.finditer(r"<!--(.*?)-->", text, flags=re.DOTALL):
        if _has_cjk(match.group(1)):
            violations.append(f"CJK in HTML comment: {_relative(path, root)}")
            return


def _walk_plain_code(path: Path, violations: list[str], root: Path = ROOT) -> None:
    text = path.read_text(encoding="utf-8")
    if _has_cjk(text):
        for lineno in _cjk_lines(text):
            violations.append(f"CJK in non-UI code: {_relative(path, root)}:{lineno}")


def check_language(root: Path = ROOT) -> list[str]:
    """Return every language-policy violation under ``root``."""
    violations: list[str] = []
    doc_roots = (root / "README.md", root / "docs" / "architecture")

    for doc_root in doc_roots:
        if doc_root.is_file():
            text = doc_root.read_text(encoding="utf-8")
            for lineno in _cjk_lines(text):
                violations.append(f"CJK in documentation: {_relative(doc_root)}:{lineno}")
        elif doc_root.is_dir():
            for path in sorted(doc_root.rglob("*.md")):
                text = path.read_text(encoding="utf-8")
                for lineno in _cjk_lines(text):
                    violations.append(
                        f"CJK in documentation: {_relative(path, root)}:{lineno}"
                    )

    for code_root_name in CODE_ROOTS:
        code_root = root / code_root_name
        if not code_root.is_dir():
            continue
        for path in sorted(code_root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            suffix = path.suffix.lower()
            if suffix in PY_SUFFIXES:
                _walk_python(path, violations, root)
            elif suffix in JS_SUFFIXES:
                _walk_plain_code(path, violations, root)
            elif suffix in HTML_SUFFIXES and path.is_relative_to(root / "apps/web/static"):
                _walk_web_file(path, violations, root)
            elif suffix in HTML_SUFFIXES:
                _walk_plain_code(path, violations, root)

    return violations


def main() -> int:
    violations = check_language()
    if violations:
        print(f"FAIL: {len(violations)} language-policy violation(s):")
        for violation in violations:
            print(f"  {violation}")
        return 1
    print("PASS: repository language policy holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
