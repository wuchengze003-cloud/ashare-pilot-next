"""Tests for the machine-enforced repository language policy."""

from pathlib import Path

from tools.check_language import check_language


def write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_clean_tree_passes(tmp_path: Path) -> None:
    write(tmp_path, "README.md", "# 项目\n")
    write(tmp_path, "tools/demo.py", "VALUE = 1\n")

    assert check_language(tmp_path) == []


def test_chinese_readme_is_allowed(tmp_path: Path) -> None:
    write(tmp_path, "README.md", "# A 股量化研究与模拟仓\n")

    assert check_language(tmp_path) == []


def test_cjk_in_architecture_docs_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "README.md", "# Project\n")
    write(tmp_path, "docs/architecture/README.md", "# 架构\n")

    assert check_language(tmp_path) == [
        "CJK in documentation: docs/architecture/README.md:1"
    ]


def test_cjk_in_python_comment_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "README.md", "# Project\n")
    write(tmp_path, "apps/research/model.py", "# 中文注释\nVALUE = 1\n")

    violations = check_language(tmp_path)
    assert any("CJK in comment" in item for item in violations)


def test_tools_ui_strings_are_allowed(tmp_path: Path) -> None:
    write(tmp_path, "README.md", "# Project\n")
    write(tmp_path, "tools/demo.py", 'print("当前无持仓")\n')

    assert check_language(tmp_path) == []


def test_champion_display_labels_are_allowed(tmp_path: Path) -> None:
    write(tmp_path, "README.md", "# Project\n")
    write(
        tmp_path,
        "apps/research/champion.py",
        'FEATURE_LABELS = {"ret_1d": "1日涨幅"}\nVALUE = 1\n',
    )

    assert check_language(tmp_path) == []


def test_champion_non_allowlisted_cjk_string_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "README.md", "# Project\n")
    write(
        tmp_path,
        "apps/research/champion.py",
        'reason_labels = {"stop": "止损"}\n',
    )

    violations = check_language(tmp_path)
    assert any("CJK in string literal" in item for item in violations)

def test_test_fixture_data_is_allowed(tmp_path: Path) -> None:
    write(tmp_path, "README.md", "# Project\n")
    write(tmp_path, "services/data_gateway/tests/test_data.py", 'NAME = "平安银行"\n')

    assert check_language(tmp_path) == []


def test_non_ui_python_string_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "README.md", "# Project\n")
    write(tmp_path, "apps/research/model.py", 'VALUE = "中文"  # English comment\n')

    violations = check_language(tmp_path)
    assert any("CJK in string literal" in item for item in violations)
