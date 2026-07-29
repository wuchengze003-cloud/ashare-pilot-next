from pathlib import Path

from tools.check_boundaries import check_repository


def write(root: Path, relative: str, content: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_clean_tree_passes(tmp_path: Path) -> None:
    write(tmp_path, "packages/quant_core/src/ashare_quant_core/model.py", "VALUE = 1\n")

    assert check_repository(tmp_path) == []


def test_legacy_dependency_is_rejected(tmp_path: Path) -> None:
    legacy_name = "ashare-pilot" + "-legacy"
    write(tmp_path, "config/source.json", f'{{"dependency": "{legacy_name}"}}\n')

    assert check_repository(tmp_path) == ["Legacy dependency: config/source.json"]


def test_generated_report_tree_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "reports/old-result.json", "{}\n")

    assert check_repository(tmp_path) == [
        "forbidden tracked tree: reports/old-result.json"
    ]


def test_market_data_file_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "fixtures/prices.csv", "date,close\n2026-01-02,10\n")

    assert check_repository(tmp_path) == [
        "forbidden artifact type: fixtures/prices.csv"
    ]


def test_symlink_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "outside.txt", "content\n")
    (tmp_path / "linked.txt").symlink_to(tmp_path / "outside.txt")

    assert check_repository(tmp_path) == ["tracked symlink: linked.txt"]


def test_quant_core_cannot_import_signal_runner(tmp_path: Path) -> None:
    write(
        tmp_path,
        "packages/quant_core/src/ashare_quant_core/model.py",
        "import ashare_signal_runner\n",
    )

    assert check_repository(tmp_path) == [
        "forbidden import in packages/quant_core/src/ashare_quant_core/model.py: "
        "ashare_signal_runner"
    ]


def test_wall_clock_in_core_is_rejected(tmp_path: Path) -> None:
    write(
        tmp_path,
        "packages/quant_core/src/ashare_quant_core/model.py",
        "from datetime import datetime\nVALUE = datetime.now()\n",
    )

    assert check_repository(tmp_path) == [
        "wall-clock token 'datetime.now(' in "
        "packages/quant_core/src/ashare_quant_core/model.py"
    ]
