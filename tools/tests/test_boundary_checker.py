import subprocess
from pathlib import Path

from tools.check_boundaries import check_repository


def write(root: Path, relative: str, content: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def init_git(root: Path) -> None:
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(root)],
        check=True,
        capture_output=True,
    )


def test_clean_tree_passes(tmp_path: Path) -> None:
    write(tmp_path, "packages/quant_core/src/ashare_quant_core/model.py", "VALUE = 1\n")

    assert check_repository(tmp_path) == []


def test_git_mode_ignores_ignored_market_data(tmp_path: Path) -> None:
    init_git(tmp_path)
    write(tmp_path, ".gitignore", "data/\n")
    write(tmp_path, "data/prices.csv", "date,close\n2026-01-02,10\n")
    write(tmp_path, "packages/quant_core/src/ashare_quant_core/model.py", "VALUE = 1\n")

    assert check_repository(tmp_path, use_git=True) == []


def test_git_mode_checks_untracked_nonignored_files(tmp_path: Path) -> None:
    init_git(tmp_path)
    write(tmp_path, "reports/new-result.json", "{}\n")

    assert check_repository(tmp_path, use_git=True) == [
        "forbidden tracked tree: reports/new-result.json"
    ]


def test_legacy_dependency_is_rejected(tmp_path: Path) -> None:
    legacy_name = "ashare-pilot" + "-legacy"
    write(tmp_path, "config/source.json", f'{{"dependency": "{legacy_name}"}}\n')

    assert check_repository(tmp_path) == ["Legacy dependency: config/source.json"]


def test_legacy_dependency_in_markdown_is_rejected(tmp_path: Path) -> None:
    legacy_name = "ashare-pilot" + "-legacy"
    write(tmp_path, "docs/source.md", f"Do not load `{legacy_name}`.\n")

    assert check_repository(tmp_path) == ["Legacy dependency: docs/source.md"]


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


def test_gitmodules_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, ".gitmodules", '[submodule "legacy"]\npath = vendor/legacy\n')

    assert check_repository(tmp_path) == ["Git submodules are forbidden: .gitmodules"]


def test_gitlink_is_rejected(tmp_path: Path) -> None:
    init_git(tmp_path)
    write(tmp_path, "README.md", "fixture\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{commit},vendor/dependency",
        ],
        check=True,
    )

    assert check_repository(tmp_path, use_git=True) == [
        "git submodule is forbidden: vendor/dependency"
    ]


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


def test_research_can_import_quant_core(tmp_path: Path) -> None:
    write(
        tmp_path,
        "apps/research/src/ashare_research_app/model.py",
        "import ashare_quant_core\n",
    )

    assert check_repository(tmp_path) == []


def test_unknown_python_area_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "apps/new_worker/worker.py", "VALUE = 1\n")

    assert check_repository(tmp_path) == [
        "unregistered Python source area: apps/new_worker/worker.py"
    ]


def test_forbidden_dynamic_import_is_rejected(tmp_path: Path) -> None:
    write(
        tmp_path,
        "packages/quant_core/src/ashare_quant_core/model.py",
        'module = __import__("ashare_signal_runner")\n',
    )

    assert check_repository(tmp_path) == [
        "forbidden dynamic import in "
        "packages/quant_core/src/ashare_quant_core/model.py:1: ashare_signal_runner"
    ]


def test_project_dependency_is_rejected(tmp_path: Path) -> None:
    write(
        tmp_path,
        "packages/quant_core/pyproject.toml",
        '[project]\nname = "ashare-quant-core"\ndependencies = ["ashare-signal-runner"]\n',
    )

    assert check_repository(tmp_path) == [
        "forbidden project dependency in packages/quant_core/pyproject.toml: "
        "ashare-signal-runner"
    ]


def test_unregistered_editable_lock_source_is_rejected(tmp_path: Path) -> None:
    write(
        tmp_path,
        "pyproject.toml",
        '[tool.uv.workspace]\nmembers = ["packages/quant_core"]\n',
    )
    write(
        tmp_path,
        "uv.lock",
        'version = 1\n[[package]]\nname = "bad"\nsource = { editable = "../outside" }\n',
    )

    assert check_repository(tmp_path) == [
        "unregistered editable source in uv.lock: ../outside"
    ]


def test_wall_clock_alias_in_core_is_rejected(tmp_path: Path) -> None:
    write(
        tmp_path,
        "packages/quant_core/src/ashare_quant_core/model.py",
        "from datetime import datetime as DT\nVALUE = DT.utcnow()\n",
    )

    assert check_repository(tmp_path) == [
        "wall-clock call in packages/quant_core/src/ashare_quant_core/model.py:2: "
        "datetime.datetime.utcnow"
    ]


def test_wall_clock_text_in_comment_is_allowed(tmp_path: Path) -> None:
    write(
        tmp_path,
        "packages/quant_core/src/ashare_quant_core/model.py",
        "# datetime.now() is forbidden here\nVALUE = 1\n",
    )

    assert check_repository(tmp_path) == []


def test_research_wall_clock_call_is_rejected(tmp_path: Path) -> None:
    write(
        tmp_path,
        "apps/research/src/ashare_research_app/run.py",
        "from datetime import date\nVALUE = date.today()\n",
    )

    assert check_repository(tmp_path) == [
        "wall-clock call in apps/research/src/ashare_research_app/run.py:2: "
        "datetime.date.today"
    ]


def test_data_gateway_wall_clock_call_is_rejected(tmp_path: Path) -> None:
    write(
        tmp_path,
        "services/data_gateway/src/ashare_data_gateway/run.py",
        "import time\nVALUE = time.time()\n",
    )

    assert check_repository(tmp_path) == [
        "wall-clock call in "
        "services/data_gateway/src/ashare_data_gateway/run.py:2: time.time"
    ]


def test_linux_personal_path_is_rejected(tmp_path: Path) -> None:
    linux_home = "/" + "home/person/data"
    write(tmp_path, "config/source.json", f'{{"path": "{linux_home}"}}\n')

    assert check_repository(tmp_path) == ["personal absolute path: config/source.json"]


def test_personal_paths_in_markdown_and_windows_form_are_rejected(
    tmp_path: Path,
) -> None:
    windows_home = "C:\\" + "Users\\person\\data"
    write(tmp_path, "docs/local.md", f"Local path: {windows_home}\n")

    assert check_repository(tmp_path) == ["personal absolute path: docs/local.md"]


def test_non_utf8_source_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config/source.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"bad":"' + bytes([0xFF]) + b'"}')

    assert check_repository(tmp_path) == ["non-UTF-8 source file: config/source.json"]
