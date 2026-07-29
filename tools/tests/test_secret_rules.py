import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_tushare_rule_detects_token_assignment() -> None:
    config = tomllib.loads((ROOT / "gitleaks.toml").read_text(encoding="utf-8"))
    rule = next(item for item in config["rules"] if item["id"] == "tushare-token")
    token = "a" * 40

    assert re.search(rule["regex"], f'tushare_token = "{token}"')


def test_tushare_rule_does_not_flag_unlabelled_hash() -> None:
    config = tomllib.loads((ROOT / "gitleaks.toml").read_text(encoding="utf-8"))
    rule = next(item for item in config["rules"] if item["id"] == "tushare-token")

    assert re.search(rule["regex"], "sha256 = " + "a" * 64) is None
