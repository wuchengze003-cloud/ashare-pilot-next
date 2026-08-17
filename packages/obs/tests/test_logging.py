import json

from ashare_obs import AuditableError, StructuredLogger, export_text, read_records


def test_writes_json_lines_with_monotonic_sequence(tmp_path) -> None:
    logger = StructuredLogger(tmp_path, component="research", clock=lambda: 1.0)
    logger.info("dataset.loaded", dataset_id="ds-1")
    logger.debug("factor.computed", factor="momentum")
    records = read_records(logger.path)
    assert [r["seq"] for r in records] == [1, 2]
    assert [r["level"] for r in records] == ["INFO", "DEBUG"]
    assert [r["component"] for r in records] == ["research", "research"]
    assert [r["event"] for r in records] == ["dataset.loaded", "factor.computed"]
    assert records[0]["mono"] == 1.0
    assert records[0]["context"] == {"dataset_id": "ds-1"}


def test_each_event_is_a_single_valid_json_line(tmp_path) -> None:
    logger = StructuredLogger(tmp_path)
    logger.info("a")
    logger.info("b")
    lines = logger.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)


def test_error_records_exception_provenance(tmp_path) -> None:
    logger = StructuredLogger(tmp_path)
    err = AuditableError("hash mismatch", module="datasets", expected="a", got="b")
    logger.error("snapshot.rejected", exc=err)
    rec = read_records(logger.path)[0]
    assert rec["level"] == "ERROR"
    assert rec["context"]["exc_type"] == "AuditableError"
    assert rec["context"]["exc_msg"] == "hash mismatch"
    assert rec["context"]["exc_context"] == {"module": "datasets", "expected": "a", "got": "b"}


def test_plain_exception_still_records_type_and_message(tmp_path) -> None:
    logger = StructuredLogger(tmp_path)
    logger.error("pipeline.failed", exc=ValueError("bad input"))
    rec = read_records(logger.path)[0]
    assert rec["context"]["exc_type"] == "ValueError"
    assert rec["context"]["exc_msg"] == "bad input"
    assert "exc_context" not in rec["context"]


def test_export_text_is_human_readable(tmp_path) -> None:
    logger = StructuredLogger(tmp_path, component="research")
    logger.info("snapshot.loaded", dataset_id="ds-1", rows=3)
    text = export_text(logger.path)
    assert "#1 [INFO] research:snapshot.loaded" in text
    assert "dataset_id" in text


def test_clock_is_injectable(tmp_path) -> None:
    calls: list[None] = []

    def fake_clock() -> float:
        calls.append(None)
        return 42.5

    logger = StructuredLogger(tmp_path, clock=fake_clock)
    logger.info("event")
    assert len(calls) == 1
    assert read_records(logger.path)[0]["mono"] == 42.5


def test_default_clock_does_not_read_wall_time(tmp_path, monkeypatch) -> None:
    import time

    def boom() -> float:
        raise AssertionError("wall clock was read")

    monkeypatch.setattr(time, "time", boom)
    logger = StructuredLogger(tmp_path)
    logger.info("event")
    assert len(read_records(logger.path)) == 1
