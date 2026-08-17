"""Structured JSON-lines logging with an injectable clock.

Design goals, from first principles:

- One event == one JSON object == one line. Reading it needs no framework:
  ``jq``, ``pandas``, or any auditor can filter and replay it.
- No implicit wall-clock. Timestamps come from an injectable clock; the default
  is ``time.monotonic``, which the repo's "no implicit current time" rule
  permits, and which never pollutes data-content identity.
- A monotonic sequence number makes ordering unambiguous even if the injected
  clock repeats.
- Errors record exception type, message, and any ``AuditableError`` context, so
  the log alone can answer "which part went wrong".
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import AuditableError

Clock = Callable[[], float]


class StructuredLogger:
    """Append-only structured logger writing JSON-lines under a run directory.

    Args:
        run_dir: directory that will receive ``logs/<filename>``.
        component: logical owner of the log (e.g. "research", "signal_runner").
        clock: monotonic clock callable; injectable for deterministic tests.
        filename: log file name under ``run_dir/logs/``.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        component: str = "",
        clock: Clock | None = None,
        filename: str = "events.jsonl",
    ) -> None:
        self._run_dir = Path(run_dir)
        self._component = component
        self._clock: Clock = clock if clock is not None else time.monotonic
        self._seq = 0
        logs_dir = self._run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = logs_dir / filename

    def _emit(self, level: str, event: str, context: dict[str, Any] | None) -> None:
        self._seq += 1
        record: dict[str, Any] = {
            "seq": self._seq,
            "mono": round(self._clock(), 6),
            "level": level,
            "component": self._component,
            "event": event,
        }
        if context:
            record["context"] = context
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def debug(self, event: str, **context: Any) -> None:
        """Record a DEBUG event with optional structured context."""
        self._emit("DEBUG", event, context or None)

    def info(self, event: str, **context: Any) -> None:
        """Record an INFO event with optional structured context."""
        self._emit("INFO", event, context or None)

    def warning(self, event: str, **context: Any) -> None:
        """Record a WARN event with optional structured context."""
        self._emit("WARN", event, context or None)

    def error(self, event: str, exc: BaseException | None = None, **context: Any) -> None:
        """Record an ERROR event; an exception adds type/message/context."""
        ctx = dict(context)
        if exc is not None:
            ctx["exc_type"] = type(exc).__name__
            ctx["exc_msg"] = str(exc)
            if isinstance(exc, AuditableError):
                ctx["exc_context"] = exc.context
        self._emit("ERROR", event, ctx or None)


def read_records(log_path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines log back into a list of dicts (for replay/audit)."""
    records: list[dict[str, Any]] = []
    with Path(log_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def export_text(log_path: Path) -> str:
    """Render a JSON-lines log as human-readable text for review or audit."""
    lines: list[str] = []
    for rec in read_records(log_path):
        head = (
            f"#{rec.get('seq')} [{rec.get('level')}] "
            f"{rec.get('component') or '-'}:{rec.get('event')}"
        )
        ctx = rec.get("context")
        if ctx:
            head += " " + json.dumps(ctx, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        lines.append(head)
    return "\n".join(lines)
