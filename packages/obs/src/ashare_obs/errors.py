"""Auditable error types that bind machine-readable context to failures.

The purpose of this module is to make every failure answer "which part went
wrong" without re-running the code. A raised exception carries a ``context``
mapping — module, function, input hashes, dataset ids, or any other provenance
the caller supplies — so an auditor can replay the origin from the log alone.
"""

from __future__ import annotations

from typing import Any


class AuditableError(Exception):
    """Base exception carrying a machine-readable ``context`` dict."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context)

    def as_dict(self) -> dict[str, Any]:
        """Render the error and its context as a JSON-ready dict."""
        return {"error": str(self), "context": self.context}
