"""Exclusive, bounded writer for diagnostic profile progress receipts."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping

MAX_PROGRESS_BYTES = 1024**2


class ExclusiveProgressWriter:
    """Own one newly-created progress file for its lifetime."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._stream = None

    def __enter__(self) -> "ExclusiveProgressWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def write(self, base: Mapping[str, Any], events: list[Mapping[str, Any]],
              *, status: str = "running", error_type: str | None = None) -> None:
        if self._stream is None:
            raise RuntimeError("progress writer is not open")
        report = {**base, "status": status, "run_started": True,
                  "validated_runs": events, "qualification_available": False}
        if error_type is not None:
            report["error_type"] = error_type
        payload = (json.dumps(report, indent=2, ensure_ascii=False,
                              allow_nan=False) + "\n").encode("utf-8")
        if len(payload) > MAX_PROGRESS_BYTES:
            raise ValueError("progress report exceeds 1 MiB")
        # Continue addressing our owned inode if the pathname is replaced.
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(payload.decode("utf-8"))
        self._stream.flush()
