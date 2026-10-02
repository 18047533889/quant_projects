"""Atomically publish verified receipts; preserve the live artifact on failure."""
import json
import os
from pathlib import Path
import tempfile


def publish_receipts(output: Path, result: dict) -> None:
    output = Path(output)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=".native-receipt-", suffix=".json", delete=False,
        ) as handle:
            pending = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, output)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
