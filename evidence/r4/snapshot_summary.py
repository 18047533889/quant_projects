"""Compact runtime identity; does not serialize the full operator catalog."""
import json
from factor_engine.runtime.operator_snapshot import (
    build_runtime_operator_snapshot, _live_runtime_digest,
)

snapshot = build_runtime_operator_snapshot(profile="runtime")
print(json.dumps({
    "schema_version": snapshot["schema_version"],
    "generated_at": snapshot["generated_at"],
    "catalog_digest": snapshot["catalog_digest"],
    "live_runtime_digest": _live_runtime_digest(),
    "counts": snapshot["counts"],
    "scope": "Discovery and declared contracts, not numerical or production certification.",
}, ensure_ascii=False, indent=2))
