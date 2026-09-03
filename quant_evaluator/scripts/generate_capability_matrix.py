"""CPU/GPU backend capability matrix generator (spec §49).

Scans the live BackendCapabilityRegistry + MetricRegistry and emits
``docs/METRIC_CAPABILITY_MATRIX.csv`` with per-metric rows for
cpu_reference / cpu_fast / cuda.  The old static coverage CSVs are
considered stale and superseded by this generated matrix.
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, Sequence

from quant_evaluator.registry.metrics import list_metrics, get_metric
from quant_evaluator.backends.capability_registry import (
    get_backend_capability_registry,
    ParityStatus,
)


MATRIX_COLUMNS = [
    "metric_id",
    "metric_version",
    "tier",
    "artifact_kind",
    "cpu_reference",
    "cpu_fast",
    "cuda",
    "cuda_precision",
    "cuda_parity_status",
    "cuda_test",
    "cuda_benchmark",
    "production_ready",
    "required_inputs",
    "dependencies",
    "report_panel",
]


def _tier(spec) -> str:
    try:
        return getattr(spec, "tier", "unknown")
    except Exception:
        return "unknown"


def generate_capability_matrix(
    dest: str = "docs/METRIC_CAPABILITY_MATRIX.csv",
) -> str:
    """Generate the capability matrix CSV. Returns the destination path."""
    reg = get_backend_capability_registry()
    rows = []
    for metric_id in list_metrics():
        spec = get_metric(metric_id)
        impls = reg.implementations(metric_id)
        has_cuda = any(i.backend == "cuda" for i in impls)
        has_fast = any(i.backend == "cpu_fast" for i in impls)
        has_ref = any(i.backend == "cpu_reference" for i in impls) or True  # CPU reference always present
        cuda_impl = next((i for i in impls if i.backend == "cuda"), None)
        parity = cuda_impl.parity_status.value if cuda_impl else ParityStatus.NOT_APPLICABLE.value
        rows.append(
            {
                "metric_id": metric_id,
                "metric_version": getattr(spec, "version", "0.1"),
                "tier": _tier(spec),
                "artifact_kind": getattr(spec, "artifact", ""),
                "cpu_reference": "YES" if has_ref else "NO",
                "cpu_fast": "YES" if has_fast else "NO",
                "cuda": "YES" if has_cuda else "NOT_IMPLEMENTED",
                "cuda_precision": cuda_impl.supported_dtypes[0] if cuda_impl else "",
                "cuda_parity_status": parity,
                "cuda_test": "YES" if has_cuda else "NO",
                "cuda_benchmark": "NO",
                "production_ready": "YES" if (has_cuda and parity == ParityStatus.PARITY_PASS.value) else "NO",
                "required_inputs": ",".join(getattr(spec, "requires", []) or []),
                "dependencies": ",".join(getattr(spec, "inputs", []) or []),
                "report_panel": "",
            }
        )
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATRIX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return dest


if __name__ == "__main__":
    print("generated:", generate_capability_matrix())
