#!/usr/bin/env python3
"""Capability-scoped scale evidence gate.

Collector success is deliberately not an execution certificate.  A caller
must request one concrete scope and minimum scale; absent, skipped, failed, or
different-unit evidence remains NOT_RUN/ineligible for that capability only.
"""
from __future__ import annotations

import argparse
import json
import re
import hashlib
from pathlib import Path
from typing import Any

VALID_SCOPES = frozenset({
    "mathematical_correctness", "reference_batch_parity", "row_scale",
    "factor_scale", "queue_resume", "gpu_cuda", "real_data_shadow",
})


def verified_factor_execution(report_path: Path, source_sha: str):
    """Normalize this repository's actual FE disk-oracle report, not job state."""
    raw = report_path.read_bytes()
    report = json.loads(raw)
    dimensions = report.get("scale_dimensions", {})
    if (report.get("source_sha") != source_sha or report.get("status") != "PASS"
            or report.get("oracle_status") != "PASS"
            or dimensions.get("formula_family") != "column_plus_distinct_scalar"
            or dimensions.get("F_factor_count") != report.get("roots")
            or not re.fullmatch(r"[0-9a-f]{64}", report.get("benchmark_sha256", ""))):
        raise ValueError("FE report source/oracle/shape identity is incomplete")
    return dict(scope="factor_scale", scale=report["roots"], scale_unit="unique_factors",
        source_sha=source_sha, status="PASS", execution_verified=True,
        execution_ref=str(report_path), execution_sha256=hashlib.sha256(raw).hexdigest(),
        backend=dimensions["backend"], real_data=False, dimensions=dimensions,
        **{k: report[k] for k in ("requested", "completed", "failed", "skipped",
                                 "unique_formulas", "metric_instances")},
        wall_seconds=report["seconds"], peak_rss_bytes=report["peak_rss_bytes"],
        output_bytes=report["output_bytes"], limitations=report["limitations"])


def qualifies(receipt: dict[str, Any], scope: str, minimum: int, *, source_sha: str | None = None) -> bool:
    if scope not in VALID_SCOPES:
        raise ValueError(f"unknown evidence scope: {scope}")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        raise ValueError("minimum must be a positive integer")
    if (not source_sha or not re.fullmatch(r"[0-9a-f]{40}", source_sha)
            or receipt.get("source_sha") != source_sha):
        return False
    if receipt.get("collector_status") != "PASS":
        return False
    for evidence in receipt.get("evidence", []):
        if (
            evidence.get("scope") == scope
            and evidence.get("status") == "PASS"
            and evidence.get("execution_verified") is True
            and evidence.get("source_sha") == source_sha
            and isinstance(evidence.get("execution_ref"), str)
            and bool(evidence.get("execution_ref"))
            and re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("execution_sha256", "")))
            and isinstance(evidence.get("scale"), int)
            and not isinstance(evidence["scale"], bool)
            and evidence["scale"] >= minimum
        ):
            unit = "rows" if scope == "row_scale" else "unique_factors"
            if evidence.get("scale_unit") != unit:
                continue
            if scope == "gpu_cuda" and evidence.get("backend") != "cuda":
                continue
            if scope == "real_data_shadow" and evidence.get("real_data") is not True:
                continue
            if scope in {"factor_scale", "gpu_cuda", "real_data_shadow", "queue_resume"}:
                count = evidence["scale"]
                if any(isinstance(evidence.get(k), bool) or not isinstance(evidence.get(k), int)
                       for k in ("requested", "completed", "failed", "skipped", "unique_formulas")):
                    continue
                if (evidence["requested"] != count or evidence["completed"] != count
                        or evidence["failed"] != 0 or evidence["skipped"] != 0
                        or evidence["unique_formulas"] < count
                        or not evidence.get("metric_instances")):
                    continue
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--scope", required=True, choices=sorted(VALID_SCOPES))
    parser.add_argument("--minimum", required=True, type=int)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text())
    return 0 if qualifies(payload, args.scope, args.minimum, source_sha=args.source_sha) else 1


if __name__ == "__main__":
    raise SystemExit(main())
