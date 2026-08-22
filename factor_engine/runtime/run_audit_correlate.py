"""factor_run 与 data_access audit 关联查询。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from storage.catalog import FactorCatalog


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_audit_log(path: str | Path | None = None) -> list[dict[str, Any]]:
    """读取 data_access audit JSONL。"""
    import os

    if path is None:
        explicit = os.environ.get("QUANT_AUDIT_LOG")
        if explicit:
            audit_path = Path(explicit)
        else:
            workspace = os.environ.get(
                "QUANTSOCIETY_WORKSPACE_DATA_ROOT",
                str(Path(__file__).resolve().parents[2] / "workspace_data"),
            )
            audit_path = Path(workspace) / "logs" / "data_access_audit.jsonl"
    else:
        audit_path = Path(path)

    if not audit_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return rows


def _factor_id_from_audit(entry: dict[str, Any]) -> str | None:
    params = entry.get("params") or {}
    if isinstance(params, dict) and params.get("factor_id"):
        return str(params["factor_id"])
    extra = entry.get("extra") or {}
    if isinstance(extra, dict):
        if extra.get("factor_id"):
            return str(extra["factor_id"])
        if extra.get("run_id"):
            return None
    return None


def correlate_runs_with_audit(
    catalog: FactorCatalog,
    audit_entries: Iterable[dict[str, Any]],
    *,
    factor_id: str | None = None,
    max_delta_seconds: int = 3600,
) -> list[dict[str, Any]]:
    """按 factor_id + 时间邻近把 factor_run 与 audit 记录关联。"""
    audits = list(audit_entries)
    if factor_id is not None:
        audits = [
            row
            for row in audits
            if _factor_id_from_audit(row) == factor_id
            or (
                row.get("dataset") in {"factor_lake", "factor_lake_staging"}
                and (row.get("params") or {}).get("factor_id") == factor_id
            )
        ]

    if factor_id is not None:
        runs = catalog.list_runs(factor_id, limit=200)
    else:
        runs = catalog.list_recent_runs(limit=200)

    correlated: list[dict[str, Any]] = []
    for run in runs:
        run_id = str(run.get("run_id") or "")
        fid = str(run.get("factor_id") or "")
        run_ts = _parse_ts(run.get("created_at"))
        extra = {}
        raw_extra = run.get("extra_json")
        if raw_extra:
            try:
                extra = json.loads(raw_extra)
            except json.JSONDecodeError:
                extra = {}

        matches: list[dict[str, Any]] = []
        for audit in audits:
            audit_fid = _factor_id_from_audit(audit)
            if audit_fid and audit_fid != fid:
                continue
            audit_extra = audit.get("extra") or {}
            if isinstance(audit_extra, dict) and audit_extra.get("run_id") == run_id:
                matches.append({**audit, "match_reason": "run_id"})
                continue
            audit_ts = _parse_ts(audit.get("ts"))
            if run_ts and audit_ts:
                delta = abs((audit_ts - run_ts).total_seconds())
                if delta <= max_delta_seconds and audit.get("op") in {"publish", "write", "upsert"}:
                    matches.append({**audit, "match_reason": "time_window", "delta_seconds": delta})

        correlated.append(
            {
                "run_id": run_id,
                "factor_id": fid,
                "created_at": run.get("created_at"),
                "row_count": run.get("row_count"),
                "dq_passed": run.get("dq_passed"),
                "data_snapshot_id": extra.get("data_snapshot_id"),
                "audit_matches": matches,
            }
        )
    return correlated
