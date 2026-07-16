"""全链路 data_snapshot_id 对账。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from runtime.lineage import hash_data_source_config
from storage.catalog import FactorCatalog


def _parse_json_field(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def _unique_parquet_snapshot_ids(factor_dir: Path) -> set[str]:
    if not factor_dir.exists():
        return set()
    values: set[str] = set()
    for pq_file in factor_dir.rglob("data.parquet"):
        table = pq.read_table(pq_file, columns=["data_snapshot_id"])
        if "data_snapshot_id" not in table.column_names:
            continue
        col = table.column("data_snapshot_id").to_pylist()
        for item in col:
            text = str(item or "").strip()
            if text:
                values.add(text)
    return values


def reconcile_data_snapshot(
    *,
    factor_id: str,
    lake_root: str | Path,
    data_source_config: dict[str, Any] | None = None,
    catalog: FactorCatalog | None = None,
) -> dict[str, Any]:
    """比对 config hash / registry / factor_run / parquet 中的 snapshot id。"""
    root = Path(lake_root)
    cat = catalog or FactorCatalog(root / "_catalog.sqlite")

    registry_info = cat.get_factor_info(factor_id) or {}
    registry_config = _parse_json_field(registry_info.get("data_source_json"))
    registry_hash = (
        hash_data_source_config(registry_config) if registry_config else None
    )

    if data_source_config:
        expected = hash_data_source_config(data_source_config)
    else:
        expected = registry_hash

    runs = cat.list_runs(factor_id, limit=1)
    run_extra: dict[str, Any] = {}
    run_snapshot: str | None = None
    if runs:
        run_extra = _parse_json_field(runs[0].get("extra_json"))
        run_snapshot = run_extra.get("data_snapshot_id")

    parquet_snapshots = _unique_parquet_snapshot_ids(root / "factors" / factor_id)

    checks = {
        "config_vs_registry": expected == registry_hash if expected and registry_hash else None,
        "config_vs_latest_run": expected == run_snapshot if expected and run_snapshot else None,
        "registry_vs_latest_run": registry_hash == run_snapshot if registry_hash and run_snapshot else None,
        "parquet_unique": len(parquet_snapshots) <= 1,
        "parquet_vs_config": (
            (expected in parquet_snapshots or not parquet_snapshots)
            if expected
            else None
        ),
        "parquet_vs_registry": (
            (registry_hash in parquet_snapshots or not parquet_snapshots)
            if registry_hash
            else None
        ),
    }
    mismatches = [name for name, ok in checks.items() if ok is False]
    ok = not mismatches

    return {
        "ok": ok,
        "factor_id": factor_id,
        "lake_root": str(root),
        "expected_snapshot_id": expected,
        "registry_snapshot_id": registry_hash,
        "latest_run_snapshot_id": run_snapshot,
        "parquet_snapshot_ids": sorted(parquet_snapshots),
        "checks": checks,
        "mismatches": mismatches,
    }
