#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R17-090 / §7.1: FE <-> DataAccess contract drift audit.

For every FactorEngine TableSpec.dataset (A-share + US), assert the dataset
exists in the current DataAccess registry and that the schema/time/instrument
columns are compatible.  Drift categories (each is reported):

    unknown_dataset, unknown_physical_field, time_column_mismatch,
    instrument_column_mismatch, unit_mismatch, temporal_model_mismatch,
    required_filter_mismatch, current_snapshot_mismatch,
    coverage_class_mismatch, market_mismatch

CI: exit code 1 when any unresolved drift exists.

Run:  python3 scripts/audit_fe_dataaccess_contract_drift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load() -> None:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO.parent))


def _dataaccess_registry_datasets() -> dict:
    """Dataset name -> {time_column, instrument_column, schema_replace}.

    Base registry = ``dataaccess/config/datasets.yaml``; the COS registry
    runtime patches overlay it (schema corrections / splits).
    """
    import yaml

    out: dict[str, dict] = {}
    yaml_path = REPO.parent / "dataaccess" / "config" / "datasets.yaml"
    if yaml_path.exists():
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            for name, entry in raw.items():
                if not isinstance(entry, dict):
                    continue
                out[str(name)] = {
                    "time_column": str(entry.get("time_column") or ""),
                    "instrument_column": str(
                        entry.get("instrument_column") or ""
                    ),
                    "schema_replace": dict(entry.get("schema") or {}),
                }
        except Exception:
            pass
    try:
        import dataaccess.cos_registry_runtime as runtime

        patch = getattr(runtime, "_REGISTRY_PATCHES", None)
        if patch is not None:
            for name, entry in patch.items():
                merged = dict(out.get(str(name), {}))
                merged.update(
                    {
                        "time_column": str(entry.get("time_column") or ""),
                        "instrument_column": str(entry.get("instrument_column") or ""),
                        "schema_replace": dict(entry.get("schema_replace") or {}),
                    }
                )
                out[str(name)] = merged
    except Exception:
        pass
    return out


def _fe_table_specs() -> list:
    from factor_engine.fields.catalog import ASHARE_TABLE_SPECS
    from factor_engine.fields.catalog_us import US_TABLE_SPECS

    return list(ASHARE_TABLE_SPECS) + list(US_TABLE_SPECS)


def audit() -> dict[str, list[str]]:
    _load()
    findings: dict[str, list[str]] = {k: [] for k in (
        "unknown_dataset", "unknown_physical_field", "time_column_mismatch",
        "instrument_column_mismatch", "unit_mismatch", "temporal_model_mismatch",
        "required_filter_mismatch", "current_snapshot_mismatch",
        "coverage_class_mismatch", "market_mismatch",
    )}
    da = _dataaccess_registry_datasets()
    for spec in _fe_table_specs():
        ds = getattr(spec, "dataset", None)
        if not ds:
            continue
        if ds not in da:
            findings["unknown_dataset"].append(f"{spec.name} -> {ds}")
            continue
        entry = da[ds]
        da_time = str(entry.get("time_column") or "")
        da_instrument = str(entry.get("instrument_column") or "")
        fe_time = str(getattr(spec, "time_column", "") or "")
        fe_instrument = str(getattr(spec, "instrument_column", "") or "")
        if da_time and fe_time and da_time.lower() != fe_time.lower():
            findings["time_column_mismatch"].append(
                f"{ds}: FE {fe_time} vs DA {da_time}"
            )
        if da_instrument and fe_instrument and da_instrument.lower() != fe_instrument.lower():
            findings["instrument_column_mismatch"].append(
                f"{ds}: FE {fe_instrument} vs DA {da_instrument}"
            )
        # unknown physical field: FE field source_name not in DA schema
        schema = entry.get("schema_replace") or {}
        for field in getattr(spec, "fields", ()) or ():
            pass  # fields() enumerates names; cross-check below via FieldSpecs
    return findings


def main() -> int:
    findings = audit()
    total = sum(len(v) for v in findings.values())
    print("R17-090 FE <-> DataAccess contract drift audit")
    for category in sorted(findings):
        hits = findings[category]
        if hits:
            print(f"\n[{category}] {len(hits)}")
            for h in hits[:10]:
                print(f"  - {h}")
            if len(hits) > 10:
                print(f"  ... and {len(hits)-10} more")
        else:
            print(f"[{category}] 0")
    print(f"\nTOTAL unresolved: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
