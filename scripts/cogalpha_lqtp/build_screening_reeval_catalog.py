#!/usr/bin/env python3
"""Build unified DSL catalog for screening manifest re-evaluation (190 factors)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "factor_engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "factor_engine"))

from scripts.cogalpha_lqtp.materialize import _normalize_dsl_for_fe  # noqa: E402
from scripts.cogalpha_lqtp.python_to_dsl import (  # noqa: E402
    DslEntry,
    _annotate_eval_route,
    _finalize_entry,
    _python_entry,
)

DEFAULT_MANIFEST = ROOT / "data/cogalpha_lqtp_production/factor_rankic_screening_manifest.json"
DEFAULT_BATCH = ROOT / "data/cogalpha_lqtp_batch/dsl_catalog.json"
DEFAULT_WEEK2 = ROOT / "week2_pv_factors/source/week2_factors_catalog.json"
DEFAULT_POOL726 = ROOT / "data/cogalpha_lqtp_production/reports_7_26/report_data.js"
DEFAULT_WORK = ROOT / "data/cogalpha_lqtp_production"
DEFAULT_EXTERNAL = ROOT / "data/external_factor_packs/unified_catalog.json"
DEFAULT_BATCH_PARSED = ROOT / "data/cogalpha_lqtp_batch/parsed_factors.json"


def _load_pool726(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"window\.REPORT_DATA\s*=\s*(\{.*\})\s*;?\s*$", text, re.S)
    if not m:
        raise RuntimeError(f"cannot parse REPORT_DATA from {path}")
    return json.loads(m.group(1)).get("factors", [])


def _entry_to_dict(entry: DslEntry, *, manifest: dict[str, Any]) -> dict[str, Any]:
    out = asdict(entry)
    out["manifest_source"] = manifest.get("source", "")
    out["manifest_rank_ic"] = manifest.get("rank_ic")
    out["manifest_display_name"] = manifest.get("display_name", entry.function_name)
    return out


def _week2_entry(item: dict[str, Any]) -> DslEntry:
    slug = str(item.get("slug") or "")
    function_name = f"week2_{slug}"
    factor_id = str(item.get("id") or function_name)
    formula = _normalize_dsl_for_fe(str(item.get("formula") or "").strip())
    if not formula:
        return _python_entry(
            factor_id=factor_id,
            function_name=function_name,
            source="week2_catalog",
            notes="missing_formula",
        )
    return _finalize_entry(
        factor_id=factor_id,
        function_name=function_name,
        dsl=formula,
        source="week2_catalog",
        notes=str(item.get("name_zh") or ""),
    )


def _pool726_entry(item: dict[str, Any]) -> DslEntry:
    function_name = str(item.get("factor_name") or item.get("factor_key") or "")
    factor_id = str(item.get("factor_id") or item.get("factor_key") or function_name)
    route = str(item.get("route") or "")
    payload = str(item.get("source_payload") or "").strip()
    if route == "factor_engine" and payload:
        return _finalize_entry(
            factor_id=factor_id,
            function_name=function_name,
            dsl=payload,
            source="factor_pool_726",
            notes=f"route={route}",
        )
    if payload.startswith("def "):
        return _python_entry(
            factor_id=factor_id,
            function_name=function_name,
            source="factor_pool_726",
            notes=f"route={route or 'python'}",
        )
    if payload:
        entry = _finalize_entry(
            factor_id=factor_id,
            function_name=function_name,
            dsl=payload,
            source="factor_pool_726",
            notes=f"route={route}",
        )
        if entry.status == "ready":
            return entry
    return _python_entry(
        factor_id=factor_id,
        function_name=function_name,
        source="factor_pool_726",
        notes="missing_or_invalid_payload",
    )


def _refresh_batch_entry(raw: dict[str, Any]) -> DslEntry:
    """Re-annotate batch entry with latest LQTP/FE routing."""
    dsl = str(raw.get("dsl") or "").strip()
    if raw.get("status") == "python" or not dsl:
        return _python_entry(
            factor_id=str(raw.get("factor_id") or raw["function_name"]),
            function_name=raw["function_name"],
            source=str(raw.get("source") or "cogalpha_batch"),
            notes=str(raw.get("notes") or ""),
        )
    return _finalize_entry(
        factor_id=str(raw.get("factor_id") or raw["function_name"]),
        function_name=raw["function_name"],
        dsl=dsl,
        source=str(raw.get("source") or "cogalpha_batch"),
        notes=str(raw.get("notes") or ""),
    )


def _formula_looks_broken(formula: str) -> bool:
    """Pack 'formula' fields sometimes embed pandas residue, not runnable DSL."""
    text = (formula or "").strip()
    if not text:
        return True
    bad_markers = (
        ".shift(",
        ".rolling(",
        ".pct_change",
        ".groupby",
        "df[",
        "min_periods",
        "EWM(",
        "ewm(",
    )
    return any(m in text for m in bad_markers)


def _external_entry(item: dict[str, Any]) -> DslEntry:
    from scripts.cogalpha_lqtp.lqtp_converter import convert_python
    from scripts.cogalpha_lqtp.lqtp_dsl_compat import is_lqtp_native_dsl

    function_name = str(item.get("function_name") or "")
    factor_id = function_name
    raw_formula = str(item.get("formula") or "").strip()
    py = str(item.get("python_code") or "").strip()
    pack = str(item.get("pack") or "external")
    notes = str((item.get("extra") or {}).get("name_zh") or "")

    candidates: list[str] = []
    if raw_formula and not _formula_looks_broken(raw_formula):
        candidates.append(_normalize_dsl_for_fe(raw_formula, factor_name=function_name))
    if py.startswith("def "):
        conv = convert_python(py, name=function_name, validate_fe=False)
        cand = (conv.lqtp_formula or "").strip()
        if cand and not any(tok in cand for tok in (".rolling", ".pct_change", ".groupby", "min_pe", ".shift")):
            candidates.append(_normalize_dsl_for_fe(cand, factor_name=function_name))

    for formula in candidates:
        if not formula or _formula_looks_broken(formula):
            continue
        entry = _finalize_entry(
            factor_id=factor_id,
            function_name=function_name,
            dsl=formula,
            source=f"external/{pack}",
            notes=notes,
        )
        if entry.status != "ready":
            continue
        if is_lqtp_native_dsl(entry.lqtp_formula or entry.dsl):
            entry.eval_route = "lqtp_dsl"
            entry.lqtp_native = True
            entry.status = "ready"
            entry.dsl = entry.lqtp_formula or entry.dsl
        return entry

    if py.startswith("def "):
        return _python_entry(
            factor_id=factor_id,
            function_name=function_name,
            source=f"external/{pack}",
            notes="python_payload",
        )
    return _python_entry(
        factor_id=factor_id,
        function_name=function_name,
        source=f"external/{pack}",
        notes="missing_formula",
    )


def resolve_function_name(
    manifest_factor: dict[str, Any],
    *,
    batch_by_id: dict[str, dict[str, Any]],
    batch_by_fn: dict[str, dict[str, Any]],
    week2_by_slug: dict[str, dict[str, Any]],
    pool726_by_name: dict[str, dict[str, Any]],
) -> str:
    name = manifest_factor["display_name"]
    source = manifest_factor.get("source", "")
    if name.startswith("week2:"):
        return f"week2_{name.split(':', 1)[1]}"
    if name.startswith("extra20:") or name.startswith("weekly:"):
        return name.split(":", 1)[1]
    if "7/26" in source or name in pool726_by_name:
        return name
    if name.startswith("factor_") and name in batch_by_id:
        return batch_by_id[name]["function_name"]
    if name in batch_by_fn:
        return name
    return name


def build_catalog(
    manifest_factors: list[dict[str, Any]],
    *,
    batch: list[dict[str, Any]],
    week2: list[dict[str, Any]],
    pool726: list[dict[str, Any]],
    external: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    batch_by_id = {e["factor_id"]: e for e in batch if e.get("factor_id")}
    batch_by_fn = {e["function_name"]: e for e in batch}
    week2_by_slug = {f["slug"]: f for f in week2}
    pool726_by_name = {f["factor_name"]: f for f in pool726}
    pool726_by_key = {f.get("factor_key", ""): f for f in pool726}
    external_by_display = {f["display_name"]: f for f in external}
    external_by_fn = {f["function_name"]: f for f in external}

    best: dict[str, tuple[float, DslEntry, dict[str, Any]]] = {}
    python_map: dict[str, str] = {}

    for mf in manifest_factors:
        display = mf["display_name"]
        source = mf.get("source", "")
        rank_ic = float(mf.get("rank_ic") or 0.0)

        if display.startswith("week2:"):
            slug = display.split(":", 1)[1]
            item = week2_by_slug.get(slug)
            if item is None:
                continue
            entry = _week2_entry(item)
        elif display.startswith("extra20:") or display.startswith("weekly:"):
            fn = display.split(":", 1)[1]
            item = external_by_display.get(display) or external_by_fn.get(fn)
            if item is None:
                continue
            entry = _external_entry(item)
            py = str(item.get("python_code") or "")
            if py.strip():
                python_map[entry.function_name] = py
        elif "7/26" in source or display in pool726_by_name:
            item = pool726_by_name.get(display) or pool726_by_key.get(display)
            if item is None:
                continue
            entry = _pool726_entry(item)
            payload = str(item.get("source_payload") or "")
            if entry.status == "python" and payload.startswith("def "):
                python_map[entry.function_name] = payload
        elif display.startswith("factor_") and display in batch_by_id:
            entry = _refresh_batch_entry(batch_by_id[display])
            raw = batch_by_id[display]
            if entry.status == "python":
                py = raw.get("python_code") or ""
                if py:
                    python_map[entry.function_name] = py
        elif display in batch_by_fn:
            entry = _refresh_batch_entry(batch_by_fn[display])
            if entry.status == "python":
                py = batch_by_fn[display].get("python_code") or ""
                if py:
                    python_map[entry.function_name] = py
        else:
            continue

        fn = entry.function_name
        prev = best.get(fn)
        if prev is None or rank_ic > prev[0]:
            best[fn] = (rank_ic, entry, mf)

    catalog = [_entry_to_dict(entry, manifest=mf) for _, entry, mf in best.values()]
    catalog.sort(key=lambda x: float(x.get("manifest_rank_ic") or 0), reverse=True)
    return catalog, python_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Build screening re-eval catalog")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-catalog", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--week2-catalog", type=Path, default=DEFAULT_WEEK2)
    parser.add_argument("--pool726", type=Path, default=DEFAULT_POOL726)
    parser.add_argument("--batch-parsed", type=Path, default=DEFAULT_BATCH_PARSED)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()

    parsed_batch: dict[str, str] = {}
    if args.batch_parsed.exists():
        for rec in json.loads(args.batch_parsed.read_text(encoding="utf-8")):
            parsed_batch[rec["function_name"]] = rec.get("python_code", "")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    batch = json.loads(args.batch_catalog.read_text(encoding="utf-8"))
    week2 = json.loads(args.week2_catalog.read_text(encoding="utf-8"))
    pool726 = _load_pool726(args.pool726)

    external: list[dict[str, Any]] = []
    if DEFAULT_EXTERNAL.exists():
        external = json.loads(DEFAULT_EXTERNAL.read_text(encoding="utf-8")).get("factors", [])

    catalog, python_map = build_catalog(
        manifest.get("factors", []),
        batch=batch,
        week2=week2.get("factors", []),
        pool726=pool726,
        external=external,
    )

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    cat_path = work / "screening_reeval_catalog.json"
    py_path = work / "screening_reeval_parsed_factors.json"
    cat_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    parsed_records: list[dict[str, str]] = []
    batch_py = dict(parsed_batch)
    seen_py: set[str] = set()

    def _add_py(fn: str, code: str, factor_id: str | None = None) -> None:
        if not code.strip() or fn in seen_py:
            return
        seen_py.add(fn)
        parsed_records.append(
            {
                "factor_id": factor_id or fn,
                "function_name": fn,
                "python_code": code,
                "tools": "",
            }
        )

    for fn, code in python_map.items():
        rec = next((e for e in catalog if e["function_name"] == fn), None)
        _add_py(fn, code, rec.get("factor_id", fn) if rec else fn)

    for entry in catalog:
        fn = entry["function_name"]
        if fn in seen_py:
            continue
        code = batch_py.get(fn, "")
        if code.strip():
            _add_py(fn, code, entry.get("factor_id", fn))

    # 7/26 pool python payloads
    for item in pool726:
        fn = str(item.get("factor_name") or "")
        payload = str(item.get("source_payload") or "")
        if fn and payload.strip().startswith("def "):
            _add_py(fn, payload, str(item.get("factor_id") or fn))
    py_path.write_text(json.dumps(parsed_records, ensure_ascii=False, indent=2), encoding="utf-8")

    routes: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for e in catalog:
        routes[e.get("eval_route", "?")] = routes.get(e.get("eval_route", "?"), 0) + 1
        statuses[e.get("status", "?")] = statuses.get(e.get("status", "?"), 0) + 1

    print(
        f"catalog {len(catalog)} factors "
        f"routes={routes} status={statuses} python_defs={len(parsed_records)} "
        f"-> {cat_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
