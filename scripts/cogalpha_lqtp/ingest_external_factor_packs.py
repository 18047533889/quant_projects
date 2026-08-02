#!/usr/bin/env python3
"""Extract extra20 / weekly factor packs → unified catalog for screening pipeline."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data/external_factor_packs"
RANK_IC_RE = re.compile(r"Rank\s*IC\s*([0-9.]+)\s*%", re.I)
CATALOG_GLOBS = (
    "**/week2_factors_catalog.json",
    "**/factors_catalog.json",
    "**/catalog.json",
    "**/selected_factors.json",
    "**/index.json",
)


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _rank_ic_from_item(item: dict[str, Any]) -> float | None:
    for key in ("mean_rank_ic", "rank_ic", "rank_ic_pct", "ic_pct"):
        v = _safe_float(item.get(key))
        if v is None:
            continue
        return v / 100.0 if key.endswith("_pct") and abs(v) > 1.5 else v
    desc = str(item.get("description") or item.get("note") or "")
    m = RANK_IC_RE.search(desc)
    if m:
        return float(m.group(1)) / 100.0
    return None


def _slug_from_item(item: dict[str, Any], *, fallback: str) -> str:
    for key in ("slug", "factor_name", "function_name", "factor_key", "factor_id", "id"):
        v = str(item.get(key) or "").strip()
        if v:
            return v.replace(":", "_")
    return fallback


def _formula_from_item(item: dict[str, Any]) -> str:
    for key in ("formula", "dsl", "source_payload", "expression"):
        v = str(item.get(key) or "").strip()
        if v and not v.startswith("def "):
            return v
    return ""


def _python_from_item(item: dict[str, Any]) -> str:
    for key in ("python_code", "source_payload"):
        v = str(item.get(key) or "").strip()
        if v.startswith("def "):
            return v
    return ""


def _load_report_data_js(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"window\.REPORT_DATA\s*=\s*(\{.*\})\s*;?\s*$", text, re.S)
    if not m:
        return []
    payload = json.loads(m.group(1))
    return list(payload.get("factors") or [])


def _parse_json_catalog(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("factors", "items", "entries"):
            if isinstance(data.get(key), list):
                return list(data[key])
    return []


def _parse_manifest_dir(manifest_path: Path) -> dict[str, Any]:
    item = json.loads(manifest_path.read_text(encoding="utf-8"))
    formula = str(item.get("formula") or "").strip()
    desc = str(item.get("description") or "")
    rank_ic = None
    m = RANK_IC_RE.search(desc)
    if m:
        rank_ic = float(m.group(1)) / 100.0
    slug = manifest_path.parent.name
    return {
        "slug": slug,
        "formula": formula,
        "rank_ic": rank_ic,
        "description": desc,
        "manifest_path": str(manifest_path),
    }


FACTOR_NAME_RE = re.compile(r"factor\.name\s*=\s*['\"]([^'\"]+)['\"]")
DEF_NAME_RE = re.compile(r"^def\s+(\w+)\s*\(", re.M)


def _rank_ic_from_summary(summary: dict[str, Any]) -> float | None:
    train = summary.get("train") or {}
    if isinstance(train, dict):
        v = _safe_float(train.get("rank_ic"))
        if v is not None:
            return v
    return _rank_ic_from_item(summary)


def _function_name_from_code(code: str) -> str:
    m = FACTOR_NAME_RE.search(code)
    if m:
        return m.group(1)
    m2 = DEF_NAME_RE.search(code)
    if m2:
        return m2.group(1)
    return ""


def _discover_summary_dirs(extract_root: Path, *, pack_label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for summary_path in sorted(extract_root.glob("**/summary.json")):
        if "__MACOSX" in summary_path.parts:
            continue
        factor_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest_path = factor_dir / "manifest.json"
        manifest = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        code_path = factor_dir / "code.py"
        code = code_path.read_text(encoding="utf-8").strip() if code_path.is_file() else ""
        if not code:
            code = str(manifest.get("code") or "").strip()

        formula_path = factor_dir / "formula.txt"
        formula = (
            str(summary.get("formula") or "").strip()
            or (formula_path.read_text(encoding="utf-8").strip() if formula_path.is_file() else "")
            or str(manifest.get("formula") or "").strip()
        )

        fn = _function_name_from_code(code) or str(summary.get("factor_name") or "").strip()
        if not fn:
            fn = str(summary.get("candidate_id") or factor_dir.name)
        if fn.startswith("cogalpha_") and code:
            named = _function_name_from_code(code)
            if named:
                fn = named

        rank_ic = _rank_ic_from_summary(summary)
        display = f"{pack_label}:{fn}"
        out.append(
            {
                "pack": pack_label,
                "display_name": display,
                "function_name": fn,
                "slug": factor_dir.name,
                "rank_ic": rank_ic,
                "formula": formula,
                "python_code": code if code.startswith("def ") else "",
                "route": str(summary.get("expression_type") or manifest.get("expression_type") or ""),
                "source_file": str(summary_path.relative_to(extract_root)),
                "extra": {
                    "candidate_id": summary.get("candidate_id"),
                    "week": summary.get("week"),
                    "fitness_label": summary.get("fitness_label"),
                    "train_ic": (summary.get("train") or {}).get("ic") if isinstance(summary.get("train"), dict) else None,
                },
            }
        )
    return out


def discover_factors(extract_root: Path, *, pack_label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in _discover_summary_dirs(extract_root, pack_label=pack_label):
        key = item["function_name"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    if out:
        return out

    for pat in CATALOG_GLOBS:
        for path in sorted(extract_root.glob(pat)):
            if "__MACOSX" in path.parts:
                continue
            rows = _parse_json_catalog(path)
            if path.name == "report_data.js" or rows and "factor_key" in (rows[0] if rows else {}):
                pass
            for i, item in enumerate(rows):
                slug = _slug_from_item(item, fallback=f"{path.stem}_{i}")
                fn_prefix = pack_label.replace("-", "_")
                if slug.startswith(f"{fn_prefix}_") or slug.startswith(f"{pack_label}:"):
                    function_name = slug
                    display_name = slug
                else:
                    function_name = f"{fn_prefix}_{slug}"
                    display_name = f"{pack_label}:{slug}"
                key = display_name.lower()
                if key in seen:
                    continue
                seen.add(key)
                rank_ic = _rank_ic_from_item(item)
                out.append(
                    {
                        "pack": pack_label,
                        "display_name": display_name,
                        "function_name": function_name,
                        "slug": slug,
                        "rank_ic": rank_ic,
                        "formula": _formula_from_item(item),
                        "python_code": _python_from_item(item),
                        "route": str(item.get("route") or item.get("expression_type") or ""),
                        "source_file": str(path.relative_to(extract_root)),
                        "extra": {k: item.get(k) for k in ("name_zh", "tier", "cluster", "factor_id") if item.get(k)},
                    }
                )

    for path in sorted(extract_root.glob("**/report_data.js")):
        if "__MACOSX" in path.parts:
            continue
        for i, item in enumerate(_load_report_data_js(path)):
            slug = _slug_from_item(item, fallback=f"report_{i}")
            display_name = f"{pack_label}:{slug}"
            if display_name.lower() in seen:
                continue
            seen.add(display_name.lower())
            rank_ic = _rank_ic_from_item(item)
            out.append(
                {
                    "pack": pack_label,
                    "display_name": display_name,
                    "function_name": f"{pack_label.replace('-', '_')}_{slug}",
                    "slug": slug,
                    "rank_ic": rank_ic,
                    "formula": _formula_from_item(item),
                    "python_code": _python_from_item(item),
                    "route": str(item.get("route") or ""),
                    "source_file": str(path.relative_to(extract_root)),
                    "extra": {},
                }
            )

    for path in sorted(extract_root.glob("**/manifest.json")):
        if "__MACOSX" in path.parts or "candidate_pool" not in path.parts:
            continue
        item = _parse_manifest_dir(path)
        slug = item["slug"]
        display_name = f"{pack_label}:{slug}"
        if display_name.lower() in seen:
            continue
        seen.add(display_name.lower())
        out.append(
            {
                "pack": pack_label,
                "display_name": display_name,
                "function_name": f"{pack_label.replace('-', '_')}_{slug}",
                "slug": slug,
                "rank_ic": item.get("rank_ic"),
                "formula": item.get("formula") or "",
                "python_code": "",
                "route": "dsl",
                "source_file": str(path.relative_to(extract_root)),
                "extra": {"description": item.get("description", "")},
            }
        )

    for path in sorted(extract_root.glob("**/*.dsl")):
        if "__MACOSX" in path.parts:
            continue
        slug = path.stem
        display_name = f"{pack_label}:{slug}"
        if display_name.lower() in seen:
            continue
        seen.add(display_name.lower())
        out.append(
            {
                "pack": pack_label,
                "display_name": display_name,
                "function_name": f"{pack_label.replace('-', '_')}_{slug}",
                "slug": slug,
                "rank_ic": None,
                "formula": path.read_text(encoding="utf-8").strip(),
                "python_code": "",
                "route": "dsl",
                "source_file": str(path.relative_to(extract_root)),
                "extra": {},
            }
        )

    return out


def extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    # flatten single top-level folder
    kids = [p for p in dest.iterdir() if p.name != "__MACOSX" and not p.name.startswith(".")]
    if len(kids) == 1 and kids[0].is_dir():
        return kids[0]
    return dest


def dedupe_against_existing(factors: list[dict[str, Any]], existing_keys: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dupes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for f in factors:
        keys = {
            f["display_name"].strip().lower(),
            f["function_name"].strip().lower(),
            f.get("slug", "").strip().lower(),
            f["function_name"].replace(f"{f['pack']}_", "").lower(),
        }
        keys |= {k.replace("week2:", "").replace("extra20:", "").replace("weekly:", "") for k in keys}
        if keys & existing_keys or keys & seen:
            dupes.append(f)
            continue
        seen |= keys
        kept.append(f)
    return kept, dupes


def load_existing_dedupe_keys(work: Path) -> set[str]:
    keys: set[str] = set()
    manifest_path = work / "factor_rankic_screening_manifest.json"
    if manifest_path.exists():
        for item in json.loads(manifest_path.read_text(encoding="utf-8")).get("factors", []):
            name = str(item.get("display_name", "")).strip().lower()
            if name:
                keys.add(name)
                keys.add(name.split(":", 1)[-1])
    cat_path = work / "screening_reeval_catalog.json"
    if cat_path.exists():
        for item in json.loads(cat_path.read_text(encoding="utf-8")):
            fn = str(item.get("function_name", "")).strip().lower()
            if fn:
                keys.add(fn)
                for prefix in ("week2_", "extra20_", "weekly_"):
                    if fn.startswith(prefix):
                        keys.add(fn[len(prefix) :])
    week2 = ROOT / "week2_pv_factors/source/week2_factors_catalog.json"
    if week2.exists():
        for item in json.loads(week2.read_text(encoding="utf-8")).get("factors", []):
            keys.add(f"week2:{item.get('slug','')}".lower())
            keys.add(str(item.get("slug", "")).lower())
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest external factor zip packs")
    parser.add_argument(
        "zips",
        nargs="*",
        type=Path,
        help="Zip paths (default: scan data/external_factor_packs/incoming/*.zip)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    args = parser.parse_args()

    incoming = args.out_dir / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    zips = list(args.zips)
    if not zips:
        zips = sorted(incoming.glob("*.zip"))
    if not zips:
        print(
            f"No zip files found. Copy packs to {incoming}/ then re-run, e.g.:\n"
            "  extra20_factors_pack.zip\n"
            "  weekly_factors_pack.zip",
            file=sys.stderr,
        )
        return 1

    existing_keys = load_existing_dedupe_keys(args.work_dir)
    all_factors: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for zip_path in zips:
        if not zip_path.is_file():
            print(f"skip missing {zip_path}", file=sys.stderr)
            continue
        pack_label = zip_path.stem.replace("_factors_pack", "").replace("_pack", "")
        extract_to = args.out_dir / "extracted" / zip_path.stem
        if extract_to.exists():
            shutil.rmtree(extract_to)
        root = extract_zip(zip_path, extract_to)
        found = discover_factors(root, pack_label=pack_label)
        kept, dupes = dedupe_against_existing(found, existing_keys)
        for f in kept:
            existing_keys.add(f["display_name"].lower())
            existing_keys.add(f["function_name"].lower())
        all_factors.extend(kept)
        summary.append(
            {
                "zip": str(zip_path),
                "extracted": str(root),
                "discovered": len(found),
                "new": len(kept),
                "duplicates": len(dupes),
            }
        )
        print(f"{zip_path.name}: discovered={len(found)} new={len(kept)} dup={len(dupes)}")

    out_catalog = args.out_dir / "unified_catalog.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "pack_count": len(summary),
        "factor_count": len(all_factors),
        "summary": summary,
        "factors": all_factors,
    }
    out_catalog.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out_catalog} ({len(all_factors)} factors)")
    return 0 if all_factors else 1


if __name__ == "__main__":
    raise SystemExit(main())
