#!/usr/bin/env python3
"""Fix remaining python_only DSL + inject/eval LOOKAHEAD_DEFERRED factors."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "factor_engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "factor_engine"))

from scripts.cogalpha_lqtp.ast_translator import dsl_to_lqtp  # noqa: E402
from scripts.cogalpha_lqtp.dsl_sanitize import sanitize_dsl  # noqa: E402
from scripts.cogalpha_lqtp.fix_lookahead_rank import fix_rank_lookahead, has_unsafe_rank  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_dsl_compat import fe_only_operators, is_lqtp_native_dsl  # noqa: E402
from scripts.cogalpha_lqtp.python_to_dsl import MANUAL_DSL, _normalize_fe_compat_ops  # noqa: E402
from scripts.cogalpha_lqtp.run_production_batch import LOOKAHEAD_DEFERRED_FACTORS  # noqa: E402

WORK = ROOT / "data/cogalpha_lqtp_production"
BATCH_PARSED = ROOT / "data/cogalpha_lqtp_batch/parsed_factors.json"

# groupby / cumcount — keep Python lake; DSL not reliable
PYTHON_ONLY_KEEP = {
    "factor_drawdown_volume_complexity",
    "factor_drawdown_volume_geometry",
    "factor_drawdown_volume_geometry_rolling",
    "factor_drawdown_volume_modulated",
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _stamp_backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, path.with_name(f"{path.stem}_pre_complete_{stamp}{path.suffix}"))


def fix_and_collect_lookahead() -> dict[str, dict]:
    """Return name -> {factor_id, function_name, python_code} with causal ranks."""
    batch = _load(BATCH_PARSED)
    by = {r["function_name"]: r for r in batch}
    out: dict[str, dict] = {}
    for name in LOOKAHEAD_DEFERRED_FACTORS:
        rec = by.get(name)
        if not rec:
            print(f"WARN missing in batch parsed: {name}")
            continue
        code = rec.get("python_code") or ""
        fixed = fix_rank_lookahead(code) if has_unsafe_rank(code) else code
        # expanding().max drawdown: also prefer rolling 252 when expanding max + full rank
        out[name] = {
            "factor_id": rec.get("factor_id") or name,
            "function_name": name,
            "python_code": fixed,
            "tools": rec.get("tools", ""),
        }
        if fixed != code:
            print(f"fixed rank lookahead: {name}")
        else:
            print(f"lookahead code ok/no-unsafe-rank: {name}")
    return out


def inject_into_screening(la: dict[str, dict]) -> list[str]:
    parsed_path = WORK / "screening_reeval_parsed_factors.json"
    cat_path = WORK / "screening_reeval_catalog.json"
    _stamp_backup(parsed_path)
    _stamp_backup(cat_path)

    parsed = _load(parsed_path)
    pby = {r["function_name"]: r for r in parsed}
    for name, rec in la.items():
        if name in pby:
            pby[name]["python_code"] = rec["python_code"]
        else:
            parsed.append(
                {
                    "factor_id": rec["factor_id"],
                    "function_name": name,
                    "python_code": rec["python_code"],
                    "tools": rec.get("tools", ""),
                }
            )
            pby[name] = parsed[-1]
    _save(parsed_path, parsed)

    cat = _load(cat_path)
    cby = {e["function_name"]: e for e in cat}
    added: list[str] = []
    for name, rec in la.items():
        dsl = ""
        if name in MANUAL_DSL and name not in PYTHON_ONLY_KEEP:
            dsl = sanitize_dsl(_normalize_fe_compat_ops(MANUAL_DSL[name]))
        if name in cby:
            e = cby[name]
        else:
            e = {
                "factor_id": rec["factor_id"],
                "function_name": name,
                "manifest_source": "lookahead_deferred",
                "manifest_rank_ic": None,
                "manifest_display_name": name,
            }
            cat.append(e)
            cby[name] = e
            added.append(name)
        e["python_code"] = rec["python_code"]
        if dsl:
            e["dsl"] = dsl
            try:
                e["lqtp_formula"] = dsl_to_lqtp(dsl)
            except Exception:
                e["lqtp_formula"] = dsl
            fe_ops = fe_only_operators(dsl)
            e["fe_only_ops"] = ",".join(fe_ops) if fe_ops else ""
            hard = name in PYTHON_ONLY_KEEP or any(
                x in rec["python_code"] for x in ("groupby", "cumcount")
            )
            if hard:
                e["eval_route"] = "local_python"
                e["lqtp_native"] = False
                e["status"] = "ready"
                e["notes"] = "python_lake; dsl_display"
            else:
                native = is_lqtp_native_dsl(dsl) and not fe_ops
                e["lqtp_native"] = bool(native)
                e["eval_route"] = "lqtp_dsl" if native else "local_dsl"
                e["status"] = "ready"
                e["notes"] = "lookahead_fixed_manual_dsl"
        else:
            e["dsl"] = ""
            e["lqtp_formula"] = ""
            e["eval_route"] = "local_python"
            e["lqtp_native"] = False
            e["status"] = "ready"
            e["notes"] = "python_only_groupby; lake from fixed python"
    _save(cat_path, cat)
    print(f"catalog size={len(cat)} newly_added={added}")
    return sorted(la.keys())


def repair_python_only_dsl() -> dict[str, int]:
    """Apply MANUAL_DSL / EXTRA map to remaining empty-DSL catalog rows."""
    from scripts.cogalpha_lqtp.repair_formulas_and_homepage import looks_bad, paren_depth

    cat_path = WORK / "screening_reeval_catalog.json"
    prog_path = WORK / "screening_reeval_progress.json"
    cat = _load(cat_path)
    prog = _load(prog_path)
    flip = prog.setdefault("flip_state", {})
    stats = {"filled": 0, "still_empty": 0, "kept": 0}
    for e in cat:
        name = e["function_name"]
        if (e.get("dsl") or "").strip() and e.get("status") != "python_only":
            stats["kept"] += 1
            continue
        if name not in MANUAL_DSL:
            stats["still_empty"] += 1
            continue
        dsl = sanitize_dsl(_normalize_fe_compat_ops(MANUAL_DSL[name]))
        if looks_bad(dsl) or paren_depth(dsl) > 16:
            print(f"WARN manual still bad: {name} {looks_bad(dsl)} depth={paren_depth(dsl)}")
            stats["still_empty"] += 1
            continue
        flipped = bool((flip.get(name) or {}).get("ic_sign_flipped"))
        show = dsl
        if flipped and not show.lstrip().startswith("-"):
            from scripts.cogalpha_lqtp.run_production_batch import _negate_formula

            show = _negate_formula(show)
        e["dsl"] = show
        try:
            e["lqtp_formula"] = dsl_to_lqtp(show)
        except Exception:
            e["lqtp_formula"] = show
        fe_ops = fe_only_operators(show)
        e["fe_only_ops"] = ",".join(fe_ops) if fe_ops else ""
        # lake already from python for these; keep route python if style/tools hard
        py = e.get("python_code") or ""
        hard = any(x in py for x in ("groupby", "cumcount"))
        if hard:
            e["eval_route"] = "local_python"
            e["lqtp_native"] = False
        else:
            native = is_lqtp_native_dsl(show) and not fe_ops
            e["lqtp_native"] = bool(native)
            # display DSL ready; values remain lake python unless rematerialized
            e["eval_route"] = e.get("eval_route") or ("lqtp_dsl" if native else "local_dsl")
            if e["eval_route"] == "local_python":
                pass  # keep python lake
        e["status"] = "ready"
        e["notes"] = "manual_dsl_filled"
        flip[name] = {**(flip.get(name) or {}), "dsl": show, "ic_sign_flipped": flipped}
        stats["filled"] += 1
    _save(cat_path, cat)
    prog["flip_state"] = flip
    prog["formula_repair_at"] = datetime.now().isoformat()
    prog["note"] = "complete_remaining: dsl filled + lookahead injected"
    _save(prog_path, prog)
    print("repair_python_only_dsl", stats)
    return stats


def clear_skipped_and_invalidate(names: list[str]) -> None:
    prog_path = WORK / "screening_reeval_progress.json"
    prog = _load(prog_path)
    skipped = [x for x in (prog.get("skipped") or []) if x not in names]
    prog["skipped"] = skipped
    # drop old index rows so --only re-eval writes fresh
    prog["index_rows"] = [r for r in prog.get("index_rows", []) if r.get("factor_name") not in names]
    completed = [x for x in prog.get("completed", []) if x not in names]
    prog["completed"] = completed
    for n in names:
        (prog.get("failed") or {}).pop(n, None)
    _save(prog_path, prog)

    mat_path = WORK / "screening_reeval_materialize.json"
    if mat_path.exists():
        mat = _load(mat_path)
        mat["completed"] = [x for x in mat.get("completed", []) if x not in names]
        for n in names:
            (mat.get("failed") or {}).pop(n, None)
        _save(mat_path, mat)

    lake = WORK / "factor_lake"
    for n in names:
        pq = lake / n / "values.parquet"
        if pq.exists():
            pq.unlink()
            print(f"invalidated lake {n}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--python-workers", type=int, default=3)
    args = parser.parse_args()

    la = fix_and_collect_lookahead()
    names = inject_into_screening(la)
    repair_python_only_dsl()
    clear_skipped_and_invalidate(names)

    if args.prepare_only:
        print("prepare-only done", names)
        return 0

    cmd = [
        sys.executable,
        "-m",
        "scripts.cogalpha_lqtp.run_screening_reeval_batch",
        "--work-dir",
        str(WORK),
        "--force-materialize",
        "--no-skip-lookahead",
        "--workers",
        str(args.workers),
        "--python-workers",
        str(args.python_workers),
        "--only",
        *names,
    ]
    print("RUN", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc

    # refresh formulas on reports + homepage/corr for full set
    subprocess.call([sys.executable, "-m", "scripts.cogalpha_lqtp.repair_formulas_and_homepage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
