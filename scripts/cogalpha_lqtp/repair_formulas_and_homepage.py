#!/usr/bin/env python3
"""Re-apply DSL formula repair + build corr matrix + refresh homepage."""
from __future__ import annotations

import html as html_lib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FE = ROOT / "factor_engine"
if str(FE) not in sys.path:
    sys.path.insert(0, str(FE))

from scripts.cogalpha_lqtp.ast_translator import dsl_to_lqtp, translate_python  # noqa: E402
from scripts.cogalpha_lqtp.dsl_sanitize import sanitize_dsl, simplify_redundant_parens  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_dsl_compat import fe_only_operators, is_lqtp_native_dsl  # noqa: E402
from scripts.cogalpha_lqtp.python_to_dsl import MANUAL_DSL, _normalize_fe_compat_ops  # noqa: E402
from scripts.cogalpha_lqtp.reconcile_ic_signs import _unwrap_formula  # noqa: E402
from scripts.cogalpha_lqtp.run_production_batch import _negate_formula  # noqa: E402

WORK = ROOT / "data/cogalpha_lqtp_production"


def looks_bad(dsl: str) -> list[str]:
    bad: list[str] = []
    if not dsl:
        return ["empty"]
    if dsl.count("(") != dsl.count(")"):
        bad.append("unbalanced")
    if '== "high"' in dsl or "== 'high'" in dsl:
        bad.append("string_cmp")
    if any(x in dsl for x in (".ts_", "std(ddof", ".rolling", ".ewm(", "pct_change", "close.delay")):
        bad.append("py_leak")
    if re.search(r"\)\s*cap\(|\)\s*clip\(", dsl):
        bad.append("concat_call")
    if "safe_div(open)" in dsl or "protected_div(open)" in dsl:
        bad.append("bad_arity")
    depth = paren_depth(dsl)
    if depth >= 14:
        bad.append("paren_depth")
    if "(((((" in dsl:
        bad.append("paren_bloat")
    return bad


def paren_depth(s: str) -> int:
    d = m = 0
    for ch in s:
        if ch == "(":
            d += 1
            m = max(m, d)
        elif ch == ")":
            d -= 1
    return m


def repair_formulas() -> dict:
    cat_path = WORK / "screening_reeval_catalog.json"
    prog_path = WORK / "screening_reeval_progress.json"
    parsed = {
        r["function_name"]: r.get("python_code", "")
        for r in json.loads((WORK / "screening_reeval_parsed_factors.json").read_text(encoding="utf-8"))
    }
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    prog = json.loads(prog_path.read_text(encoding="utf-8"))
    flip = prog.setdefault("flip_state", {})

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(cat_path, WORK / f"screening_reeval_catalog_pre_formula_repair2_{stamp}.json")
    shutil.copy2(prog_path, WORK / f"screening_reeval_progress_pre_formula_repair2_{stamp}.json")

    stats = {"manual": 0, "auto_ok": 0, "python_only": 0, "keep_old": 0, "reports": 0}

    for e in cat:
        name = e["function_name"]
        py = re.sub(r"^# IC_SIGN_FLIPPED.*\n", "", parsed.get(name) or "")
        old = (e.get("dsl") or "").strip()
        if name in MANUAL_DSL:
            new_dsl = sanitize_dsl(_normalize_fe_compat_ops(MANUAL_DSL[name]))
            source = "manual"
            stats["manual"] += 1
        else:
            tr = translate_python(py) if py else None
            cand = (
                sanitize_dsl(_normalize_fe_compat_ops(tr.dsl))
                if tr and tr.status == "ready" and tr.dsl
                else ""
            )
            if cand and not looks_bad(cand) and paren_depth(cand) <= 12:
                new_dsl = cand
                source = "auto"
                stats["auto_ok"] += 1
            else:
                cleaned = simplify_redundant_parens(old) if old else ""
                if cleaned and not looks_bad(cleaned) and paren_depth(cleaned) < 14:
                    new_dsl = cleaned
                    source = "keep"
                    stats["keep_old"] += 1
                else:
                    new_dsl = ""
                    source = "python_only"
                    stats["python_only"] += 1

        st = flip.get(name) or {}
        flipped = bool(st.get("ic_sign_flipped"))
        if new_dsl:
            unsigned = _unwrap_formula(new_dsl)
            if flipped and not unsigned.lstrip().startswith("-"):
                show = _negate_formula(unsigned)
            elif flipped and unsigned.lstrip().startswith("-"):
                show = unsigned
            else:
                show = new_dsl
            e["dsl"] = show
            try:
                e["lqtp_formula"] = dsl_to_lqtp(show)
            except Exception:
                e["lqtp_formula"] = show
            fe_ops = fe_only_operators(show)
            e["fe_only_ops"] = ",".join(fe_ops) if fe_ops else ""
            hard = any(
                x in py for x in ("groupby", "alpha_tools", "style_gate", "classify_volume_regime")
            )
            if hard and source != "manual":
                e["eval_route"] = "local_python"
                e["lqtp_native"] = False
            else:
                native = is_lqtp_native_dsl(show) and not fe_ops
                e["lqtp_native"] = bool(native)
                e["eval_route"] = "lqtp_dsl" if native else "local_dsl"
            e["status"] = "ready"
            if name in flip:
                flip[name] = {**st, "dsl": show}
            else:
                flip[name] = {"dsl": show, "ic_sign_flipped": flipped}
        else:
            e["dsl"] = ""
            e["lqtp_formula"] = ""
            e["eval_route"] = "local_python"
            e["lqtp_native"] = False
            e["status"] = "python_only"
            e["notes"] = "auto DSL unreliable; lake values from python"
            if name in flip:
                st2 = dict(flip[name])
                st2["dsl"] = ""
                flip[name] = st2

    cat_path.write_text(json.dumps(cat, ensure_ascii=False, indent=2), encoding="utf-8")
    prog["flip_state"] = flip
    prog["formula_repair_at"] = datetime.now().isoformat()
    prog["formula_repair_stats"] = stats
    prog["note"] = "realto complete; formulas repaired post-eval"
    prog_path.write_text(json.dumps(prog, ensure_ascii=False, indent=2), encoding="utf-8")

    report_dir = WORK / "reports_screening_reeval"
    patched = 0
    for e in cat:
        name = e["function_name"]
        dsl = (e.get("dsl") or "").strip()
        rp = report_dir / f"{name}.html"
        if not dsl or not rp.is_file():
            continue
        text = rp.read_text(encoding="utf-8", errors="ignore")
        esc = html_lib.escape(dsl)
        new, n = re.subn(
            r"(<h3[^>]*>\s*DSL[\s\S]*?</h3>\s*<pre>)([\s\S]*?)(</pre>)",
            lambda m: m.group(1) + esc + m.group(3),
            text,
            count=1,
            flags=re.I,
        )
        if n == 0:
            new, n = re.subn(
                r"(公式[\s\S]{0,400}?<pre>)([\s\S]*?)(</pre>)",
                lambda m: m.group(1) + esc + m.group(3),
                text,
                count=1,
            )
        if n:
            if e.get("eval_route") == "local_python":
                new = new.replace("LQTP 原生 DSL 直算", "纯 Python 落值")
            rp.write_text(new, encoding="utf-8")
            patched += 1
    stats["reports"] = patched

    bad = []
    for e in cat:
        d = e.get("dsl") or ""
        if not d:
            bad.append((e["function_name"], ["EMPTY"]))
            continue
        b = looks_bad(d)
        if b:
            bad.append((e["function_name"], b))
    return {"stats": stats, "remaining_bad": bad, "catalog": len(cat)}


def main() -> int:
    print("=== repair formulas ===")
    out = repair_formulas()
    print("stats", out["stats"])
    print("remaining_bad", len(out["remaining_bad"]))
    for name, b in out["remaining_bad"]:
        print(f"  {name}: {b}")

    cat = json.loads((WORK / "screening_reeval_catalog.json").read_text(encoding="utf-8"))
    for name in [
        "factor_vol_weighted_pct_range_clip_ema10",
        "factor_crossover_herding_stress_span12_volrank_clip3",
        "factor_volume_confirmed_momentum_clipped",
        "factor_intraday_volume_tanh",
    ]:
        e = next(x for x in cat if x["function_name"] == name)
        print(name, "=>", (e.get("dsl") or "")[:200])

    print("=== corr matrix ===")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "scripts.cogalpha_lqtp.compute_factor_corr_matrix",
            "--work-dir",
            str(WORK),
            "--sample-days",
            "40",
            "--min-ic",
            "0.02",
        ],
        cwd=str(ROOT),
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": f"factor_engine:{ROOT}"},
    )

    print("=== homepage ===")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "scripts.cogalpha_lqtp.render_rankic_screening_index",
            "--work-dir",
            str(WORK),
            "--threshold",
            "0.02",
            "--corr-sample-days",
            "40",
        ],
        cwd=str(ROOT),
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": f"factor_engine:{ROOT}"},
    )

    html = (WORK / "reports/factor_rankic_screening_index.html").read_text(encoding="utf-8")
    print("has corr embed", "corr-embed" in html or "秩相关" in html)
    print("has corr table", "table class=\"corr\"" in html or "table class='corr'" in html or "<table class=\"corr\"" in html)
    m = re.search(r"<tbody>\s*<tr>(.*?)</tr>", html, re.S)
    if m:
        cells = re.findall(r"<td>(.*?)</td>", m.group(1))
        print("row1 cells", [re.sub("<[^>]+>", "", c)[:40] for c in cells])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
