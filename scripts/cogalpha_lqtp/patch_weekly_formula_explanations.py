#!/usr/bin/env python3
"""Add Chinese formula explanations to weekly-dug detail HTML + index table."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "cogalpha_lqtp")]

from scripts.cogalpha_lqtp.crawl_candidate_pool_neutral_rankic import (  # noqa: E402
    display_factor_name,
    display_lqtp_formula,
    patch_screening_html,
)
from scripts.cogalpha_lqtp.factor_report_zh import (  # noqa: E402
    explain_dsl_zh,
    render_interpretation_block,
)

WORK = ROOT / "data/cogalpha_lqtp_production"


def _patch_detail_html(path: Path, *, name: str, dsl: str, guide: dict) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    block = render_interpretation_block(
        name,
        ann={
            "theme": guide.get("theme"),
            "summary": guide.get("summary"),
            "steps": guide.get("steps"),
            "direction": guide.get("direction"),
            "operators": guide.get("operators"),
            "formula_display": dsl,
        },
        dsl=dsl,
    ).strip()
    pat = re.compile(
        r'<div class="card">\s*<h2>因子说明</h2>[\s\S]*?</div>\s*(?=<div class="card">|<h2>|$)',
        re.M,
    )
    if pat.search(text):
        new_text, n = pat.subn(block + "\n", text, count=1)
        if n:
            # scrub mining brand leftovers if any
            for brand in ("evoalpha", "EvoAlpha", "alphasage", "AlphaSage", "cogalpha", "CogAlpha"):
                new_text = new_text.replace(brand, "cand")
            path.write_text(new_text, encoding="utf-8")
            return True
    m = re.search(r"(<h2>公式\s*/\s*代码</h2>[\s\S]*?</div>)", text)
    if m:
        insert_at = m.end()
        new_text = text[:insert_at] + "\n" + block + "\n" + text[insert_at:]
        path.write_text(new_text, encoding="utf-8")
        return True
    if "</body>" in text:
        path.write_text(text.replace("</body>", block + "\n</body>", 1), encoding="utf-8")
        return True
    return False


def main() -> None:
    wp = WORK / "reports/weekly_dug_neutral_rankic.json"
    weekly = json.loads(wp.read_text(encoding="utf-8"))
    selected = list(weekly.get("selected") or [])
    ok = fail = generic = 0
    for r in selected:
        name = display_factor_name(r)
        dsl = display_lqtp_formula(r)
        guide = explain_dsl_zh(dsl)
        if not guide:
            generic += 1
            guide = {
                "theme": "量价因子",
                "summary": f"公式：{dsl}",
                "steps": [f"按公式计算：{dsl}", "每个交易日仅使用历史数据"],
                "direction": "结合 RankIC 方向理解因子值高低。",
                "operators": [],
            }
        r["formula_theme"] = guide.get("theme")
        r["formula_summary"] = guide.get("summary")
        r["formula_steps"] = guide.get("steps")
        r["formula_direction"] = guide.get("direction")
        r["formula_display"] = dsl

        path = WORK / "reports_weekly_dug" / f"{name}.html"
        if not path.exists():
            fail += 1
            print("MISSING", name, flush=True)
            continue
        if _patch_detail_html(path, name=name, dsl=dsl, guide=guide):
            ok += 1
        else:
            fail += 1
            print("FAIL_PATCH", name, flush=True)

    weekly["selected"] = selected
    weekly["formula_explanations_patched"] = True
    wp.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")

    class A:
        work_dir = WORK
        html = WORK / "reports/factor_rankic_screening_index.html"
        threshold = float(weekly.get("threshold") or 0.02)

    patch_screening_html(A(), payload=weekly)
    # sanity: sample a few explanations
    samples = []
    for r in selected[:3]:
        samples.append((r.get("display_name"), r.get("formula_theme"), (r.get("formula_summary") or "")[:60]))
    print(
        {"detail_ok": ok, "detail_fail": fail, "fallback_generic": generic, "n": len(selected), "samples": samples},
        flush=True,
    )


if __name__ == "__main__":
    main()
