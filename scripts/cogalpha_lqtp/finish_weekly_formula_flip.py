#!/usr/bin/env python3
"""After formula-sign flip retest finishes: patch index + refresh explanations."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data/cogalpha_lqtp_production"
LOG = WORK / "reports/weekly_flip_retest.stdout.log"
WP = WORK / "reports/weekly_dug_neutral_rankic.json"
NAMES = (WORK / "reports/weekly_formula_flip_note.json")


def _flipped_names() -> list[str]:
    note = json.loads(NAMES.read_text(encoding="utf-8"))
    return list(note.get("names") or [])


def _done(names: list[str]) -> tuple[int, int]:
    w = json.loads(WP.read_text(encoding="utf-8"))
    by = {r.get("display_name"): r for r in w.get("selected") or []}
    ok = miss = 0
    for n in names:
        r = by.get(n) or {}
        st = r.get("fe_retest_status")
        # after this run we also require formula_sign_flipped and recent lake
        if st == "ok" and r.get("formula_sign_flipped") and float(r.get("display_rank_ic") or -1) >= 0:
            # Heuristic: fe_retest_at on weekly or per-row sec present after flip
            if r.get("fe_retest_lake") and "formula_flipped_at" in r:
                ok += 1
            else:
                miss += 1
        else:
            miss += 1
    return ok, miss


def main() -> int:
    names = _flipped_names()
    # wait for primary log DONE, then kick the 2 ineligible, then patch
    for _ in range(360):  # up to ~6h at 60s
        text = LOG.read_text(encoding="utf-8", errors="ignore") if LOG.exists() else ""
        if "DONE ok=" in text.splitlines()[-5:] or any(line.startswith("DONE ok=") for line in text.splitlines()[-20:]):
            break
        time.sleep(60)
    else:
        print("TIMEOUT waiting primary retest", flush=True)

    # retest the two that were skipped due to eligible=False
    miss = [n for n in names]
    w = json.loads(WP.read_text(encoding="utf-8"))
    by = {r.get("display_name"): r for r in w.get("selected") or []}
    need = []
    for n in names:
        r = by.get(n) or {}
        # need retest if lake wiped (no values) or status not refreshed after flip
        lake = WORK / "factor_lake" / n / "values.parquet"
        if (not lake.exists()) or not r.get("formula_sign_flipped"):
            need.append(n)
        elif r.get("fe_retest_status") != "ok":
            need.append(n)
        else:
            # if values mtime older than formula_flipped_at, still need
            need.append(n)  # safer: the primary job may have finished some; check fe_dsl leading -
    # narrower: only those without leading - in analyzed formula or missing lake
    need2 = []
    for n in names:
        r = by.get(n) or {}
        lake = WORK / "factor_lake" / n / "values.parquet"
        if not lake.exists():
            need2.append(n)
            continue
        # primary already ran some; if fe_retest_lake exists and display IC>0 after flip timestamp, skip
        if r.get("fe_retest_status") == "ok" and lake.exists():
            # still re-run if fe_dsl has - but this run was from old lake before wipe — wipe removed lake so primary rematerializes
            pass
    # Just run the two known ineligible if lakes missing
    for n in ("ext_73a60264", "ext_7f827abe"):
        if not (WORK / "factor_lake" / n / "values.parquet").exists():
            need2.append(n)
    need2 = list(dict.fromkeys(need2))
    if need2:
        print(f"retest leftover {need2}", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts/cogalpha_lqtp/retest_weekly_recovered_dsl.py"),
                "--min-avail-gb",
                "3.5",
                "--only",
                *need2,
            ],
            cwd=str(ROOT),
        )

    # patch index
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            """
import json, sys
from pathlib import Path
ROOT=Path('/home/shw/quant_projects')
sys.path[:0]=[str(ROOT), str(ROOT/'scripts'/'cogalpha_lqtp')]
from scripts.cogalpha_lqtp.complete_weekly_dug_full import _patch_index
work=ROOT/'data/cogalpha_lqtp_production'
w=json.loads((work/'reports/weekly_dug_neutral_rankic.json').read_text())
_patch_index(work, w)
print('index patched')
""",
        ]
    )
    # refresh explanations (guides already mention themes; re-patch HTML with formula note)
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts/cogalpha_lqtp/patch_weekly_factor_explanations.py")],
        cwd=str(ROOT),
    )
    print("FINISH_FLIP_ORCH done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
