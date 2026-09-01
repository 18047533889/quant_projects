#!/usr/bin/env python3
"""factor_engine 更新后第三批：pivot/支撑阻力/摆动/剩余形态等。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT.parent / "factor_engine"), str(ROOT.parent / "data_access")):
    if p not in sys.path:
        sys.path.insert(0, p)

W = (20, 40, 60)


def _add(bag, *, fid, expr, topic, desc, family):
    bag.append(
        {
            "factor_id": fid,
            "expr": " ".join(expr.split()),
            "topic": topic,
            "description": desc,
            "family": family,
            "source_library": "fe_update_v4",
        }
    )


def generate():
    out = []
    n = 0

    def nid(p):
        nonlocal n
        n += 1
        return f"fe_v4_{p}_{n:05d}"

    for w in W:
        _add(out, fid=nid("ts"), expr=f"rank(ts_average_volume(volume, {w}))", topic="V9/fe_v4_ts", desc=f"avg_vol({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_consolidation_width(close, {w}))", topic="V9/fe_v4_ts", desc=f"consol_w({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_consolidation_slope(close, {w}))", topic="V9/fe_v4_ts", desc=f"consol_slope({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_consolidation_volume_decay(close, volume, {w}))", topic="V9/fe_v4_ts", desc=f"consol_vd({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_ewm_var(close, {w}, 0.1))", topic="V9/fe_v4_ts", desc=f"ewm_var({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_ewm_corr(close, volume, {w}, 0.1))", topic="V9/fe_v4_ts", desc=f"ewm_corr({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_ewm_cov(close, volume, {w}, 0.1))", topic="V9/fe_v4_ts", desc=f"ewm_cov({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_max_buildup(volume, {w}))", topic="V9/fe_v4_ts", desc=f"max_buildup({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_moment(close, {w}, 3))", topic="V9/fe_v4_ts", desc=f"moment3({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_nth_value(close, {w}, 2))", topic="V9/fe_v4_ts", desc=f"nth2({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_pattern_symmetry(close, {w}))", topic="V9/fe_v4_ts", desc=f"symmetry({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_poly2_resid(close, {w}))", topic="V9/fe_v4_ts", desc=f"poly2_resid({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_poly2_coeff(close, {w}, 1))", topic="V9/fe_v4_ts", desc=f"poly2_c1({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(safe_div_null(subtract(close, ts_prev_high(close, {w})), close))", topic="V9/fe_v4_ts", desc=f"vs prev_high", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(safe_div_null(subtract(close, ts_prev_low(close, {w})), close))", topic="V9/fe_v4_ts", desc=f"vs prev_low", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_sma_cn(close, {w}))", topic="V9/fe_v4_ts", desc=f"sma_cn({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(roll_spread_proxy(close, {w}))", topic="V9/fe_v4_ts", desc=f"roll_spread({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(rank_corr(ts_pct(close, 1), ts_pct(volume, 1), {w}))", topic="V9/fe_v4_ts", desc=f"rank_corr({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(Covariance(close, volume, {w}))", topic="V9/fe_v4_ts", desc=f"Covariance({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_mean_if(gt(close, ts_delay(close, 1)), close, {w}))", topic="V9/fe_v4_ts", desc=f"mean_if up({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_std_if(gt(close, ts_delay(close, 1)), close, {w}))", topic="V9/fe_v4_ts", desc=f"std_if up({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_last_if(gt(close, ts_delay(close, 1)), close, {w}))", topic="V9/fe_v4_ts", desc=f"last_if({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_line_convergence(close, {w}))", topic="V9/fe_v4_ts", desc=f"line_conv({w})", family="fe_v4_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_line_parallelism(close, {w}))", topic="V9/fe_v4_ts", desc=f"line_par({w})", family="fe_v4_ts")

        _add(out, fid=nid("sr"), expr=f"rank(safe_div_null(subtract(close, ts_resistance_level(close, {w})), close))", topic="V9/fe_v4_sr", desc=f"vs resist", family="fe_v4_sr")
        _add(out, fid=nid("sr"), expr=f"rank(safe_div_null(subtract(close, ts_support_level(close, {w})), close))", topic="V9/fe_v4_sr", desc=f"vs support", family="fe_v4_sr")
        _add(out, fid=nid("sr"), expr=f"rank(ts_resistance_slope(close, {w}))", topic="V9/fe_v4_sr", desc=f"res_slope({w})", family="fe_v4_sr")
        _add(out, fid=nid("sr"), expr=f"rank(ts_support_slope(close, {w}))", topic="V9/fe_v4_sr", desc=f"sup_slope({w})", family="fe_v4_sr")
        _add(out, fid=nid("sr"), expr=f"rank(ts_resistance_fit_r2(close, {w}))", topic="V9/fe_v4_sr", desc=f"res_r2({w})", family="fe_v4_sr")
        _add(out, fid=nid("sr"), expr=f"rank(ts_support_fit_r2(close, {w}))", topic="V9/fe_v4_sr", desc=f"sup_r2({w})", family="fe_v4_sr")
        _add(out, fid=nid("sr"), expr=f"rank(ts_resistance_break(close, {w}))", topic="V9/fe_v4_sr", desc=f"res_break({w})", family="fe_v4_sr")
        _add(out, fid=nid("sr"), expr=f"rank(ts_support_break(close, {w}))", topic="V9/fe_v4_sr", desc=f"sup_break({w})", family="fe_v4_sr")
        _add(out, fid=nid("sw"), expr=f"rank(ts_swing_amplitude(close, {w}))", topic="V9/fe_v4_sw", desc=f"swing_amp({w})", family="fe_v4_sw")
        _add(out, fid=nid("sw"), expr=f"rank(ts_swing_amplitude_pct(close, {w}))", topic="V9/fe_v4_sw", desc=f"swing_amp_pct({w})", family="fe_v4_sw")
        _add(out, fid=nid("sw"), expr=f"rank(ts_swing_amplitude_atr(close, high, low, {w}))", topic="V9/fe_v4_sw", desc=f"swing_amp_atr({w})", family="fe_v4_sw")
        _add(out, fid=nid("sw"), expr=f"rank(ts_swing_duration(close, {w}))", topic="V9/fe_v4_sw", desc=f"swing_dur({w})", family="fe_v4_sw")
        _add(out, fid=nid("sw"), expr=f"rank(ts_swing_velocity(close, {w}))", topic="V9/fe_v4_sw", desc=f"swing_vel({w})", family="fe_v4_sw")
        _add(out, fid=nid("reg"), expr=f"rank(ts_regression_intercept(close, ts_mean(close, 20), {w}))", topic="V9/fe_v4_reg", desc=f"reg_int({w})", family="fe_v4_reg")
        _add(out, fid=nid("reg"), expr=f"rank(ts_regression_tstat(close, ts_mean(close, 20), {w}))", topic="V9/fe_v4_reg", desc=f"reg_t({w})", family="fe_v4_reg")

    for left, right in ((3, 3), (5, 5), (5, 10)):
        _add(out, fid=nid("pv"), expr=f"rank(ts_confirmed_pivot_high(close, {left}, {right}))", topic="V9/fe_v4_pv", desc=f"pivot_hi({left},{right})", family="fe_v4_pv")
        _add(out, fid=nid("pv"), expr=f"rank(ts_confirmed_pivot_low(close, {left}, {right}))", topic="V9/fe_v4_pv", desc=f"pivot_lo({left},{right})", family="fe_v4_pv")
        _add(out, fid=nid("pv"), expr=f"rank(safe_div_null(subtract(close, ts_last_pivot_high(close, {left})), close))", topic="V9/fe_v4_pv", desc="vs last_pivot_hi", family="fe_v4_pv")
        _add(out, fid=nid("pv"), expr=f"rank(safe_div_null(subtract(close, ts_last_pivot_low(close, {left})), close))", topic="V9/fe_v4_pv", desc="vs last_pivot_lo", family="fe_v4_pv")
        _add(out, fid=nid("pv"), expr=f"rank(ts_pivot_high_age(close, {left}))", topic="V9/fe_v4_pv", desc=f"pivot_hi_age({left})", family="fe_v4_pv")
        _add(out, fid=nid("pv"), expr=f"rank(ts_pivot_low_age(close, {left}))", topic="V9/fe_v4_pv", desc=f"pivot_lo_age({left})", family="fe_v4_pv")
        _add(out, fid=nid("pv"), expr=f"rank(ts_nth_pivot_high(close, {left}, 1))", topic="V9/fe_v4_pv", desc=f"nth_pivot_hi", family="fe_v4_pv")
        _add(out, fid=nid("pv"), expr=f"rank(ts_nth_pivot_low(close, {left}, 1))", topic="V9/fe_v4_pv", desc=f"nth_pivot_lo", family="fe_v4_pv")
        _add(out, fid=nid("pv"), expr=f"rank(ts_nth_pivot_high_age(close, {left}, 1))", topic="V9/fe_v4_pv", desc="nth_pivot_hi_age", family="fe_v4_pv")
        _add(out, fid=nid("pv"), expr=f"rank(ts_nth_pivot_low_age(close, {left}, 1))", topic="V9/fe_v4_pv", desc="nth_pivot_lo_age", family="fe_v4_pv")

    for w, left in ((40, 5), (60, 5)):
        _add(out, fid=nid("pv"), expr=f"rank(ts_pivot_high_count(close, {w}, {left}))", topic="V9/fe_v4_pv", desc=f"pivot_hi_cnt", family="fe_v4_pv")
        _add(out, fid=nid("pv"), expr=f"rank(ts_pivot_low_count(close, {w}, {left}))", topic="V9/fe_v4_pv", desc=f"pivot_lo_cnt", family="fe_v4_pv")
        _add(out, fid=nid("pv"), expr=f"rank(ts_pivot_high_spacing(close, {w}, {left}))", topic="V9/fe_v4_pv", desc=f"pivot_hi_sp", family="fe_v4_pv")
        _add(out, fid=nid("pv"), expr=f"rank(ts_pivot_low_spacing(close, {w}, {left}))", topic="V9/fe_v4_pv", desc=f"pivot_lo_sp", family="fe_v4_pv")

    for n_ in (3, 5, 10):
        _add(out, fid=nid("imp"), expr=f"rank(ts_impulse_return(close, {n_}))", topic="V9/fe_v4_imp", desc=f"impulse_ret({n_})", family="fe_v4_imp")
        _add(out, fid=nid("imp"), expr=f"rank(ts_impulse_volume(volume, {n_}))", topic="V9/fe_v4_imp", desc=f"impulse_vol({n_})", family="fe_v4_imp")
        _add(out, fid=nid("imp"), expr=f"rank(ts_impulse_strength(close, volume, {n_}))", topic="V9/fe_v4_imp", desc=f"impulse_str({n_})", family="fe_v4_imp")

    _add(out, fid=nid("cdl"), expr=f"rank(candle_gap(open, pre_close))", topic="V9/fe_v4_cdl", desc="candle_gap", family="fe_v4_cdl")
    _add(out, fid=nid("cdl"), expr=f"rank(safe_div_null(candle_gap(open, pre_close), pre_close))", topic="V9/fe_v4_cdl", desc="candle_gap/pre", family="fe_v4_cdl")

    for name in (
        "pattern_triple_bottom",
        "pattern_triple_top",
        "pattern_bull_pennant",
        "pattern_bear_pennant",
        "pattern_rising_channel",
        "pattern_falling_channel",
        "pattern_rectangle",
        "pattern_breakout_retest",
        "pattern_breakdown_retest",
        "pattern_cup",
        "pattern_broadening",
    ):
        for w in (40, 60):
            _add(out, fid=nid("pat"), expr=f"rank({name}(close, {w}))", topic="V9/fe_v4_pat", desc=f"{name}({w})", family="fe_v4_pat")

    seen = set()
    uniq = []
    for item in out:
        if item["expr"] in seen:
            continue
        seen.add(item["expr"])
        uniq.append(item)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    from api.dsl_parser import parse_expr
    from cold_start_library.runtime.dsl import extract_dsl_operator_names

    gen_path = ROOT / "data" / "ashare" / "expand_fe_update_v4.json"
    yaml_path = ROOT / "data" / "ashare" / "backend_v9_core.yaml"
    factors = generate()
    parse_ok, parse_fail = [], []
    for item in factors:
        try:
            parse_expr(item["expr"], surface="compat")
            item = dict(item)
            item["operators"] = list(extract_dsl_operator_names(item["expr"]))
            parse_ok.append(item)
        except Exception as exc:
            parse_fail.append({"factor_id": item["factor_id"], "expr": item["expr"], "error": f"{type(exc).__name__}: {exc}"})
    print(f"[gen] {len(factors)} unique, parse_ok={len(parse_ok)} fail={len(parse_fail)}", flush=True)
    if parse_fail[:12]:
        print(json.dumps(parse_fail[:12], ensure_ascii=False, indent=2))
    gen_path.write_text(json.dumps({"count": len(parse_ok), "items": parse_ok, "parse_fail": parse_fail}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.generate_only:
        return 0 if parse_ok else 1

    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    entries = list(payload.get("entries") or [])
    old_exprs = {" ".join(str(e.get("expr") or "").split()) for e in entries}
    old_ids = {str(e.get("factor_id")) for e in entries}
    added = 0
    for item in parse_ok:
        if item["expr"] in old_exprs or item["factor_id"] in old_ids:
            continue
        entries.append(
            {
                "factor_id": item["factor_id"],
                "expr": item["expr"],
                "topic": item["topic"],
                "description": item["description"],
                "operators": item.get("operators") or [],
                "family": item.get("family"),
                "source_library": item.get("source_library"),
            }
        )
        added += 1
    payload["entries"] = entries
    payload["entry_count"] = len(entries)
    payload["source"] = str(payload.get("source") or "") + "; +fe_update_v4"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False, width=120)
    print(f"[merge] added={added} total={len(entries)} -> {yaml_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
