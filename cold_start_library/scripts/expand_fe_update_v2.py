#!/usr/bin/env python3
"""factor_engine 更新后：为尚未覆盖的 PV/技术算子扩充冷启动库。

- 生成 expand_fe_update_v2.json
- 合并进 backend_v9_core.yaml（不去掉已有条目）
- 可选：对新条目做短窗 IC 预计算

不中断已在跑的 full-span value_cache 任务时，可先 --merge-only；
之后对新增因子再跑 full_splits --skip-existing-cache。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FE = ROOT.parent / "factor_engine"
DA = ROOT.parent / "data_access"
for p in (str(SRC), str(FE), str(DA)):
    if p not in sys.path:
        sys.path.insert(0, p)

W = (5, 10, 14, 20, 40, 60, 120)
W2 = (10, 20, 60)
PAIRS = ((5, 60), (10, 120), (20, 60), (20, 250))


def _add(bag: list[dict[str, Any]], *, fid: str, expr: str, topic: str, desc: str, family: str) -> None:
    bag.append(
        {
            "factor_id": fid,
            "expr": " ".join(expr.split()),
            "topic": topic,
            "description": desc,
            "family": family,
            "source_library": "fe_update_v2",
        }
    )


def generate() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n = 0

    def nid(p: str) -> str:
        nonlocal n
        n += 1
        return f"fe_v2_{p}_{n:05d}"

    # --- moving averages / MACD family ---
    for w in W:
        _add(out, fid=nid("ma"), expr=f"rank(SMA(close, {w}))", topic="V9/fe_v2_ma", desc=f"SMA({w})", family="fe_v2_ma")
        _add(out, fid=nid("ma"), expr=f"rank(WMA(close, {w}))", topic="V9/fe_v2_ma", desc=f"WMA({w})", family="fe_v2_ma")
        _add(out, fid=nid("ma"), expr=f"rank(DEMA(close, {w}))", topic="V9/fe_v2_ma", desc=f"DEMA({w})", family="fe_v2_ma")
        _add(out, fid=nid("ma"), expr=f"rank(TEMA(close, {w}))", topic="V9/fe_v2_ma", desc=f"TEMA({w})", family="fe_v2_ma")
        _add(out, fid=nid("ma"), expr=f"rank(safe_div_null(subtract(close, SMA(close, {w})), close))", topic="V9/fe_v2_ma", desc=f"close/SMA-1 {w}", family="fe_v2_ma")
        _add(out, fid=nid("ma"), expr=f"rank(ewm_mean(close, {w}))", topic="V9/fe_v2_ma", desc=f"ewm_mean({w})", family="fe_v2_ma")
        _add(out, fid=nid("ma"), expr=f"rank(ewm_std((safe_div_null(close, pre_close) - 1.0), {w}))", topic="V9/fe_v2_ma", desc=f"ewm_std ret({w})", family="fe_v2_ma")
        _add(out, fid=nid("kama"), expr=f"rank(KAMA(close, {w}, 2, 30))", topic="V9/fe_v2_kama", desc=f"KAMA({w})", family="fe_v2_kama")
        _add(out, fid=nid("er"), expr=f"rank(efficiency_ratio(close, {w}))", topic="V9/fe_v2_er", desc=f"ER({w})", family="fe_v2_er")

    for fast, slow, sig in ((12, 26, 9), (8, 17, 9), (5, 35, 5)):
        _add(out, fid=nid("macd"), expr=f"rank(MACD(close, {fast}, {slow}, {sig}))", topic="V9/fe_v2_macd", desc=f"MACD({fast},{slow},{sig})", family="fe_v2_macd")
        _add(out, fid=nid("ppo"), expr=f"rank(PPO(close, {fast}, {slow}))", topic="V9/fe_v2_ppo", desc=f"PPO({fast},{slow})", family="fe_v2_ppo")
        _add(out, fid=nid("ppo"), expr=f"rank(PPO_hist(close, {fast}, {slow}, {sig}))", topic="V9/fe_v2_ppo", desc=f"PPO_hist", family="fe_v2_ppo")
        _add(out, fid=nid("ppo"), expr=f"rank(PPO_signal(close, {fast}, {slow}, {sig}))", topic="V9/fe_v2_ppo", desc=f"PPO_signal", family="fe_v2_ppo")
        _add(out, fid=nid("pvo"), expr=f"rank(PVO(volume, {fast}, {slow}))", topic="V9/fe_v2_pvo", desc=f"PVO vol", family="fe_v2_pvo")
        _add(out, fid=nid("pvo"), expr=f"rank(PVO_hist(volume, {fast}, {slow}, {sig}))", topic="V9/fe_v2_pvo", desc=f"PVO_hist", family="fe_v2_pvo")

    # --- oscillators / volume ---
    for n_ in (7, 14, 20, 28):
        _add(out, fid=nid("osc"), expr=f"rank(CMO(close, {n_}))", topic="V9/fe_v2_osc", desc=f"CMO({n_})", family="fe_v2_osc")
        _add(out, fid=nid("osc"), expr=f"rank(MFI(high, low, close, volume, {n_}))", topic="V9/fe_v2_osc", desc=f"MFI({n_})", family="fe_v2_osc")
        _add(out, fid=nid("osc"), expr=f"rank(CMF(high, low, close, volume, {n_}))", topic="V9/fe_v2_osc", desc=f"CMF({n_})", family="fe_v2_osc")
        _add(out, fid=nid("osc"), expr=f"rank(NATR(high, low, close, {n_}))", topic="V9/fe_v2_osc", desc=f"NATR({n_})", family="fe_v2_osc")
        _add(out, fid=nid("osc"), expr=f"rank(ForceIndex(close, volume, {n_}))", topic="V9/fe_v2_osc", desc=f"ForceIndex({n_})", family="fe_v2_osc")
        _add(out, fid=nid("osc"), expr=f"rank(EaseOfMovement(high, low, volume, {n_}))", topic="V9/fe_v2_osc", desc=f"EOM({n_})", family="fe_v2_osc")
        _add(out, fid=nid("osc"), expr=f"rank(choppiness_index(high, low, close, {n_}))", topic="V9/fe_v2_osc", desc=f"CHOP({n_})", family="fe_v2_osc")
        _add(out, fid=nid("dmi"), expr=f"rank(DMI_plus(high, low, close, {n_}))", topic="V9/fe_v2_dmi", desc=f"DMI+({n_})", family="fe_v2_dmi")
        _add(out, fid=nid("dmi"), expr=f"rank(DMI_minus(high, low, close, {n_}))", topic="V9/fe_v2_dmi", desc=f"DMI-({n_})", family="fe_v2_dmi")
        _add(out, fid=nid("dmi"), expr=f"rank(DX(high, low, close, {n_}))", topic="V9/fe_v2_dmi", desc=f"DX({n_})", family="fe_v2_dmi")
        _add(out, fid=nid("vtx"), expr=f"rank(VortexPlus(high, low, close, {n_}))", topic="V9/fe_v2_vtx", desc=f"Vortex+({n_})", family="fe_v2_vtx")
        _add(out, fid=nid("vtx"), expr=f"rank(VortexMinus(high, low, close, {n_}))", topic="V9/fe_v2_vtx", desc=f"Vortex-({n_})", family="fe_v2_vtx")
        _add(out, fid=nid("liq"), expr=f"rank(amihud_illiquidity(close, volume, {n_}))", topic="V9/fe_v2_liq", desc=f"amihud({n_})", family="fe_v2_liq")
        _add(out, fid=nid("liq"), expr=f"rank(abnormal_volume(volume, {n_}))", topic="V9/fe_v2_liq", desc=f"abn_vol({n_})", family="fe_v2_liq")
        _add(out, fid=nid("liq"), expr=f"rank(down_volume_ratio(close, volume, {n_}))", topic="V9/fe_v2_liq", desc=f"down_vol_ratio({n_})", family="fe_v2_liq")

    _add(out, fid=nid("adl"), expr=f"rank(ADL(high, low, close, volume))", topic="V9/fe_v2_adl", desc="ADL", family="fe_v2_adl")
    _add(out, fid=nid("adl"), expr=f"rank(ts_delta(ADL(high, low, close, volume), 5))", topic="V9/fe_v2_adl", desc="dADL5", family="fe_v2_adl")
    _add(out, fid=nid("adl"), expr=f"rank(ts_delta(ADL(high, low, close, volume), 20))", topic="V9/fe_v2_adl", desc="dADL20", family="fe_v2_adl")
    _add(out, fid=nid("chaikin"), expr=f"rank(ChaikinOscillator(high, low, close, volume, 3, 10))", topic="V9/fe_v2_chaikin", desc="ChaikinOsc", family="fe_v2_chaikin")
    _add(out, fid=nid("dv"), expr=f"rank(dollar_volume(close, volume))", topic="V9/fe_v2_dv", desc="dollar_volume", family="fe_v2_dv")
    _add(out, fid=nid("dv"), expr=f"rank(log(add(dollar_volume(close, volume), 1.0)))", topic="V9/fe_v2_dv", desc="log dollar_volume", family="fe_v2_dv")

    for r, s in ((25, 13), (40, 20)):
        _add(out, fid=nid("tsi"), expr=f"rank(TSI(close, {r}, {s}))", topic="V9/fe_v2_tsi", desc=f"TSI({r},{s})", family="fe_v2_tsi")
        _add(out, fid=nid("tsi"), expr=f"rank(TSI_signal(close, {r}, {s}, 7))", topic="V9/fe_v2_tsi", desc=f"TSI_signal", family="fe_v2_tsi")

    _add(out, fid=nid("uo"), expr=f"rank(UltimateOscillator(high, low, close, 7, 14, 28))", topic="V9/fe_v2_uo", desc="UO", family="fe_v2_uo")
    _add(out, fid=nid("psar"), expr=f"rank(safe_div_null(subtract(close, PSAR(high, low, 0.02, 0.2)), close))", topic="V9/fe_v2_psar", desc="close vs PSAR", family="fe_v2_psar")
    for per, mult in ((10, 3.0), (14, 2.0), (20, 3.0)):
        _add(out, fid=nid("st"), expr=f"rank(Supertrend(high, low, close, {per}, {mult}))", topic="V9/fe_v2_st", desc=f"Supertrend({per},{mult})", family="fe_v2_st")
        _add(out, fid=nid("st"), expr=f"rank(SupertrendDirection(high, low, close, {per}, {mult}))", topic="V9/fe_v2_st", desc=f"ST direction", family="fe_v2_st")

    # --- bollinger / keltner / donchian / channels ---
    for w in W2:
        _add(out, fid=nid("bb"), expr=f"rank(bollinger_pct_b(close, {w}, 2.0))", topic="V9/fe_v2_bb", desc=f"%b({w})", family="fe_v2_bb")
        _add(out, fid=nid("bb"), expr=f"rank(bollinger_width(close, {w}, 2.0))", topic="V9/fe_v2_bb", desc=f"bb_width({w})", family="fe_v2_bb")
        _add(out, fid=nid("kel"), expr=f"rank(KeltnerPosition(high, low, close, {w}, 10, 1.5))", topic="V9/fe_v2_kel", desc=f"KeltnerPos({w})", family="fe_v2_kel")
        _add(out, fid=nid("kel"), expr=f"rank(safe_div_null(subtract(close, KeltnerMid(close, {w})), close))", topic="V9/fe_v2_kel", desc=f"close/KMid-1", family="fe_v2_kel")
        _add(out, fid=nid("kel"), expr=f"rank(safe_div_null(subtract(KeltnerUpper(high, low, close, {w}, 10, 1.5), KeltnerLower(high, low, close, {w}, 10, 1.5)), close))", topic="V9/fe_v2_kel", desc=f"Keltner width", family="fe_v2_kel")
        _add(out, fid=nid("don"), expr=f"rank(donchian_position(close, high, low, {w}))", topic="V9/fe_v2_don", desc=f"donchian_pos({w})", family="fe_v2_don")
        _add(out, fid=nid("don"), expr=f"rank(safe_div_null(subtract(donchian_upper(high, {w}), donchian_lower(low, {w})), close))", topic="V9/fe_v2_don", desc=f"donchian width", family="fe_v2_don")
        _add(out, fid=nid("ch"), expr=f"rank(ts_channel_position(close, {w}))", topic="V9/fe_v2_ch", desc=f"channel_pos({w})", family="fe_v2_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_channel_width_pct(close, {w}))", topic="V9/fe_v2_ch", desc=f"channel_width_pct({w})", family="fe_v2_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_breakout_high(close, {w}))", topic="V9/fe_v2_ch", desc=f"breakout_high({w})", family="fe_v2_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_breakdown_low(close, {w}))", topic="V9/fe_v2_ch", desc=f"breakdown_low({w})", family="fe_v2_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_distance_to_high(close, {w}))", topic="V9/fe_v2_ch", desc=f"dist_high({w})", family="fe_v2_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_distance_to_low(close, {w}))", topic="V9/fe_v2_ch", desc=f"dist_low({w})", family="fe_v2_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_days_since_high(close, {w}))", topic="V9/fe_v2_ch", desc=f"days_since_high({w})", family="fe_v2_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_days_since_low(close, {w}))", topic="V9/fe_v2_ch", desc=f"days_since_low({w})", family="fe_v2_ch")
        _add(out, fid=nid("vol"), expr=f"rank(garman_klass_vol(open, high, low, close, {w}))", topic="V9/fe_v2_gk", desc=f"GK vol({w})", family="fe_v2_gk")
        _add(out, fid=nid("ewm"), expr=f"rank(ts_ewm_std(close, {w}, 0.1))", topic="V9/fe_v2_ewm", desc=f"ts_ewm_std({w})", family="fe_v2_ewm")

    # --- ichimoku ---
    _add(out, fid=nid("ichi"), expr=f"rank(safe_div_null(subtract(close, ichimoku_tenkan(high, low, 9)), close))", topic="V9/fe_v2_ichi", desc="close vs tenkan", family="fe_v2_ichi")
    _add(out, fid=nid("ichi"), expr=f"rank(safe_div_null(subtract(close, ichimoku_kijun(high, low, 26)), close))", topic="V9/fe_v2_ichi", desc="close vs kijun", family="fe_v2_ichi")
    _add(out, fid=nid("ichi"), expr=f"rank(ichimoku_cloud_position(high, low, close, 9, 26, 52))", topic="V9/fe_v2_ichi", desc="cloud position", family="fe_v2_ichi")
    _add(out, fid=nid("ichi"), expr=f"rank(safe_div_null(ichimoku_cloud_width(high, low, 9, 26, 52), close))", topic="V9/fe_v2_ichi", desc="cloud width", family="fe_v2_ichi")

    # --- candles / patterns (static + ranked) ---
    for name in (
        "candle_body_ratio",
        "candle_close_location",
        "candle_upper_shadow_ratio",
        "candle_lower_shadow_ratio",
        "candle_range",
        "candle_direction",
        "candle_gap_pct",
    ):
        if name == "candle_gap_pct":
            expr = f"rank({name}(open, pre_close))"
        elif name == "candle_range":
            expr = f"rank(safe_div_null({name}(high, low), pre_close))"
        elif name == "candle_direction":
            expr = f"rank({name}(open, close))"
        else:
            expr = f"rank({name}(open, high, low, close))"
        _add(out, fid=nid("cdl"), expr=expr, topic="V9/fe_v2_cdl", desc=name, family="fe_v2_cdl")

    for name in (
        "cdl_doji",
        "cdl_hammer",
        "cdl_inverted_hammer",
        "cdl_shooting_star",
        "cdl_engulfing",
        "cdl_harami",
        "cdl_morning_star",
        "cdl_evening_star",
        "cdl_three_white_soldiers",
        "cdl_three_black_crows",
        "cdl_marubozu",
        "cdl_spinning_top",
        "cdl_dark_cloud_cover",
        "cdl_piercing",
    ):
        _add(out, fid=nid("pat"), expr=f"rank({name}(open, high, low, close))", topic="V9/fe_v2_pat", desc=name, family="fe_v2_pat")
        _add(out, fid=nid("pat"), expr=f"rank(ts_sum({name}(open, high, low, close), 5))", topic="V9/fe_v2_pat", desc=f"{name} sum5", family="fe_v2_pat")

    # --- cs extras / multi-window ---
    for w in W2:
        _add(out, fid=nid("cs"), expr=f"rank(cs_resid(ts_pct(close, {w}), log(add(market_cap, 1.0))))", topic="V9/fe_v2_cs", desc=f"cs_resid size{w}", family="fe_v2_cs")
        _add(out, fid=nid("cs"), expr=f"cs_quantile(ts_pct(close, {w}), 0.9)", topic="V9/fe_v2_cs", desc=f"cs_q90 pct{w}", family="fe_v2_cs")
        _add(out, fid=nid("cs"), expr=f"cs_zscore(ts_pct(close, {w}))", topic="V9/fe_v2_cs", desc=f"cs_zscore pct{w}", family="fe_v2_cs")

    for w1, w2 in PAIRS:
        _add(out, fid=nid("mw"), expr=f"rank(subtract(SMA(close, {w1}), SMA(close, {w2})))", topic="V9/fe_v2_mw", desc=f"SMA {w1}-{w2}", family="fe_v2_mw")
        _add(out, fid=nid("mw"), expr=f"rank(subtract(TEMA(close, {w1}), TEMA(close, {w2})))", topic="V9/fe_v2_mw", desc=f"TEMA {w1}-{w2}", family="fe_v2_mw")
        _add(out, fid=nid("mw"), expr=f"rank(subtract(CMO(close, {w1}), CMO(close, {min(w2, 28)})))", topic="V9/fe_v2_mw", desc=f"CMO {w1}-{w2}", family="fe_v2_mw")
        _add(out, fid=nid("mw"), expr=f"if_else(gt(SupertrendDirection(high, low, close, 10, 3.0), 0.0), rank(ts_pct(close, {w1})), neg(rank(ts_pct(close, {w1}))))", topic="V9/fe_v2_mw", desc="ST gate momentum", family="fe_v2_mw")

    # dedupe
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for item in out:
        if item["expr"] in seen:
            continue
        seen.add(item["expr"])
        uniq.append(item)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-only", action="store_true", help="只合并已生成 json，不预计算")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    from api.dsl_parser import parse_expr
    from cold_start_library.runtime.dsl import extract_dsl_operator_names

    gen_path = ROOT / "data" / "ashare" / "expand_fe_update_v2.json"
    yaml_path = ROOT / "data" / "ashare" / "backend_v9_core.yaml"

    factors = generate()
    parse_ok: list[dict[str, Any]] = []
    parse_fail: list[dict[str, Any]] = []
    for item in factors:
        try:
            parse_expr(item["expr"], surface="compat")
            item = dict(item)
            item["operators"] = list(extract_dsl_operator_names(item["expr"]))
            parse_ok.append(item)
        except Exception as exc:
            parse_fail.append({"factor_id": item["factor_id"], "expr": item["expr"], "error": f"{type(exc).__name__}: {exc}"})

    print(f"[gen] {len(factors)} unique, parse_ok={len(parse_ok)} fail={len(parse_fail)}", flush=True)
    if parse_fail[:8]:
        print(json.dumps(parse_fail[:8], ensure_ascii=False, indent=2))
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
    payload["source"] = str(payload.get("source") or "") + "; +fe_update_v2"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False, width=120)
    print(f"[merge] added={added} total={len(entries)} -> {yaml_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
