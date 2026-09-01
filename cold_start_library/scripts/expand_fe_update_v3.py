#!/usr/bin/env python3
"""factor_engine 更新后第二批：成交量/换手/波动率/通道/K线等尚未入库算子。"""

from __future__ import annotations

import argparse
import json
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

W = (10, 20, 40, 60)
W2 = (10, 20, 60)


def _add(bag: list[dict[str, Any]], *, fid: str, expr: str, topic: str, desc: str, family: str) -> None:
    bag.append(
        {
            "factor_id": fid,
            "expr": " ".join(expr.split()),
            "topic": topic,
            "description": desc,
            "family": family,
            "source_library": "fe_update_v3",
        }
    )


def generate() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n = 0

    def nid(p: str) -> str:
        nonlocal n
        n += 1
        return f"fe_v3_{p}_{n:05d}"

    for w in W:
        _add(out, fid=nid("vol"), expr=f"rank(parkinson_vol(high, low, {w}))", topic="V9/fe_v3_vol", desc=f"parkinson({w})", family="fe_v3_vol")
        _add(out, fid=nid("vol"), expr=f"rank(yang_zhang_vol(open, high, low, close, {w}))", topic="V9/fe_v3_vol", desc=f"yang_zhang({w})", family="fe_v3_vol")
        _add(out, fid=nid("vol"), expr=f"rank(rogers_satchell_vol(open, high, low, close, {w}))", topic="V9/fe_v3_vol", desc=f"rogers_satchell({w})", family="fe_v3_vol")
        _add(out, fid=nid("vol"), expr=f"rank(ulcer_index(close, {w}))", topic="V9/fe_v3_vol", desc=f"ulcer({w})", family="fe_v3_vol")
        _add(out, fid=nid("vol"), expr=f"rank(range_volatility(high, low, close, {w}))", topic="V9/fe_v3_vol", desc=f"range_vol({w})", family="fe_v3_vol")
        _add(out, fid=nid("vol"), expr=f"rank(overnight_volatility(open, pre_close, {w}))", topic="V9/fe_v3_vol", desc=f"overnight_vol({w})", family="fe_v3_vol")
        _add(out, fid=nid("vol"), expr=f"rank(high_low_spread_proxy(high, low, {w}))", topic="V9/fe_v3_vol", desc=f"hl_spread({w})", family="fe_v3_vol")
        _add(out, fid=nid("vol"), expr=f"rank(corwin_schultz_spread(high, low, {w}))", topic="V9/fe_v3_vol", desc=f"corwin_schultz({w})", family="fe_v3_vol")
        _add(out, fid=nid("vol"), expr=f"rank(zero_return_ratio(close, {w}))", topic="V9/fe_v3_vol", desc=f"zero_ret({w})", family="fe_v3_vol")

        _add(out, fid=nid("liq"), expr=f"rank(relative_volume(volume, {w}))", topic="V9/fe_v3_liq", desc=f"rel_vol({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(volume_zscore(volume, {w}))", topic="V9/fe_v3_liq", desc=f"vol_z({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(volume_momentum(volume, {w}))", topic="V9/fe_v3_liq", desc=f"vol_mom({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(volume_shock(volume, {w}))", topic="V9/fe_v3_liq", desc=f"vol_shock({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(volume_volatility(volume, {w}))", topic="V9/fe_v3_liq", desc=f"vol_vol({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(volume_autocorr(volume, {w}))", topic="V9/fe_v3_liq", desc=f"vol_ac({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(adv(volume, {w}))", topic="V9/fe_v3_liq", desc=f"adv({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(up_volume_ratio(close, volume, {w}))", topic="V9/fe_v3_liq", desc=f"up_vol_ratio({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(up_down_volume_ratio(close, volume, {w}))", topic="V9/fe_v3_liq", desc=f"up_down_vol({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(return_volume_corr(close, volume, {w}))", topic="V9/fe_v3_liq", desc=f"ret_vol_corr({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(abs_return_volume_corr(close, volume, {w}))", topic="V9/fe_v3_liq", desc=f"abs_ret_vol_corr({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(price_volume_divergence(close, volume, {w}))", topic="V9/fe_v3_liq", desc=f"pv_div({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(price_impact(close, volume, {w}))", topic="V9/fe_v3_liq", desc=f"price_impact({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(volume_to_range(volume, high, low, {w}))", topic="V9/fe_v3_liq", desc=f"vol_to_range({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(volume_weighted_return(close, volume, {w}))", topic="V9/fe_v3_liq", desc=f"vw_ret({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(volume_weighted_momentum(close, volume, {w}))", topic="V9/fe_v3_liq", desc=f"vw_mom({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(rolling_obv(close, volume, {w}))", topic="V9/fe_v3_liq", desc=f"rolling_obv({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(rolling_pvt(close, volume, {w}))", topic="V9/fe_v3_liq", desc=f"rolling_pvt({w})", family="fe_v3_liq")
        _add(out, fid=nid("liq"), expr=f"rank(safe_div_null(subtract(close, rolling_vwap(close, volume, {w})), close))", topic="V9/fe_v3_liq", desc=f"close/rvwap-1", family="fe_v3_liq")

        _add(out, fid=nid("to"), expr=f"rank(turnover_zscore(turnover_ratio, {w}))", topic="V9/fe_v3_to", desc=f"to_z({w})", family="fe_v3_to")
        _add(out, fid=nid("to"), expr=f"rank(abnormal_turnover(turnover_ratio, {w}))", topic="V9/fe_v3_to", desc=f"abn_to({w})", family="fe_v3_to")
        _add(out, fid=nid("to"), expr=f"rank(average_turnover(turnover_ratio, {w}))", topic="V9/fe_v3_to", desc=f"avg_to({w})", family="fe_v3_to")
        _add(out, fid=nid("to"), expr=f"rank(turnover_momentum(turnover_ratio, {w}))", topic="V9/fe_v3_to", desc=f"to_mom({w})", family="fe_v3_to")
        _add(out, fid=nid("to"), expr=f"rank(turnover_shock(turnover_ratio, {w}))", topic="V9/fe_v3_to", desc=f"to_shock({w})", family="fe_v3_to")
        _add(out, fid=nid("to"), expr=f"rank(turnover_volatility(turnover_ratio, {w}))", topic="V9/fe_v3_to", desc=f"to_vol({w})", family="fe_v3_to")
        _add(out, fid=nid("to"), expr=f"rank(turnover_autocorr(turnover_ratio, {w}))", topic="V9/fe_v3_to", desc=f"to_ac({w})", family="fe_v3_to")
        _add(out, fid=nid("to"), expr=f"rank(turnover_acceleration(turnover_ratio, {w}))", topic="V9/fe_v3_to", desc=f"to_accel({w})", family="fe_v3_to")
        _add(out, fid=nid("to"), expr=f"rank(return_per_turnover(close, turnover_ratio, {w}))", topic="V9/fe_v3_to", desc=f"ret_per_to({w})", family="fe_v3_to")
        _add(out, fid=nid("to"), expr=f"rank(price_turnover_divergence(close, turnover_ratio, {w}))", topic="V9/fe_v3_to", desc=f"price_to_div({w})", family="fe_v3_to")
        _add(out, fid=nid("to"), expr=f"rank(turnover_adjusted_volatility(close, turnover_ratio, {w}))", topic="V9/fe_v3_to", desc=f"to_adj_vol({w})", family="fe_v3_to")

        _add(out, fid=nid("ch"), expr=f"rank(ts_new_high(close, {w}))", topic="V9/fe_v3_ch", desc=f"new_high({w})", family="fe_v3_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_new_low(close, {w}))", topic="V9/fe_v3_ch", desc=f"new_low({w})", family="fe_v3_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_channel_width(close, {w}))", topic="V9/fe_v3_ch", desc=f"ch_width({w})", family="fe_v3_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_channel_width_atr(close, high, low, {w}))", topic="V9/fe_v3_ch", desc=f"ch_width_atr({w})", family="fe_v3_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_channel_width_slope(close, {w}))", topic="V9/fe_v3_ch", desc=f"ch_width_slope({w})", family="fe_v3_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_distance_to_resistance(close, {w}))", topic="V9/fe_v3_ch", desc=f"dist_res({w})", family="fe_v3_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_distance_to_support(close, {w}))", topic="V9/fe_v3_ch", desc=f"dist_sup({w})", family="fe_v3_ch")
        _add(out, fid=nid("ch"), expr=f"rank(ts_range_expansion(close, {w}))", topic="V9/fe_v3_ch", desc=f"range_exp({w})", family="fe_v3_ch")
        _add(out, fid=nid("ts"), expr=f"rank(ts_sum_decay(close, {w}))", topic="V9/fe_v3_ts", desc=f"sum_decay({w})", family="fe_v3_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_decay_exp_window(close, {w}, 0.1))", topic="V9/fe_v3_ts", desc=f"decay_exp({w})", family="fe_v3_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ts_bottomk_sum(volume, {w}, 5))", topic="V9/fe_v3_ts", desc=f"bottomk_sum({w})", family="fe_v3_ts")
        _add(out, fid=nid("ts"), expr=f"rank(ewm_var((safe_div_null(close, pre_close) - 1.0), {w}))", topic="V9/fe_v3_ts", desc=f"ewm_var({w})", family="fe_v3_ts")
        _add(out, fid=nid("don"), expr=f"rank(safe_div_null(subtract(close, donchian_mid(high, low, {w})), close))", topic="V9/fe_v3_don", desc=f"close/don_mid-1", family="fe_v3_don")

    for w1, w2 in ((5, 20), (10, 60), (20, 60)):
        _add(out, fid=nid("ts"), expr=f"rank(ts_ratio(close, {w1}, {w2}))", topic="V9/fe_v3_ts", desc=f"ts_ratio {w1}/{w2}", family="fe_v3_ts")
        _add(out, fid=nid("liq"), expr=f"rank(volume_acceleration(volume, {w1}, {w2}))", topic="V9/fe_v3_liq", desc=f"vol_accel {w1}/{w2}", family="fe_v3_liq")

    _add(out, fid=nid("pvo"), expr=f"rank(PVO_signal(volume, 12, 26, 9))", topic="V9/fe_v3_pvo", desc="PVO_signal", family="fe_v3_pvo")
    _add(out, fid=nid("vwap"), expr=f"rank(vwap_deviation(close, vwap))", topic="V9/fe_v3_vwap", desc="vwap_dev", family="fe_v3_vwap")
    _add(out, fid=nid("iv"), expr=f"rank(intraday_volatility(high, low, close, open))", topic="V9/fe_v3_iv", desc="intraday_vol", family="fe_v3_iv")
    _add(out, fid=nid("nvi"), expr=f"rank(bounded_nvi(close, volume))", topic="V9/fe_v3_nvi", desc="bounded_nvi", family="fe_v3_nvi")
    _add(out, fid=nid("pvi"), expr=f"rank(bounded_pvi(close, volume))", topic="V9/fe_v3_pvi", desc="bounded_pvi", family="fe_v3_pvi")
    _add(out, fid=nid("pow"), expr=f"rank(signed_power(ts_pct(close, 1), 1.5))", topic="V9/fe_v3_pow", desc="signed_power", family="fe_v3_pow")
    _add(out, fid=nid("hump"), expr=f"rank(hump_decay(ts_pct(close, 1), 20, 0.5))", topic="V9/fe_v3_hump", desc="hump_decay", family="fe_v3_hump")

    _add(out, fid=nid("ichi"), expr=f"rank(safe_div_null(subtract(close, ichimoku_senkou_a(high, low, 9, 26)), close))", topic="V9/fe_v3_ichi", desc="close vs senkou_a", family="fe_v3_ichi")
    _add(out, fid=nid("ichi"), expr=f"rank(safe_div_null(subtract(close, ichimoku_senkou_b(high, low, 52)), close))", topic="V9/fe_v3_ichi", desc="close vs senkou_b", family="fe_v3_ichi")

    for name in (
        "cdl_hanging_man",
        "cdl_dragonfly_doji",
        "cdl_gravestone_doji",
        "cdl_harami_cross",
        "cdl_inside_bar",
        "cdl_outside_bar",
        "cdl_tweezer_top",
        "cdl_tweezer_bottom",
    ):
        _add(out, fid=nid("cdl"), expr=f"rank({name}(open, high, low, close))", topic="V9/fe_v3_cdl", desc=name, family="fe_v3_cdl")
        _add(out, fid=nid("cdl"), expr=f"rank(ts_sum({name}(open, high, low, close), 5))", topic="V9/fe_v3_cdl", desc=f"{name} sum5", family="fe_v3_cdl")

    for name in (
        "candle_body",
        "candle_abs_body",
        "candle_body_position",
        "candle_close_strength",
        "candle_upper_shadow",
        "candle_lower_shadow",
        "candle_rejection_upper",
        "candle_rejection_lower",
        "candle_overlap_ratio",
        "candle_inside_ratio",
    ):
        if name in {"candle_body", "candle_abs_body", "candle_upper_shadow", "candle_lower_shadow"}:
            expr = f"rank(safe_div_null({name}(open, high, low, close), pre_close))"
        else:
            expr = f"rank({name}(open, high, low, close))"
        _add(out, fid=nid("cdl2"), expr=expr, topic="V9/fe_v3_cdl2", desc=name, family="fe_v3_cdl2")

    for name, args in (
        ("candle_body_zscore", "(open, high, low, close, 20)"),
        ("candle_body_percentile", "(open, high, low, close, 60)"),
        ("candle_range_zscore", "(high, low, 20)"),
        ("candle_range_percentile", "(high, low, 60)"),
        ("candle_gap_atr", "(open, pre_close, high, low, close, 14)"),
        ("candle_range_atr", "(high, low, close, 14)"),
        ("candle_lower_shadow_zscore", "(open, high, low, close, 20)"),
        ("candle_upper_shadow_zscore", "(open, high, low, close, 20)"),
    ):
        _add(out, fid=nid("cdl3"), expr=f"rank({name}{args})", topic="V9/fe_v3_cdl3", desc=name, family="fe_v3_cdl3")

    for name in (
        "pattern_double_bottom",
        "pattern_double_top",
        "pattern_head_shoulders",
        "pattern_inverse_head_shoulders",
        "pattern_ascending_triangle",
        "pattern_descending_triangle",
        "pattern_sym_triangle",
        "pattern_bull_flag",
        "pattern_bear_flag",
        "pattern_cup_handle",
        "pattern_rounding_bottom",
        "pattern_rounding_top",
        "pattern_rising_wedge",
        "pattern_falling_wedge",
        "pattern_123_bull",
        "pattern_123_bear",
    ):
        for w in (40, 60):
            _add(out, fid=nid("pat"), expr=f"rank({name}(close, {w}))", topic="V9/fe_v3_pat", desc=f"{name}({w})", family="fe_v3_pat")

    for w in W2:
        _add(out, fid=nid("mw"), expr=f"rank(subtract(relative_volume(volume, {w}), relative_volume(volume, {w * 3})))", topic="V9/fe_v3_mw", desc=f"rel_vol spread {w}", family="fe_v3_mw")
        _add(out, fid=nid("mw"), expr=f"if_else(gt(ts_new_high(close, {w}), 0.0), rank(volume_zscore(volume, {w})), neg(rank(volume_zscore(volume, {w}))))", topic="V9/fe_v3_mw", desc="new_high gate volz", family="fe_v3_mw")
        _add(out, fid=nid("mw"), expr=f"rank(safe_div_null(yang_zhang_vol(open, high, low, close, {w}), parkinson_vol(high, low, {w})))", topic="V9/fe_v3_mw", desc=f"yz/park {w}", family="fe_v3_mw")

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
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    from api.dsl_parser import parse_expr
    from cold_start_library.runtime.dsl import extract_dsl_operator_names

    gen_path = ROOT / "data" / "ashare" / "expand_fe_update_v3.json"
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
            parse_fail.append(
                {"factor_id": item["factor_id"], "expr": item["expr"], "error": f"{type(exc).__name__}: {exc}"}
            )

    print(f"[gen] {len(factors)} unique, parse_ok={len(parse_ok)} fail={len(parse_fail)}", flush=True)
    if parse_fail[:15]:
        print(json.dumps(parse_fail[:15], ensure_ascii=False, indent=2))
    gen_path.write_text(
        json.dumps({"count": len(parse_ok), "items": parse_ok, "parse_fail": parse_fail}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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
    payload["source"] = str(payload.get("source") or "") + "; +fe_update_v3"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False, width=120)
    print(f"[merge] added={added} total={len(entries)} -> {yaml_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
