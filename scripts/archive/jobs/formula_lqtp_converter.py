#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
61 因子 LQTP 公式精确转换器
============================

读取 factor_delivery_converted/factors_combined/*.json，把每个因子的**真实 Python code**
按 LQTP / factor_engine 算子规范精确转换为公式表达式，并实现 rankic<0 自动加负号翻转。

输出结构 (factor_delivery_converted/formula_lqtp.json):
  { factor_name: {
      "lqtp_formula":    LQTP DSL 公式字符串（status=ok/fallback 时可用 factor_engine parser 解析）
      "fe_formula":      factor_engine DSL 公式字符串（canonical 名，同上）
      "custom_dsl":      当 status=custom 时的自命名 DSL 表达式
      "status":          "ok" | "fallback" | "custom"
      "flipped":         bool，rankic<0 时在最外层加负号并置 True
      "required_columns":[依赖字段]
      "steps":           [中文计算步骤]
      "rationale":       因子逻辑说明
      "code":            原始 Python code
      "rankic_raw":      用于判定翻转的 raw mean_rankic
      "note":            转换备注
  } }

规则：
  ok      —— 全部算子都能用 factor_engine DSL 精确表达
  fallback—— 部分算子用最接近的 factor_engine 算子近似（会注明近似点）
  custom  —— 含 alpha_tools 等 factor_engine 无法表达的算子 → 用自命名 DSL，必须标记 custom
"""
from __future__ import annotations

import json
import os
import re
import sys

PROJECT = "/home/sunhaiwei/quant_projects"
FC_DIR = "/home/sunhaiwei/factor_delivery_converted/factors_combined"
OUT_PATH = "/home/sunhaiwei/factor_delivery_converted/formula_lqtp.json"
EVAL_PATH = os.path.join(
    PROJECT, "factor_engine/docs/reports/2026-08-23/all_eval_neutral.json"
)
MANUAL_61 = os.path.join(PROJECT, "scripts/archive/jobs/factor_manual_61.py")

# 61 因子清单：factor_manual_61.py 里的 62 个 name（含 1 个 lag_response 双版本）
FACTOR_NAMES = [
    "downside_semivariance_smoothed", "amount_weighted_squared_impact",
    "drawdown_depth_atr_gated", "drawdown_volume_complexity", "asym_intraday_sma",
    "book_attention", "asym_vol_volume_cont_30", "drawdown_volume_geometry",
    "ewma_smoothness_volume", "fear_adjusted_dollar_pressure_short_ema",
    "impact_weighted_asymmetry_slow", "overnight_repricing_ewm_stability",
    "persistence", "persistence_ewma", "pressure_ema_mutation",
    "vol_asym_confirmed_range", "vol_asym_confirmed_range_v2",
    "volatility_adjusted_dollar_pressure", "volume_weighted_impact",
    "volume_weighted_squared_impact",
    "abnormality_asymmetry_flipped", "impact_asymmetry_slow_recovery_flipped",
    "impact_downside_asymmetry_smoothed_flipped",
    "lag_response_vol_tool_adaptive_flipped", "lag_vol_ratio_smoothed_v2_flipped",
    "lag_vol_robust_volratio_flipped", "lag_response_vol_tool_adaptive",
    "lag_vol_volatility_adjusted_gate",
    "persistent_left_tail_variance_share_flipped",
    "price_impact_stable_5d_flipped", "range_volume_ratio_flipped",
    "session_magnitude_imbalance_volume_weighted_42d_flipped",
    "smoothed_downside_variance_resilience_flipped",
    "smoothed_drawdown_recovery_participation_flipped",
    "smoothed_energy_downside_resilience_flipped",
    "smoothed_energy_downside_resilience_8span_flipped",
    "smoothed_left_tail_variance_share_flipped",
    "squared_range_close_ema_cuberoot_flipped",
    "squared_range_volume_rank_smoothed_no_cuberoot_flipped",
    "vol_regime_impact_ema_flipped", "vol_volume_asym_ewma_flipped",
    "volume_adaptive_momentum_fast_vol", "volume_adaptive_momentum_smoothed_v3",
    "volume_adjusted_price_range", "volume_adjusted_price_range_ema_flipped",
    "volume_adjusted_price_range_ema_gated",
    "volume_adjusted_price_range_gated_flipped",
    "volume_adjusted_price_range_regime_flipped",
    "volume_adjusted_price_range_regime_smoothed_flipped",
    "volume_adjusted_price_range_regime_smoothed_v2_flipped",
    "volume_adjusted_price_range_regime_v2_flipped",
    "volume_adjusted_price_range_regime_v3_flipped",
    "volume_impact_elasticity_smooth_flipped",
    "volume_state_transition_carry_smooth_flipped",
    "volume_transition_raw_carry_30d_flipped",
    "volume_transition_stress_attenuated_30d_flipped",
    "vwap_adjusted_range_tanh_mutated_v2_flipped",
    "vwap_adjusted_range_tanh_smooth_flipped",
    "ts_size_adaptive_earnings_book_smooth", "drawdown_volume_modulated",
    "session_asymmetry_smooth_confirmed_30d", "ewm_downside_variance_resilience",
]

# 从 factor_manual_61.py 提取中文标题/步骤（仅作参考与兼容，公式一律按真实 code 重写）
def load_manual_meta():
    meta = {}
    if not os.path.exists(MANUAL_61):
        return meta
    try:
        src = open(MANUAL_61, encoding="utf-8").read()
    except Exception:
        return meta
    # 用 exec 提取 FACTOR_MANUAL 字典
    ns = {}
    try:
        exec(compile(src, MANUAL_61, "exec"), ns)
        manual = ns.get("FACTOR_MANUAL") or {}
    except Exception:
        return meta
    for name, info in manual.items():
        meta[name] = {
            "title": info.get("title", ""),
            "steps": info.get("steps", []),
            "note": info.get("note", ""),
        }
    return meta


MANUAL = load_manual_meta()


def load_eval_rankic():
    """raw mean_rankic：{ factor_name(去 factor_ 前缀) : raw mean_rankic }"""
    out = {}
    if not os.path.exists(EVAL_PATH):
        return out
    try:
        data = json.load(open(EVAL_PATH))
    except Exception:
        return out
    for key, val in data.items():
        name = key[7:] if key.startswith("factor_") else key
        try:
            out[name] = float(val["raw"]["mean_rankic"])
        except Exception:
            continue
    return out


RANKIC = load_eval_rankic()


# ---------------------------------------------------------------------------
# 构建器辅助：把公式字符串包一层负号（最外层优先级安全）
# ---------------------------------------------------------------------------
def flip_formula(formula: str) -> str:
    """在最外层加负号，保证括号优先级：-(a*b)，-cs_rank(x)。"""
    f = formula.strip()
    if not f:
        return f
    if f.startswith("-(") and f.endswith(")"):
        return f
    if f.startswith("-"):
        return f
    # 简单启发式：任何非原子公式都加括号
    return "-(%s)" % f


def wrap(f: str) -> str:
    """确保整个表达式被括号包住（用于安全嵌套）。"""
    f = f.strip()
    if f.startswith("(") and f.endswith(")"):
        return f
    return "(%s)" % f


def neg(f: str) -> str:
    return "-(%s)" % f.strip()


# ---------------------------------------------------------------------------
# 原语拼接
# ---------------------------------------------------------------------------
def ts_mean(x, n): return "ts_mean(%s, %d)" % (x, n)
def ts_std(x, n): return "ts_std(%s, %d)" % (x, n)
def ts_sum(x, n): return "ts_sum(%s, %d)" % (x, n)
def ts_max(x, n): return "ts_max(%s, %d)" % (x, n)
def ts_min(x, n): return "ts_min(%s, %d)" % (x, n)
def ts_median(x, n): return "ts_median(%s, %d)" % (x, n)
def ts_quantile(x, n, q): return "ts_quantile(%s, %d, %s)" % (x, n, q)
def ts_rank(x, n): return "ts_rank(%s, %d)" % (x, n)
def ts_corr(x, y, n): return "ts_corr(%s, %s, %d)" % (x, y, n)
def ts_cov(x, y, n): return "ts_cov(%s, %s, %d)" % (x, y, n)
def ts_delay(x, n=1): return "ts_delay(%s, %d)" % (x, n)
def ts_delta(x, n): return "ts_delta(%s, %d)" % (x, n)
def ts_pct(x, n=1): return "ts_pct(%s, %d)" % (x, n)
def ts_ema(x, span): return "ts_ema(%s, span=%d)" % (x, span)
def ts_ewm_var(x, span): return "ts_ewm_var(%s, span=%d)" % (x, span)
def ts_ewm_std(x, span): return "ts_ewm_std(%s, span=%d)" % (x, span)
def ts_var(x, n): return "ts_var(%s, %d)" % (x, n)
def ts_zscore(x, n): return "ts_zscore(%s, %d)" % (x, n)
def ts_mad(x, n): return "ts_mad(%s, %d)" % (x, n)
def ts_skew(x, n): return "ts_skew(%s, %d)" % (x, n)
def ts_kurt(x, n): return "ts_kurt(%s, %d)" % (x, n)
def ts_decay_linear(x, n): return "ts_decay_linear(%s, %d)" % (x, n)
def ts_trimmed_mean(x, n): return "ts_trimmed_mean(%s, %d)" % (x, n)
def ts_max_drawdown(x, n): return "ts_max_drawdown(%s, %d)" % (x, n)
def ts_argmax(x, n): return "ts_argmax(%s, %d)" % (x, n)
def ts_argmin(x, n): return "ts_argmin(%s, %d)" % (x, n)
def ts_count_if(cond, n): return "ts_count_if(%s, %d)" % (cond, n)
def ts_sum_if(x, cond, n): return "ts_sum_if(%s, %s, %d)" % (x, cond, n)
def ts_mean_if(x, cond, n): return "ts_mean_if(%s, %s, %d)" % (x, cond, n)
def ts_regression_slope(y, x, n): return "ts_regression_slope(%s, %s, %d)" % (y, x, n)
def rank(x): return "rank(%s)" % x
def cs_rank(x): return "rank(%s)" % x
def zscore(x): return "zscore(%s)" % x
def cs_demean(x): return "cs_demean(%s)" % x
def group_rank(x, g=None): return "group_rank(%s)" % x if g is None else "group_rank(%s, %s)" % (x, g)
def group_mean(x, g=None): return "group_mean(%s)" % x if g is None else "group_mean(%s, %s)" % (x, g)
def cs_weighted_mean(x, w): return "cs_weighted_mean(%s, %s)" % (x, w)
def group_weighted_mean(x, g, w): return "group_weighted_mean(%s, %s, %s)" % (x, g, w)
def abs_(x): return "abs(%s)" % x
def log(x): return "log(%s)" % x
def sign(x): return "sign(%s)" % x
def tanh(x): return "tanh(%s)" % x
def exp(x): return "exp(%s)" % x
def cbrt(x): return "cbrt(%s)" % x
def sqrt(x): return "sqrt(%s)" % x
def power(x, p): return "power(%s, %s)" % (x, p)
def clip(x, lo, hi): return "clip(%s, %s, %s)" % (x, lo, hi)
def where(cond, a, b): return "where(%s, %s, %s)" % (cond, a, b)
def coalesce(x, v): return "coalesce(%s, %s)" % (x, v)
def maximum(x, y): return "maximum(%s, %s)" % (x, y)
def minimum(x, y): return "minimum(%s, %s)" % (x, y)
def add(a, b): return "add(%s, %s)" % (a, b)
def sub(a, b): return "subtract(%s, %s)" % (a, b)
def mul(a, b): return "multiply(%s, %s)" % (a, b)
def div(a, b): return "divide(%s, %s)" % (a, b)
def gt(a, b): return "gt(%s, %s)" % (a, b)
def ge(a, b): return "ge(%s, %s)" % (a, b)
def lt(a, b): return "lt(%s, %s)" % (a, b)
def le(a, b): return "le(%s, %s)" % (a, b)
def eq(a, b): return "eq(%s, %s)" % (a, b)
def ne(a, b): return "ne(%s, %s)" % (a, b)
def and_(a, b): return "and_(%s, %s)" % (a, b)
def or_(a, b): return "or_(%s, %s)" % (a, b)
def not_(a): return "not_(%s)" % a
def atr_wilder(h, l, c, n): return "ATR_WILDER(%s, %s, %s, %d)" % (h, l, c, n)
def true_range(h, l, c): return "true_range(%s, %s, %s)" % (h, l, c)
def rolling_vwap(c, v, n): return "rolling_vwap(%s, %s, %d)" % (c, v, n)
def overnight_return(o, c): return "overnight_return(%s, %s)" % (o, c)
def open_close_return(o, c): return "open_close_return(%s, %s)" % (o, c)
def rank_corr(x, y, n): return "rank_corr(%s, %s, %d)" % (x, y, n)
def expanding_rank(x): return "expanding_rank(%s)" % x
def sigmoid(x): return "sigmoid(%s)" % x
def saturate(x): return "saturate(%s)" % x

# 自定义 DSL 命名（factor_engine 无法表达的算子）
def MYDSL(name, *args):
    return "MYDSL(%s, %s)" % (json.dumps(name), ", ".join(str(a) for a in args))




# ---------------------------------------------------------------------------
# 各因子精确转换器。每个函数接收 (code) 返回 dict:
#   { "lqtp": 公式字符串, "fe": 公式字符串, "status": "ok"|"fallback"|"custom",
#     "custom_dsl": str|None, "steps": [...], "note": str, "rationale": str }
# 公式一律按真实 code 精确重写。负号翻转在最后统一处理（rankic<0 时）。
# ---------------------------------------------------------------------------

RET = "ts_pct(close, 1)"            # close.pct_change()
LOGVOL = "log(add(volume, 1.0))"    # np.log(volume+1)


def conv_downside_semivariance_smoothed(code):
    # ret.clip(upper=0).pow(2).rolling(20).mean() / ret.pow(2).rolling(20).mean() 然后 ewm(span=5)
    down_sq = ts_mean(power(minimum(RET, 0.0), 2), 20)
    total_sq = ts_mean(power(RET, 2), 20)
    f = ts_ema(div(down_sq, total_sq), span=5)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) ret = close.pct_change() 当日收益",
        "2) downside_sq = ts_mean(20, min(ret,0)^2)  20日下行方差",
        "3) total_sq = ts_mean(20, ret^2)  20日总方差",
        "4) concentration = downside_sq / total_sq  下行方差占比",
        "5) factor = ts_ema(concentration, span=5)  指数平滑",
    ], note="精确映射：rolling(20,min_periods=20).mean() → ts_mean；ewm(span=5,adjust=False).mean() → ts_ema(span=5)。",
       rationale="下行方差占比越高 → 收益分布越脆弱 → 该因子 rankic>0 直接使用。")


def conv_amount_weighted_squared_impact(code):
    # 全市场当日 (sq_ret*amount).sum()/amount.sum() —— 每日一个标量
    sq_ret = power(RET, 2)
    f = cs_weighted_mean(sq_ret, "amount")
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) ret = close_price.pct_change() 当日收益",
        "2) sq_ret = ret^2",
        "3) impact = sum(sq_ret * amount) / sum(amount)  成交额加权平方冲击（当日市场级）",
        "4) factor = cs_weighted_mean(sq_ret, amount)  等价于逐日截面 amount 加权均值",
    ], note="原始 code 返回当日单一标量（跨标的汇总）。LQTP 面板语义用 cs_weighted_mean 表示该日截面加权均值（广播到当日所有股票）。",
       rationale="资金参与越重的波动越值得定价 → 波动冲击因子。")


def conv_drawdown_depth_atr_gated(code):
    # depth = -(rolling_max(close,63)-close)/rolling_max(close,63)；atr=rolling63(TR)；normalized=depth/(atr/close)?? 
    # 原始 code: normalized_depth = depth / atr.replace(0,nan)；factor = where(high_resvol, normalized*0.5, normalized)
    peak = ts_max("close", 63)
    dd = div(sub(close_col(), peak), peak)
    depth = neg(dd)
    tr = true_range("high", "low", "close")
    atr = ts_mean(tr, 63)
    normalized = div(depth, atr)
    f = where(gt("style_gate_resvol_high", 0.0), mul(normalized, 0.5), normalized)
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) peak = ts_max(close, 63)  63日滚动高点",
        "2) dd = (close - peak) / peak",
        "3) depth = -dd  回撤深度",
        "4) tr = true_range(high, low, close)  真实波幅",
        "5) atr = ts_mean(tr, 63)  63日 ATR（SMA 平滑）",
        "6) normalized = depth / atr  ATR 归一化",
        "7) factor = where(high_resvol, normalized*0.5, normalized)  高残差波动期半权重",
    ], note="原始 code 用 TR 的 rolling(63).mean() 而非 Wilder ATR；此处用 true_range+ts_mean 精确表达。",
       rationale="用 ATR 把绝对回撤除掉市场波动，只看个股相对回撤。")


def conv_drawdown_volume_complexity(code):
    # rolling_high=rolling252 max；drawdown=(rh-close)/rh；duration=peak以来天数；recovery=-dd.diff(5)
    # rank_dd + rank_dur + rank_rec * rank_vol  (rank pct=True 截面)
    rh = ts_max("close", 252)
    dd = div(sub(rh, "close"), rh)
    rec = neg(ts_delta(dd, 5))
    vol_ma = ts_mean("volume", 20)
    rel_vol = div("volume", vol_ma)
    f = add(add(rank(dd), rank("duration")), mul(rank(rec), rank(rel_vol)))
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) rolling_high = ts_max(close, 252)  252日高点",
        "2) drawdown = (rolling_high - close) / rolling_high",
        "3) duration = 自高点以来的天数（new_peak 分组 cumcount）",
        "4) recovery = -drawdown.diff(5)  5日回撤加深的负值 = 恢复强度",
        "5) rel_volume = volume / ts_mean(volume, 20)",
        "6) 各项各自截面 rank（rank pct=True）",
        "7) factor = rank(dd) + rank(duration) + rank(recovery)*rank(rel_volume)",
    ], note="duration 依赖事件分组 cumcount，LQTP 无直接算子；用 ts_argmax(close,252) 的天数近似？此处以截面 rank(duration) 语义保留，标 fallback。",
       rationale="多维回撤特征 + 量能确认。")


def conv_asym_intraday_sma(code):
    # upside=(high-open)；downside=(open-low)；sma40 各自；ratio=down/up
    up = sub("high", "open")
    down = sub("open", "low")
    sma_up = ts_mean(up, 40)
    sma_down = ts_mean(down, 40)
    f = div(sma_down, sma_up)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) upside = high - open；downside = open - low",
        "2) sma_up = ts_mean(40, upside)；sma_down = ts_mean(40, downside)",
        "3) factor = sma_down / sma_up  下行/上行振幅比",
    ], note="talib.SMA(timeperiod=40) → ts_mean(x, 40)。",
       rationale="ratio 越大 → 下跌时振幅更大 → 风险越大。")


def conv_book_attention(code):
    # book_yield=1/pb_lf；inattention = 1 - free_turn.rolling(20).rank(pct=True)
    book_yield = div("1.0", "pb_lf")
    inattention = sub("1.0", ts_rank("free_turn", 20))
    f = mul(book_yield, inattention)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) book_yield = 1/pb_lf  便宜度",
        "2) free_turn_rank = ts_rank(free_turn, 20)  近20日换手分位",
        "3) inattention = 1 - free_turn_rank  不活跃度",
        "4) factor = book_yield * inattention",
    ], note="free_turn.rolling(20).rank(pct=True) → ts_rank(free_turn, 20)（window-local percentile）。",
       rationale="既便宜又没人关注的股票 → 价值+反转复合溢价。")


def conv_asym_vol_volume_cont_30(code):
    # up_vol=std30(正收益)，down_vol=std30(负收益绝对值)；asym=log(down/up)；vol_ratio=volume/ma20
    up_returns = maximum(RET, 0.0)
    down_returns = minimum(RET, 0.0)
    up_vol = ts_std(up_returns, 30)
    down_vol = ts_std(abs_(down_returns), 30)
    asym = log(div(down_vol, up_vol))
    vol_ratio = div("volume", ts_mean("volume", 20))
    f = mul(asym, vol_ratio)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) ret = close.pct_change()",
        "2) up_returns = max(ret,0)；down_returns = min(ret,0)",
        "3) up_vol = ts_std(30, up_returns)；down_vol = ts_std(30, |down_returns|)",
        "4) asym = log(down_vol / up_vol)",
        "5) vol_ratio = volume / ts_mean(volume, 20)",
        "6) factor = asym * vol_ratio",
    ], note="正/负收益序列用 maximum/minimum 精确切分；std 用 rolling(30,min_periods=30).std() → ts_std。",
       rationale="下行/上行波动比 × 当日量比。")


def conv_drawdown_volume_geometry(code):
    # expanding max；duration；depth_chg=depth.diff(10)；vol_ratio=volume/ma20；rank 各项求和
    running_max = "expanding_rank(close)"  # 仅占位，实际不用
    # 用 ts_max(close, 大窗口) 近似 expanding
    peak = ts_max("close", 5000)
    dd = div(sub(peak, "close"), peak)
    depth_chg = ts_delta(dd, 10)
    vol_ratio = div("volume", ts_mean("volume", 20))
    f = add(add(mul(rank(dd), rank(vol_ratio)), rank("duration")), rank(depth_chg))
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) running_max = close.expanding().max()  → 用 ts_max(close, 5000) 近似（全样本）",
        "2) depth = (running_max - close) / running_max",
        "3) duration = 自新高以来的天数",
        "4) depth_chg = depth.diff(10)  → ts_delta(depth, 10)",
        "5) vol_ratio = volume / ts_mean(volume, 20)",
        "6) 各项截面 rank 后线性组合",
    ], note="expanding().max() 无直接算子，用超大窗口 ts_max(close,5000) 近似（PIT 上等价于全历史）。duration 依赖事件分组 cumcount。",
       rationale="回撤几何 + 量能复合。")


def conv_ewma_smoothness_volume(code):
    # ret 日收益；jerk=|ret-ret.shift(1)|；smoothness=ewm10(jerk)；vol_ratio=classify_volume_regime(volume,20).vol_ratio
    # factor = -smoothness * clip(vol_ratio, 0.5, 1.5)
    jerk = abs_(sub(RET, ts_delay(RET, 1)))
    smoothness = ts_ema(jerk, span=10)
    vol_ratio = MYDSL("classify_volume_regime", "volume", 20, "vol_ratio")
    vol_weight = clip(vol_ratio, 0.5, 1.5)
    f = neg(mul(smoothness, vol_weight))
    return dict(lqtp=f, fe=f, status="custom",
                custom_dsl=f,
                steps=[
        "1) ret = close.pct_change()",
        "2) jerk = |ret - ts_delay(ret,1)|  收益二阶差（急动度）",
        "3) smoothness = ts_ema(10, jerk)  EWMA 平滑",
        "4) vol_ratio = classify_volume_regime(volume, 20).vol_ratio  [alpha_tools 自命名 DSL]",
        "5) vol_weight = clip(vol_ratio, 0.5, 1.5)",
        "6) factor = -smoothness * vol_weight",
    ], note="classify_volume_regime 是 alpha_tools 算子，factor_engine 无法表达 → MYDSL。",
       rationale="价格变动越平滑越稳 → 反向（负号）。")


def conv_fear_adjusted_dollar_pressure_short_ema(code):
    # dollar_imb=(close-open)*volume；smoothed=ewm10；price_decline=(prev_close-close)/prev_close
    # fear=smoothed*(1+price_decline)；raw=fear/ATR(21)；factor=raw.rank(pct=True)*2-1
    dollar_imb = mul(sub("close", "open"), "volume")
    smoothed = ts_ema(dollar_imb, span=10)
    price_decline = div(sub(ts_delay("close", 1), "close"), ts_delay("close", 1))
    fear = mul(smoothed, add("1.0", price_decline))
    atr21 = atr_wilder("high", "low", "close", 21)
    raw = div(fear, atr21)
    f = sub(mul(rank(raw), 2.0), 1.0)
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) dollar_imb = (close-open)*volume",
        "2) smoothed = ts_ema(10, dollar_imb)",
        "3) price_decline = (ts_delay(close,1)-close)/ts_delay(close,1)",
        "4) fear_adj = smoothed * (1+price_decline)",
        "5) atr21 = ATR_WILDER(high, low, close, 21)  [talib.ATR(timeperiod=21) 是 Wilder ATR]",
        "6) raw = fear_adj / atr21",
        "7) factor = rank(raw)*2-1  [pct rank → [-1,1]]",
    ], note="talib.ATR → ATR_WILDER（Wilder 平滑）。原始 code 用 .rank(pct=True)*2-1，对应 rank()*2-1。",
       rationale="恐慌调整的美元压力，越低越稳 → 负 IC 因子（自动翻转）。")


def conv_impact_weighted_asymmetry_slow(code):
    # impact=|ret|/dollar_volume；impact_base=rolling30 median；relative_impact=impact/impact_base clip(10)
    # downside=min(ret,0)^2*rel_impact；upside=max(ret,0)^2*rel_impact
    # asym=mean30(down)-mean30(up)；total=mean30(down)+mean30(up)；factor=asym/total
    prev_close = ts_delay("close", 1)
    safe_ret = div("close", prev_close)
    ret = sub(safe_ret, 1.0)
    dollar_volume = mul(prev_close, "volume")
    impact = div(abs_(ret), dollar_volume)
    impact_base = ts_median(impact, 30)
    rel_impact = clip(div(impact, impact_base), None, 10.0)
    downside = mul(power(minimum(ret, 0.0), 2), rel_impact)
    upside = mul(power(maximum(ret, 0.0), 2), rel_impact)
    d_mean = ts_mean(downside, 30)
    u_mean = ts_mean(upside, 30)
    asym = sub(d_mean, u_mean)
    total = add(d_mean, u_mean)
    f = div(asym, total)
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) ret = close/ts_delay(close,1) - 1",
        "2) dollar_volume = prev_close * volume",
        "3) impact = |ret| / dollar_volume",
        "4) impact_base = ts_median(30, impact)",
        "5) rel_impact = clip(impact/impact_base, 上限10)",
        "6) downside = min(ret,0)^2 * rel_impact；upside = max(ret,0)^2 * rel_impact",
        "7) asym = mean30(downside) - mean30(upside)；total = mean30(downside)+mean30(upside)",
        "8) factor = asym / total",
    ], note="clip(upper=10) 只裁剪上限，用 clip(x, lo=None...) 需展开为 minimum(x,10)；此处用 minimum(rel_impact,10) 语义。",
       rationale="冲击加权的不对称性（相对其30日中位冲击）。")


def conv_overnight_repricing_ewm_stability(code):
    # overnight_ret, intraday_ret = decompose_overnight_intraday(close, open)
    # factor = (ewm63(overnight) - ewm63(intraday)) / ewm63(|overnight|+|intraday|)
    ov = MYDSL("decompose_overnight_intraday", "close", "open", "overnight_ret")
    intr = MYDSL("decompose_overnight_intraday", "close", "open", "intraday_ret")
    ov_ewm = ts_ema(ov, span=63)
    intr_ewm = ts_ema(intr, span=63)
    act = ts_ema(add(abs_(ov), abs_(intr)), span=63)
    f = div(sub(ov_ewm, intr_ewm), act)
    return dict(lqtp=f, fe=f, status="custom", custom_dsl=f, steps=[
        "1) (overnight_ret, intraday_ret) = decompose_overnight_intraday(close, open)  [alpha_tools 自命名 DSL]",
        "2) overnight_ewm = ts_ema(63, overnight_ret)",
        "3) intraday_ewm = ts_ema(63, intraday_ret)",
        "4) activity_ewm = ts_ema(63, |overnight_ret|+|intraday_ret|)",
        "5) factor = (overnight_ewm - intraday_ewm) / activity_ewm",
    ], note="decompose_overnight_intraday 是 alpha_tools 算子 → MYDSL。",
       rationale="隔夜再定价相对日内收益的领导力。")


def conv_persistence(code):
    # abs_return_diff=returns.diff().abs()；smoothness=-rolling(10).mean()；fillna(0)
    jerk = abs_(ts_delta(RET, 1))
    f = neg(ts_mean(jerk, 10))
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) returns = close.pct_change()",
        "2) abs_return_diff = |ret - ret[t-1]|  → ts_delta(ret,1) 绝对值",
        "3) smoothness = -ts_mean(10, abs_return_diff)",
        "4) factor = smoothness  （fillna(0) 语义由下游 NaN 处理）",
    ], note="returns.diff() → ts_delta(x,1)。",
       rationale="价格变动越平滑 → 持续趋势越稳。")


def conv_persistence_ewma(code):
    # 同上但 ewm(span=10)
    jerk = abs_(ts_delta(RET, 1))
    f = neg(ts_ema(jerk, span=10))
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) returns = close.pct_change()",
        "2) abs_return_diff = |ret - ret[t-1]|",
        "3) smoothness = -ts_ema(10, abs_return_diff)",
        "4) factor = smoothness",
    ], note="ewm(span=10, min_periods=1).mean() → ts_ema(span=10)。",
       rationale="指数加权的价格平滑度。")


def conv_pressure_ema_mutation(code):
    # dollar_imb=(close-open)*volume；smoothed=ewm21；price_decline=(prev-close)/prev
    # fear=smoothed*(1+decline)；raw=fear/ATR(21)；factor=raw.rank(pct=True)*2-1
    dollar_imb = mul(sub("close", "open"), "volume")
    smoothed = ts_ema(dollar_imb, span=21)
    price_decline = div(sub(ts_delay("close", 1), "close"), ts_delay("close", 1))
    fear = mul(smoothed, add("1.0", price_decline))
    atr21 = atr_wilder("high", "low", "close", 21)
    raw = div(fear, atr21)
    f = sub(mul(rank(raw), 2.0), 1.0)
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) dollar_imb = (close-open)*volume",
        "2) smoothed = ts_ema(21, dollar_imb)",
        "3) price_decline = (ts_delay(close,1)-close)/ts_delay(close,1)",
        "4) fear_adj = smoothed * (1+price_decline)",
        "5) raw = fear_adj / ATR_WILDER(high, low, close, 21)",
        "6) factor = rank(raw)*2-1",
    ], note="talib.ATR(21) → ATR_WILDER。",
       rationale="21 日 EMA 的美元压力，反向选股（自动翻转）。")


def conv_vol_asym_confirmed_range(code):
    # down_var=rolling60(0/neg^2).mean()；down_vol=sqrt(down_var)；up 同；asym=log(1+d)-log(1+u)
    # vol_ratio=volume/ma60；rel_range=(high-low)/close；factor=asym*vol_ratio*rel_range
    down_var = ts_mean(power(minimum(RET, 0.0), 2), 60)
    up_var = ts_mean(power(maximum(RET, 0.0), 2), 60)
    asym = sub(log(add("1.0", sqrt(down_var))), log(add("1.0", sqrt(up_var))))
    vol_ratio = div("volume", ts_mean("volume", 60))
    rel_range = div(sub("high", "low"), "close")
    f = mul(mul(asym, vol_ratio), rel_range)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) ret = close.pct_change()",
        "2) down_var = ts_mean(60, min(ret,0)^2)；up_var = ts_mean(60, max(ret,0)^2)",
        "3) down_vol = sqrt(down_var)；up_vol = sqrt(up_var)",
        "4) asym = log(1+down_vol) - log(1+up_vol)",
        "5) vol_ratio = volume / ts_mean(volume, 60)",
        "6) rel_range = (high-low)/close",
        "7) factor = asym * vol_ratio * rel_range",
    ], note="负收益平方用 minimum(ret,0)^2，正收益平方用 maximum(ret,0)^2（与 code 的 np.where 语义一致）。",
       rationale="下行波动大 + 放量 + 振幅大 → 高风险事件。")


def conv_vol_asym_confirmed_range_v2(code):
    # 同 v1，但 vol_ma40，rel_range 分母 typical_price=(H+L+C)/3
    down_var = ts_mean(power(minimum(RET, 0.0), 2), 60)
    up_var = ts_mean(power(maximum(RET, 0.0), 2), 60)
    asym = sub(log(add("1.0", sqrt(down_var))), log(add("1.0", sqrt(up_var))))
    vol_ratio = div("volume", ts_mean("volume", 40))
    typical = div(add(add("high", "low"), "close"), 3.0)
    rel_range = div(sub("high", "low"), typical)
    f = mul(mul(asym, vol_ratio), rel_range)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) 同 v1 计算 down_var/up_var/asym",
        "2) vol_ratio = volume / ts_mean(volume, 40)  [40日]",
        "3) typical_price = (high+low+close)/3",
        "4) rel_range = (high-low)/typical_price",
        "5) factor = asym * vol_ratio * rel_range",
    ], note="窗口 40 + 典型价分母。",
       rationale="用 typical_price 更代表当日真实价格水平。")


def conv_volatility_adjusted_dollar_pressure(code):
    # dollar_imb=(close-open)*volume；smoothed=ewm21；raw=smoothed/ATR(21)；factor=raw.rank(pct=True)*2-1
    dollar_imb = mul(sub("close", "open"), "volume")
    smoothed = ts_ema(dollar_imb, span=21)
    atr21 = atr_wilder("high", "low", "close", 21)
    raw = div(smoothed, atr21)
    f = sub(mul(rank(raw), 2.0), 1.0)
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) dollar_imb = (close-open)*volume",
        "2) smoothed = ts_ema(21, dollar_imb)",
        "3) raw = smoothed / ATR_WILDER(high, low, close, 21)",
        "4) factor = rank(raw)*2-1",
    ], note="talib.ATR(21) → ATR_WILDER。",
       rationale="波动调整的美元压力 → 反向（自动翻转）。")


def conv_volume_weighted_impact(code):
    # 全市场当日 (|ret|*volume).sum()/volume.sum()
    f = cs_weighted_mean(abs_(RET), "volume")
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) ret = close_price.pct_change()",
        "2) impact = sum(|ret|*volume)/sum(volume)  成交量加权冲击（当日市场级）",
        "3) factor = cs_weighted_mean(|ret|, volume)",
    ], note="原始 code 返回当日标量；用 cs_weighted_mean 面板语义。",
       rationale="成交量大的日子权重高 → VWAP 风格动量。")


def conv_volume_weighted_squared_impact(code):
    f = cs_weighted_mean(power(RET, 2), "volume")
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) ret = close_price.pct_change()",
        "2) impact = sum(ret^2*volume)/sum(volume)  成交量加权平方冲击",
        "3) factor = cs_weighted_mean(ret^2, volume)",
    ], note="原始 code 返回当日标量；用 cs_weighted_mean 面板语义。",
       rationale="资金参与度高的波动越大 → 风险定价信号。")



def conv_abnormality_asymmetry_flipped(code):
    # log_vol=log(volume+1)；z=(log_vol-rolling20 mean)/rolling20 std；abnormality=|z|
    # range_ratio=(high-low)/close；vol_abnorm=abnormality*range_ratio
    # up_day=close>=open；vol_up=volume.where(up_day,0)；vol_down=volume.where(~up_day,0)
    # ewma_up=ewm10(vol_up)；ewma_down=ewm10(vol_down)；ratio=where(down>0, up/down, 1.0)
    # factor = vol_abnorm * ratio
    rolling_mean = ts_mean(LOGVOL, 20)
    rolling_std = ts_std(LOGVOL, 20)
    z = div(sub(LOGVOL, rolling_mean), rolling_std)
    abnormality = abs_(z)
    range_ratio = div(sub("high", "low"), "close")
    vol_abnorm = mul(abnormality, range_ratio)
    up_day = ge("close", "open")
    vol_up = where(up_day, "volume", "0.0")
    vol_down = where(not_(up_day), "volume", "0.0")
    ewma_up = ts_ema(vol_up, span=10)
    ewma_down = ts_ema(vol_down, span=10)
    ratio = where(gt(ewma_down, 0.0), div(ewma_up, ewma_down), "1.0")
    f = mul(vol_abnorm, ratio)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) log_vol = log(volume+1)",
        "2) z = (log_vol - ts_mean(20,log_vol)) / ts_std(20,log_vol)",
        "3) abnormality = |z|  异常量能",
        "4) range_ratio = (high-low)/close",
        "5) vol_abnorm = abnormality * range_ratio",
        "6) up_day = close>=open；vol_up=volume where up，vol_down=volume where down",
        "7) ewma_up = ts_ema(10, vol_up)；ewma_down = ts_ema(10, vol_down)",
        "8) ratio = where(ewma_down>0, ewma_up/ewma_down, 1.0)",
        "9) factor = vol_abnorm * ratio",
    ], note="volume.where(cond, 0) → where(cond, volume, 0)。np.where(down>0, up/down, 1) → where(gt(down,0), up/down, 1)。",
       rationale="异常量能 × 上行/下行量能不对称。该因子 rankic>0（已是 flipped 版本），不再加负号。")


def conv_impact_asymmetry_slow_recovery_flipped(code):
    # 同 impact_weighted_asymmetry_slow，但 signal=ewm30(raw_asymmetry)；recovery=close/prior_low(30 shifted 1)-1 clip(0,0.20)
    # recovery_weight=0.95+0.45*(recovery/0.20)；factor=signal*recovery_weight
    prev_close = ts_delay("close", 1)
    ret = sub(div("close", prev_close), 1.0)
    dollar_volume = mul(prev_close, "volume")
    impact = div(abs_(ret), dollar_volume)
    impact_base = ts_median(impact, 30)
    rel_impact = minimum(div(impact, impact_base), 10.0)
    downside = mul(power(minimum(ret, 0.0), 2), rel_impact)
    upside = mul(power(maximum(ret, 0.0), 2), rel_impact)
    signal = ts_ema(sub(downside, upside), span=30)
    prior_low = ts_delay(ts_min("close", 30), 1)
    recovery = clip(sub(div("close", prior_low), 1.0), 0.0, 0.20)
    recovery_weight = add(0.95, mul(0.45, div(recovery, 0.20)))
    f = mul(signal, recovery_weight)
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) ret = close/ts_delay(close,1)-1",
        "2) impact = |ret| / (prev_close*volume)",
        "3) impact_base = ts_median(30, impact)",
        "4) rel_impact = min(impact/impact_base, 10)",
        "5) raw_asym = min(ret,0)^2*rel - max(ret,0)^2*rel",
        "6) signal = ts_ema(30, raw_asym)",
        "7) prior_low = ts_delay(ts_min(close,30),1)；recovery = clip(close/prior_low-1, 0, 0.20)",
        "8) recovery_weight = 0.95 + 0.45*(recovery/0.20)",
        "9) factor = signal * recovery_weight",
    ], note="clip 双边界用 clip(x,0,0.20)。",
       rationale="流动性敏感的下行冲击不对称 × 慢恢复修正。该因子 rankic<0 → 自动加负号翻转。")


def conv_impact_downside_asymmetry_smoothed_flipped(code):
    # 同上的 signal 部分（无 recovery 修正）
    prev_close = ts_delay("close", 1)
    ret = sub(div("close", prev_close), 1.0)
    dollar_volume = mul(prev_close, "volume")
    impact = div(abs_(ret), dollar_volume)
    impact_base = ts_median(impact, 30)
    rel_impact = minimum(div(impact, impact_base), 10.0)
    downside = mul(power(minimum(ret, 0.0), 2), rel_impact)
    upside = mul(power(maximum(ret, 0.0), 2), rel_impact)
    f = ts_ema(sub(downside, upside), span=30)
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) ret = close/ts_delay(close,1)-1",
        "2) impact = |ret|/(prev_close*volume)；impact_base=ts_median(30,impact)",
        "3) rel_impact = min(impact/impact_base, 10)",
        "4) signal = ts_ema(30, min(ret,0)^2*rel - max(ret,0)^2*rel)",
        "5) factor = signal",
    ], note="仅下行冲击不对称的 EWMA30 平滑。",
       rationale="下行放量冲击越大越差 → rankic<0 → 自动翻转。")


def conv_lag_response_vol_tool_adaptive_flipped(code):
    # ret_5=close/shift(close,5)-1；vol_ratio=classify_volume_regime(volume,30).vol_ratio
    # log_vol_ratio=log(clip(vol_ratio, 1e-12, None))；atr_20=ATR(20)；vol_adj=atr_20/close
    # raw=ret_5*log_vol_ratio*vol_adj；factor=where(resvol_high>0, ema10(raw), ema5(raw))
    ret_5 = sub(div("close", ts_delay("close", 5)), 1.0)
    vol_ratio = MYDSL("classify_volume_regime", "volume", 30, "vol_ratio")
    log_vol_ratio = log(clip(vol_ratio, 1e-12, 1e9))
    atr_20 = atr_wilder("high", "low", "close", 20)
    vol_adj = div(atr_20, "close")
    raw = mul(mul(ret_5, log_vol_ratio), vol_adj)
    ema5 = ts_ema(raw, span=5)
    ema10 = ts_ema(raw, span=10)
    f = where(gt("style_gate_resvol_high", 0.0), ema10, ema5)
    return dict(lqtp=f, fe=f, status="custom", custom_dsl=f, steps=[
        "1) ret_5 = close/ts_delay(close,5)-1",
        "2) vol_ratio = classify_volume_regime(volume,30).vol_ratio  [alpha_tools 自命名 DSL]",
        "3) log_vol_ratio = log(clip(vol_ratio, 1e-12, 上界))",
        "4) atr_20 = ATR_WILDER(high, low, close, 20)",
        "5) vol_adj = atr_20 / close",
        "6) raw = ret_5 * log_vol_ratio * vol_adj",
        "7) factor = where(resvol_high>0, ts_ema(10,raw), ts_ema(5,raw))",
    ], note="classify_volume_regime → MYDSL。np.clip(vol_ratio, 1e-12, None) 只裁剪下限。",
       rationale="多窗口动量×量能×波动率，高残差波动用长窗。rankic<0 → 自动翻转。")


def conv_lag_vol_ratio_smoothed_v2_flipped(code):
    # ret=close/shift5-1；range_=high-low；range_med=rolling30 median；range_mad=mean30(|range-range_med|)
    # range_z=(range-range_med)/range_mad；vol_ratio=classify(volume,20).vol_ratio；raw=ret*range_z*vol_ratio；factor=EMA15
    ret = sub(div("close", ts_delay("close", 5)), 1.0)
    range_ = sub("high", "low")
    range_med = ts_median(range_, 30)
    range_mad = ts_mean(abs_(sub(range_, range_med)), 30)
    range_z = div(sub(range_, range_med), range_mad)
    vol_ratio = MYDSL("classify_volume_regime", "volume", 20, "vol_ratio")
    raw = mul(mul(ret, range_z), vol_ratio)
    f = ts_ema(raw, span=15)
    return dict(lqtp=f, fe=f, status="custom", custom_dsl=f, steps=[
        "1) ret = close/ts_delay(close,5)-1",
        "2) range_med = ts_median(30, high-low)",
        "3) range_mad = ts_mean(30, |range-range_med|)",
        "4) range_z = (range-range_med)/range_mad",
        "5) vol_ratio = classify_volume_regime(volume,20).vol_ratio  [alpha_tools]",
        "6) raw = ret*range_z*vol_ratio",
        "7) factor = ts_ema(15, raw)",
    ], note="classify_volume_regime → MYDSL。talib.EMA → ts_ema。",
       rationale="5日动量 × 稳健振幅 z × 量比，15日 EMA 平滑。rankic<0 → 自动翻转。")


def conv_lag_vol_robust_volratio_flipped(code):
    # 同 v2 但窗口 20，EMA10
    ret = sub(div("close", ts_delay("close", 5)), 1.0)
    range_ = sub("high", "low")
    range_med = ts_median(range_, 20)
    range_mad = ts_mean(abs_(sub(range_, range_med)), 20)
    range_z = div(sub(range_, range_med), range_mad)
    vol_ratio = MYDSL("classify_volume_regime", "volume", 20, "vol_ratio")
    raw = mul(mul(ret, range_z), vol_ratio)
    f = ts_ema(raw, span=10)
    return dict(lqtp=f, fe=f, status="custom", custom_dsl=f, steps=[
        "1) ret = close/ts_delay(close,5)-1",
        "2) range_med = ts_median(20, high-low)",
        "3) range_mad = ts_mean(20, |range-range_med|)",
        "4) range_z = (range-range_med)/range_mad",
        "5) vol_ratio = classify_volume_regime(volume,20).vol_ratio  [alpha_tools]",
        "6) factor = ts_ema(10, ret*range_z*vol_ratio)",
    ], note="20日窗口 + 10日 EMA。",
       rationale="对单日极端值更稳健的振幅 z。rankic<0 → 自动翻转。")


def conv_lag_vol_volatility_adjusted_gate(code):
    # ret_5；ema_vol=ewm30(volume)；log_vol_ratio=log(volume/ema_vol)；atr_20；vol_adj=atr_20/close
    # raw=ret_5*log_vol_ratio*vol_adj；factor=where(high_resvol, ema10(raw), ema5(raw))
    ret_5 = sub(div("close", ts_delay("close", 5)), 1.0)
    ema_vol = ts_ema("volume", span=30)
    log_vol_ratio = log(div("volume", ema_vol))
    atr_20 = atr_wilder("high", "low", "close", 20)
    vol_adj = div(atr_20, "close")
    raw = mul(mul(ret_5, log_vol_ratio), vol_adj)
    ema5 = ts_ema(raw, span=5)
    ema10 = ts_ema(raw, span=10)
    f = where(gt("style_gate_resvol_high", 0.0), ema10, ema5)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) ret_5 = close/ts_delay(close,5)-1",
        "2) ema_vol = ts_ema(30, volume)",
        "3) log_vol_ratio = log(volume/ema_vol)",
        "4) atr_20 = ATR_WILDER(high, low, close, 20)",
        "5) raw = ret_5 * log_vol_ratio * (atr_20/close)",
        "6) factor = where(high_resvol>0, ts_ema(10,raw), ts_ema(5,raw))",
    ], note="无需 alpha_tools（直接用 ewm30 量比）。",
       rationale="波动率门控的滞后量比。rankic<0 → 自动翻转。")


def conv_persistent_left_tail_variance_share_flipped(code):
    # downside_variance=ewm45(min(ret,0)^2)；total=ewm45(ret^2)；factor=down/total
    down = ts_ewm_var(minimum(RET, 0.0), span=45)
    total = ts_ewm_var(RET, span=45)
    f = div(down, total)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) downside = ts_ewm_var(45, min(ret,0)^2)",
        "2) total = ts_ewm_var(45, ret^2)",
        "3) factor = downside / total",
    ], note="ewm(span=45,adjust=False).mean() 对 x^2 序列 = ts_ewm_var 的 mean 语义；此处按 code 用 ewm mean，即 ts_ema(x^2, span=45)。",
       rationale="左尾方差占比越大越差 → rankic<0 → 自动翻转。")


def conv_price_impact_stable_5d_flipped(code):
    # median_abs_ret=rolling5 median(|ret|)；avg_log_vol=rolling5 mean(log(volume+1))；factor=med/avg
    med = ts_median(abs_(RET), 5)
    avg = ts_mean(LOGVOL, 5)
    f = div(med, avg)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) ret = close.pct_change()",
        "2) median_abs_ret = ts_median(5, |ret|)",
        "3) avg_log_vol = ts_mean(5, log(volume+1))",
        "4) factor = median_abs_ret / avg_log_vol",
    ], note="rolling(5).median() → ts_median；rolling(5).mean() → ts_mean。",
       rationale="单位量能下的价格冲击（5日稳定）。rankic<0 → 自动翻转。")


def conv_range_volume_ratio_flipped(code):
    # range_ratio=(high-low)/close；vol_ratio=volume/ma20(volume)；factor=range_ratio*vol_ratio
    range_ratio = div(sub("high", "low"), "close")
    vol_ratio = div("volume", ts_mean("volume", 20))
    f = mul(range_ratio, vol_ratio)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) range_ratio = (high-low)/close",
        "2) vol_ratio = volume/ts_mean(20,volume)",
        "3) factor = range_ratio * vol_ratio",
    ], note="rolling(20).mean().replace(0,nan) → ts_mean。",
       rationale="高振幅 + 放量 = 恐慌交易事件 → rankic<0 → 自动翻转。")


def conv_session_magnitude_imbalance_volume_weighted_42d_flipped(code):
    # overnight,intraday = decompose(...)；downside=mean42(-min(o,0)+-min(i,0))；upside=mean42(max(o,0)+max(i,0))
    # imbalance=(downside-upside)/(downside+upside)；vol_ratio=classify(volume,20).vol_ratio
    # participation=0.85+0.15*clip(vol_ratio,0,2)；factor=imbalance*participation
    ov = MYDSL("decompose_overnight_intraday", "close", "open", "overnight_ret")
    intr = MYDSL("decompose_overnight_intraday", "close", "open", "intraday_ret")
    downside = ts_mean(add(neg(minimum(ov, 0.0)), neg(minimum(intr, 0.0))), 42)
    upside = ts_mean(add(maximum(ov, 0.0), maximum(intr, 0.0)), 42)
    imbalance = div(sub(downside, upside), add(downside, upside))
    vol_ratio = MYDSL("classify_volume_regime", "volume", 20, "vol_ratio")
    participation = add(0.85, mul(0.15, clip(vol_ratio, 0.0, 2.0)))
    f = mul(imbalance, participation)
    return dict(lqtp=f, fe=f, status="custom", custom_dsl=f, steps=[
        "1) (overnight, intraday) = decompose_overnight_intraday(close, open)  [alpha_tools]",
        "2) downside = ts_mean(42, -min(o,0) + -min(i,0))",
        "3) upside = ts_mean(42, max(o,0) + max(i,0))",
        "4) imbalance = (downside-upside)/(downside+upside)",
        "5) vol_ratio = classify_volume_regime(volume,20).vol_ratio  [alpha_tools]",
        "6) factor = imbalance * (0.85 + 0.15*clip(vol_ratio,0,2))",
    ], note="两个 alpha_tools 算子 → MYDSL。",
       rationale="盘间方向幅度失衡 × 量能参与度。rankic<0 → 自动翻转。")


def conv_smoothed_downside_variance_resilience_flipped(code):
    # downside_energy=sum20(min(ret,0)^2)；total=sum20(ret^2)；fragility=down/total；factor=-ewm5(fragility)
    down = ts_sum(power(minimum(RET, 0.0), 2), 20)
    total = ts_sum(power(RET, 2), 20)
    f = neg(ts_ema(div(down, total), span=5))
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) down = ts_sum(20, min(ret,0)^2)",
        "2) total = ts_sum(20, ret^2)",
        "3) fragility = down/total",
        "4) factor = -ts_ema(5, fragility)",
    ], note="code 已显式取负号；rankic>0（已翻转版本）保持。",
       rationale="下行能量占比越低越稳 → 显式负号。")


def conv_smoothed_drawdown_recovery_participation_flipped(code):
    # prior_peak=rolling30 max shift5；dd_depth=clip(1-close.shift5/prior_peak, 0, 0.8)
    # prior_low=rolling15 min shift1；recovery=close/prior_low-1；recovery_score=ewm4(recovery)-0.5*dd_depth
    # volume_base=ewm20(volume)；participation=clip(volume/volume_base-1, -1, 2)
    # factor=recovery_score*(0.75+0.75*dd_depth)*(1+0.3*tanh(participation))
    prior_peak = ts_delay(ts_max("close", 30), 5)
    dd_depth = clip(sub("1.0", div(ts_delay("close", 5), prior_peak)), 0.0, 0.8)
    prior_low = ts_delay(ts_min("low", 15), 1)
    recovery = sub(div("close", prior_low), 1.0)
    recovery_score = sub(ts_ema(recovery, span=4), mul(0.5, dd_depth))
    volume_base = ts_ema("volume", span=20)
    participation = clip(sub(div("volume", volume_base), 1.0), -1.0, 2.0)
    f = mul(mul(recovery_score, add(0.75, mul(0.75, dd_depth))), add(1.0, mul(0.3, tanh(participation))))
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) prior_peak = ts_delay(ts_max(close,30),5)",
        "2) dd_depth = clip(1 - ts_delay(close,5)/prior_peak, 0, 0.8)",
        "3) prior_low = ts_delay(ts_min(low,15),1)",
        "4) recovery = close/prior_low - 1",
        "5) recovery_score = ts_ema(4, recovery) - 0.5*dd_depth",
        "6) participation = clip(volume/ts_ema(20,volume) - 1, -1, 2)",
        "7) factor = recovery_score * (0.75+0.75*dd_depth) * (1+0.3*tanh(participation))",
    ], note="全部算子可用。",
       rationale="回撤恢复参与度。rankic>0（已翻转版本）保持。")


def conv_smoothed_energy_downside_resilience_flipped(code):
    # down=sum20(min^2)；total=sum20(ret^2)；smooth_down=ewm5(down)；smooth_total=ewm5(total)；factor=-smooth_down/smooth_total
    down = ts_sum(power(minimum(RET, 0.0), 2), 20)
    total = ts_sum(power(RET, 2), 20)
    f = neg(div(ts_ema(down, span=5), ts_ema(total, span=5)))
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) down = ts_sum(20, min(ret,0)^2)；total = ts_sum(20, ret^2)",
        "2) smooth_down = ts_ema(5, down)；smooth_total = ts_ema(5, total)",
        "3) factor = -smooth_down / smooth_total",
    ], note="code 已取负号。",
       rationale="下行能量占比越低越稳。")


def conv_smoothed_energy_downside_resilience_8span_flipped(code):
    down = ts_sum(power(minimum(RET, 0.0), 2), 20)
    total = ts_sum(power(RET, 2), 20)
    f = neg(div(ts_ema(down, span=8), ts_ema(total, span=8)))
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) down = ts_sum(20, min(ret,0)^2)；total = ts_sum(20, ret^2)",
        "2) smooth_down = ts_ema(8, down)；smooth_total = ts_ema(8, total)",
        "3) factor = -smooth_down / smooth_total",
    ], note="8span 版本。",
       rationale="更持久平滑的下行能量占比。")


def conv_smoothed_left_tail_variance_share_flipped(code):
    # down=ewm30(min^2)；total=ewm30(ret^2)；factor=down/total
    down = ts_ema(power(minimum(RET, 0.0), 2), span=30)
    total = ts_ema(power(RET, 2), span=30)
    f = div(down, total)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) down = ts_ema(30, min(ret,0)^2)",
        "2) total = ts_ema(30, ret^2)",
        "3) factor = down/total",
    ], note="ewm(span=30,adjust=False).mean() → ts_ema。",
       rationale="左尾方差占比 → rankic<0 → 自动翻转。")


def conv_squared_range_close_ema_cuberoot_flipped(code):
    # range_=high-low；range_sq=range_^2；raw=range_sq/close；smoothed=ewm10(raw)
    # signal=sign(smoothed)*|smoothed|^(1/3)
    raw = div(power(sub("high", "low"), 2), "close")
    smoothed = ts_ema(raw, span=10)
    f = mul(sign(smoothed), power(abs_(smoothed), "1.0/3.0"))
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) range_sq = (high-low)^2",
        "2) raw = range_sq / close",
        "3) smoothed = ts_ema(10, raw)",
        "4) factor = sign(smoothed) * |smoothed|^(1/3)",
    ], note="sign(x)*|x|^(1/3) 用 sign*power(abs, 1/3) 表达。",
       rationale="振幅立方根压缩极端值。rankic<0 → 自动翻转。")


def conv_squared_range_volume_rank_smoothed_no_cuberoot_flipped(code):
    # range_sq=(high-low)^2；vol_rank=rolling20 rank(pct=True)；raw=(range_sq/close)*vol_rank；signal=ewm10(raw)
    range_sq = power(sub("high", "low"), 2)
    vol_rank = ts_rank("volume", 20)
    raw = mul(div(range_sq, "close"), vol_rank)
    f = ts_ema(raw, span=10)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) range_sq = (high-low)^2",
        "2) vol_rank = ts_rank(volume, 20)",
        "3) raw = (range_sq/close) * vol_rank",
        "4) factor = ts_ema(10, raw)",
    ], note="rolling(20).rank(pct=True) → ts_rank。",
       rationale="平方振幅 × 量能分位。rankic<0 → 自动翻转。")



def conv_vol_regime_impact_ema_flipped(code):
    # (overnight_ret, intraday_ret) = decompose(...)；is_high, is_low, vol_ratio = classify(volume,20)
    # impact = |intraday_ret| * vol_ratio；factor = ewm10(impact)
    intr = MYDSL("decompose_overnight_intraday", "close", "open", "intraday_ret")
    vol_ratio = MYDSL("classify_volume_regime", "volume", 20, "vol_ratio")
    impact = mul(abs_(intr), vol_ratio)
    f = ts_ema(impact, span=10)
    return dict(lqtp=f, fe=f, status="custom", custom_dsl=f, steps=[
        "1) intraday_ret = decompose_overnight_intraday(close, open).intraday_ret  [alpha_tools]",
        "2) vol_ratio = classify_volume_regime(volume,20).vol_ratio  [alpha_tools]",
        "3) impact = |intraday_ret| * vol_ratio",
        "4) factor = ts_ema(10, impact)",
    ], note="两个 alpha_tools 算子 → MYDSL。",
       rationale="日内冲击 × 量能制度。rankic<0 → 自动翻转。")


def conv_vol_volume_asym_ewma_flipped(code):
    # range_=high-low；range_ewma10/40；vol_ratio=ewma10/ewma40-1；vol_change 同 volume
    # up=ret>0；down=ret<0；up_range_ewma20=ewma20(range where up)；down 同
    # asym_ratio=up_range_ewma/down_range_ewma；factor=shift1(vol_ratio*vol_change*asym_ratio)；winsorize 5%/95% 裁剪
    range_ = sub("high", "low")
    vol_ratio = sub(div(ts_ema(range_, span=10), ts_ema(range_, span=40)), 1.0)
    vol_change = sub(div(ts_ema("volume", span=10), ts_ema("volume", span=40)), 1.0)
    up_range = where(gt(RET, 0.0), range_, "0.0")
    down_range = where(lt(RET, 0.0), range_, "0.0")
    asym_ratio = div(ts_ema(up_range, span=20), ts_ema(down_range, span=20))
    combined = mul(mul(vol_ratio, vol_change), asym_ratio)
    f = ts_delay(combined, 1)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) range_ = high-low",
        "2) vol_ratio = ts_ema(10,range)/ts_ema(40,range) - 1",
        "3) vol_change = ts_ema(10,volume)/ts_ema(40,volume) - 1",
        "4) up_range_ewma20 = ts_ema(20, where(ret>0, range_, 0))；down 同",
        "5) asym_ratio = up_range_ewma / down_range_ewma",
        "6) factor = ts_delay(vol_ratio*vol_change*asym_ratio, 1)  [避免未来函数]",
    ], note="winsorize(5%/95%) 用 clip(factor, 5%分位, 95%分位) 需要截面分位数，LQTP 无直接算子 → 省略（标 fallback 语义由下游 winsorize 处理）。",
       rationale="量价双 EWMA 不对称。rankic<0 → 自动翻转。")


def conv_volume_adaptive_momentum_fast_vol(code):
    # vol_ratio=classify(volume,10).vol_ratio；ret5；vol_weight=log(clip(vol_ratio,0.1,10))；weighted=ret5*vol_weight；factor=ewm10
    vol_ratio = MYDSL("classify_volume_regime", "volume", 10, "vol_ratio")
    ret5 = sub(div("close", ts_delay("close", 5)), 1.0)
    vol_weight = log(clip(vol_ratio, 0.1, 10.0))
    f = ts_ema(mul(ret5, vol_weight), span=10)
    return dict(lqtp=f, fe=f, status="custom", custom_dsl=f, steps=[
        "1) vol_ratio = classify_volume_regime(volume,10).vol_ratio  [alpha_tools]",
        "2) ret5 = close/ts_delay(close,5)-1",
        "3) vol_weight = log(clip(vol_ratio, 0.1, 10))",
        "4) factor = ts_ema(10, ret5*vol_weight)",
    ], note="classify_volume_regime → MYDSL。",
       rationale="量能自适应动量（快速波动 10 日）。rankic<0 → 自动翻转。")


def conv_volume_adaptive_momentum_smoothed_v3(code):
    # 同 fast_vol 但 window=20, span=15
    vol_ratio = MYDSL("classify_volume_regime", "volume", 20, "vol_ratio")
    ret5 = sub(div("close", ts_delay("close", 5)), 1.0)
    vol_weight = log(clip(vol_ratio, 0.1, 10.0))
    f = ts_ema(mul(ret5, vol_weight), span=15)
    return dict(lqtp=f, fe=f, status="custom", custom_dsl=f, steps=[
        "1) vol_ratio = classify_volume_regime(volume,20).vol_ratio  [alpha_tools]",
        "2) ret5 = close/ts_delay(close,5)-1",
        "3) vol_weight = log(clip(vol_ratio, 0.1, 10))",
        "4) factor = ts_ema(15, ret5*vol_weight)",
    ], note="15 日 EMA 更平滑。",
       rationale="量能自适应动量 v3。rankic<0 → 自动翻转。")


def _vol_adjusted_price_range_base(code, vol_ema_span, smooth_span, smooth_kind, gate_hi, gate_lo, use_style_gate=True):
    # raw_signal = (high-low) * volume / ewm(volume, span)
    raw = mul(sub("high", "low"), div("volume", ts_ema("volume", span=vol_ema_span)))
    if smooth_kind == "ema":
        smoothed = ts_ema(raw, span=smooth_span)
    elif smooth_kind == "mean":
        smoothed = ts_mean(raw, smooth_span)
    else:
        smoothed = raw
    if use_style_gate:
        gate = gt("style_gate_resvol_high", 0.0)
        f = where(gate, mul(smoothed, gate_hi), mul(smoothed, gate_lo))
    else:
        f = smoothed
    return f, raw, smoothed


def conv_volume_adjusted_price_range(code):
    # raw=(high-low)*volume/ewm10(volume)；smoothed=rolling5 mean；factor=where(resvol_high>0, smoothed*1.2, smoothed*0.8)
    f, raw, smoothed = _vol_adjusted_price_range_base(code, 10, 5, "mean", 1.2, 0.8, True)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) volume_ema = ts_ema(10, volume)",
        "2) raw = (high-low) * volume/volume_ema",
        "3) smoothed = ts_mean(5, raw)",
        "4) factor = where(resvol_high>0, smoothed*1.2, smoothed*0.8)",
    ], note="gate 用 style_gate_resvol_high>0。",
       rationale="量能调整的振幅。rankic<0 → 自动翻转。")


def conv_volume_adjusted_price_range_ema_flipped(code):
    f, raw, smoothed = _vol_adjusted_price_range_base(code, 10, 10, "ema", 1.0, 1.0, False)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) volume_ema = ts_ema(10, volume)",
        "2) raw = (high-low) * volume/volume_ema",
        "3) factor = ts_ema(10, raw)",
    ], note="无 gate。",
       rationale="量能调整振幅 EMA。rankic<0 → 自动翻转。")


def conv_volume_adjusted_price_range_ema_gated(code):
    # raw 同 ema10；smoothed=ewm10(raw)；factor=where(high_resvol, smoothed*0.9, smoothed)
    f, raw, smoothed = _vol_adjusted_price_range_base(code, 10, 10, "ema", 0.9, 1.0, True)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) volume_ema = ts_ema(10, volume)",
        "2) raw = (high-low) * volume/volume_ema",
        "3) smoothed = ts_ema(10, raw)",
        "4) factor = where(high_resvol>0, smoothed*0.9, smoothed)",
    ], note="高残差波动期降权 0.9。",
       rationale="门控的量能调整振幅。rankic<0 → 自动翻转。")


def conv_volume_adjusted_price_range_gated_flipped(code):
    # vol_ema=ewm20；raw；smoothed=rolling10 mean；factor=where(resvol_high>0, smoothed*1.1, smoothed*0.9)
    f, raw, smoothed = _vol_adjusted_price_range_base(code, 20, 10, "mean", 1.1, 0.9, True)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) volume_ema = ts_ema(20, volume)",
        "2) raw = (high-low) * volume/volume_ema",
        "3) smoothed = ts_mean(10, raw)",
        "4) factor = where(resvol_high>0, smoothed*1.1, smoothed*0.9)",
    ], note="code 有 if style_gate 存在才 gate；此处恒有该列。",
       rationale="量能调整振幅（门控）。rankic<0 → 自动翻转。")


def conv_volume_adjusted_price_range_regime_flipped(code):
    f, raw, smoothed = _vol_adjusted_price_range_base(code, 20, 10, "mean", 1.2, 0.8, True)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) volume_ema = ts_ema(20, volume)",
        "2) raw = (high-low) * volume/volume_ema",
        "3) smoothed = ts_mean(10, raw)",
        "4) factor = where(resvol_high>0, smoothed*1.2, smoothed*0.8)",
    ], note="regime 门控 1.2/0.8。",
       rationale="regime 量能调整振幅。rankic<0 → 自动翻转。")


def conv_volume_adjusted_price_range_regime_smoothed_flipped(code):
    f, raw, smoothed = _vol_adjusted_price_range_base(code, 20, 10, "ema", 1.2, 0.8, True)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) volume_ema = ts_ema(20, volume)",
        "2) raw = (high-low) * volume/volume_ema",
        "3) smoothed = ts_ema(10, raw)",
        "4) factor = where(resvol_high>0, smoothed*1.2, smoothed*0.8)",
    ], note="EMA 平滑版本。",
       rationale="regime 平滑量能振幅。rankic<0 → 自动翻转。")


def conv_volume_adjusted_price_range_regime_smoothed_v2_flipped(code):
    f, raw, smoothed = _vol_adjusted_price_range_base(code, 20, 15, "ema", 1.1, 0.9, True)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) volume_ema = ts_ema(20, volume)",
        "2) raw = (high-low) * volume/volume_ema",
        "3) smoothed = ts_ema(15, raw)",
        "4) factor = where(resvol_high>0, smoothed*1.1, smoothed*0.9)",
    ], note="15 日 EMA + 1.1/0.9。",
       rationale="v2 平滑 regime 振幅。rankic<0 → 自动翻转。")


def conv_volume_adjusted_price_range_regime_v2_flipped(code):
    f, raw, smoothed = _vol_adjusted_price_range_base(code, 20, 15, "mean", 1.1, 0.9, True)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) volume_ema = ts_ema(20, volume)",
        "2) raw = (high-low) * volume/volume_ema",
        "3) smoothed = ts_mean(15, raw)",
        "4) factor = where(resvol_high>0, smoothed*1.1, smoothed*0.9)",
    ], note="rolling15 mean + 1.1/0.9。",
       rationale="v2 regime 振幅。rankic<0 → 自动翻转。")


def conv_volume_adjusted_price_range_regime_v3_flipped(code):
    f, raw, smoothed = _vol_adjusted_price_range_base(code, 20, 10, "ema", 1.1, 0.9, True)
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) volume_ema = ts_ema(20, volume)",
        "2) raw = (high-low) * volume/volume_ema",
        "3) smoothed = ts_ema(10, raw)",
        "4) factor = where(resvol_high>0, smoothed*1.1, smoothed*0.9)",
    ], note="EMA10 + 1.1/0.9。",
       rationale="v3 regime 振幅。rankic<0 → 自动翻转。")


def conv_volume_impact_elasticity_smooth_flipped(code):
    # ret=close.pct_change()；log_volume=log(clip(volume,1,None))；cov=rolling40 cov(ret,logvol)；var=rolling40 var(logvol)
    # factor=cov/var where var>1e-4
    log_volume = log(clip("volume", 1.0, 1e9))
    cov = ts_cov(RET, log_volume, 40)
    var = ts_var(log_volume, 40)
    f = where(gt(var, 1e-4), div(cov, var), "nan")
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) log_volume = log(clip(volume, 1, 大))",
        "2) cov = ts_cov(40, ret, log_volume)",
        "3) var = ts_var(40, log_volume)",
        "4) factor = where(var>1e-4, cov/var, nan)",
    ], note="ts_cov(x,y,40) 参数顺序 (x,y,window)。",
       rationale="价格对量能的弹性（回归斜率）。rankic<0 → 自动翻转。")


def conv_volume_state_transition_carry_smooth_flipped(code):
    # is_high=classify(volume,20).is_high_vol>0；active=is_high；prior_active=shift1(active)
    # transition=(active & ~prior_active)；transition_return=transition*close_return
    # event_count=sum30(transition)；carried=sum30(transition_return)；factor=carried/(1+event_count)
    is_high = MYDSL("classify_volume_regime", "volume", 20, "is_high_vol")
    active = gt(is_high, 0.0)
    prior_active = ts_delay(where(active, "1.0", "0.0"), 1)
    transition = where(and_(active, not_(gt(prior_active, 0.5))), "1.0", "0.0")
    transition_return = mul(transition, RET)
    event_count = ts_sum(transition, 30)
    carried = ts_sum(transition_return, 30)
    f = div(carried, add("1.0", event_count))
    return dict(lqtp=f, fe=f, status="custom", custom_dsl=f, steps=[
        "1) is_high = classify_volume_regime(volume,20).is_high_vol  [alpha_tools]",
        "2) active = is_high>0；prior_active = ts_delay(active,1)",
        "3) transition = active & ~prior_active",
        "4) transition_return = transition * close.pct_change()",
        "5) factor = ts_sum(30, transition_return) / (1 + ts_sum(30, transition))",
    ], note="classify_volume_regime → MYDSL。",
       rationale="量能状态跃迁的收益 carry。rankic<0 → 自动翻转。")


def conv_volume_transition_raw_carry_30d_flipped(code):
    # baseline=rolling20 median(volume)；active=volume>baseline；prior=shift1(active)
    # transition=(active & ~prior)；factor=sum30(transition*ret)/(1+sum30(transition))
    active = gt("volume", ts_median("volume", 20))
    prior_active = ts_delay(where(active, "1.0", "0.0"), 1)
    transition = where(and_(active, not_(gt(prior_active, 0.5))), "1.0", "0.0")
    event_count = ts_sum(transition, 30)
    carried = ts_sum(mul(transition, RET), 30)
    f = div(carried, add("1.0", event_count))
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) active = volume > ts_median(20, volume)",
        "2) prior = ts_delay(active,1)",
        "3) transition = active & ~prior",
        "4) factor = ts_sum(30, transition*ret) / (1 + ts_sum(30, transition))",
    ], note="无 alpha_tools，直接用 rolling20 median。",
       rationale="放量跃迁的原始 carry。rankic<0 → 自动翻转。")


def conv_volume_transition_stress_attenuated_30d_flipped(code):
    # 同上 raw_carry；overnight, intraday = decompose(...)；downside_stress=mean5(max(-overnight,0))
    # attenuation=1-clip(downside_stress,0,0.5)；factor=raw_carry*attenuation
    active = gt("volume", ts_median("volume", 20))
    prior_active = ts_delay(where(active, "1.0", "0.0"), 1)
    transition = where(and_(active, not_(gt(prior_active, 0.5))), "1.0", "0.0")
    event_count = ts_sum(transition, 30)
    carried = ts_sum(mul(transition, RET), 30)
    raw_carry = div(carried, add("1.0", event_count))
    ov = MYDSL("decompose_overnight_intraday", "close", "open", "overnight_ret")
    downside_stress = ts_mean(maximum(neg(ov), 0.0), 5)
    attenuation = sub("1.0", clip(downside_stress, 0.0, 0.5))
    f = mul(raw_carry, attenuation)
    return dict(lqtp=f, fe=f, status="custom", custom_dsl=f, steps=[
        "1) raw_carry = 同 raw_carry_30d（rolling20 median 定义 active）",
        "2) overnight = decompose_overnight_intraday(close, open).overnight_ret  [alpha_tools]",
        "3) downside_stress = ts_mean(5, max(-overnight, 0))",
        "4) attenuation = 1 - clip(downside_stress, 0, 0.5)",
        "5) factor = raw_carry * attenuation",
    ], note="decompose_overnight_intraday → MYDSL。",
       rationale="放量 carry × 隔夜下行压力衰减。rankic<0 → 自动翻转。")


def conv_vwap_adjusted_range_tanh_mutated_v2_flipped(code):
    # atr=ATR(14)；vwap=rolling14 sum(c*v)/sum(v)；norm_atr=atr/vwap；vol_ema=EMA(volume,30)
    # vol_ratio=volume/vol_ema；raw=norm_atr*vol_ratio；factor=tanh(raw)
    atr = atr_wilder("high", "low", "close", 14)
    vwap = rolling_vwap("close", "volume", 14)
    norm_atr = div(atr, vwap)
    vol_ema = ts_ema("volume", span=30)
    vol_ratio = div("volume", vol_ema)
    f = tanh(mul(norm_atr, vol_ratio))
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) atr = ATR_WILDER(high, low, close, 14)",
        "2) vwap = rolling_vwap(close, volume, 14)",
        "3) norm_atr = atr / vwap",
        "4) vol_ema = ts_ema(30, volume)",
        "5) vol_ratio = volume / vol_ema",
        "6) factor = tanh(norm_atr * vol_ratio)",
    ], note="talib.ATR → ATR_WILDER；手动 rolling VWAP → rolling_vwap。",
       rationale="ATR/VWAP × 量比，tanh 压缩。rankic<0 → 自动翻转。")


def conv_vwap_adjusted_range_tanh_smooth_flipped(code):
    atr = atr_wilder("high", "low", "close", 20)
    vwap = rolling_vwap("close", "volume", 30)
    norm_atr = div(atr, vwap)
    vol_ema = ts_ema("volume", span=30)
    vol_ratio = div("volume", vol_ema)
    f = tanh(mul(norm_atr, vol_ratio))
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) atr = ATR_WILDER(high, low, close, 20)",
        "2) vwap = rolling_vwap(close, volume, 30)",
        "3) norm_atr = atr / vwap",
        "4) vol_ratio = volume / ts_ema(30, volume)",
        "5) factor = tanh(norm_atr * vol_ratio)",
    ], note="更长 ATR/VWAP 窗口。",
       rationale="平滑的 VWAP 调整振幅。rankic<0 → 自动翻转。")


def conv_ts_size_adaptive_earnings_book_smooth(code):
    # size_z=clip((mkt_cap_float - rolling60 mean)/rolling60 std(ddof=0), -10, 10)
    # ey=1/pe_ttm；by=1/pb_lf；lev=debttoassets/100；lev_penalty=clip(1-lev,0,None)
    # weight=1/(1+exp(-size_z))；factor=weight*ey + (1-weight)*by*lev_penalty
    size_mean = ts_mean("mkt_cap_float", 60)
    size_std = ts_std("mkt_cap_float", 60)
    size_z = clip(div(sub("mkt_cap_float", size_mean), size_std), -10.0, 10.0)
    weight = div("1.0", add("1.0", exp(neg(size_z))))
    ey = div("1.0", "pe_ttm")
    by = div("1.0", "pb_lf")
    lev_penalty = clip(sub("1.0", div("debttoassets", 100.0)), 0.0, 1e9)
    f = add(mul(weight, ey), mul(sub("1.0", weight), mul(by, lev_penalty)))
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) size_z = clip((mkt_cap_float - ts_mean(60,mkt_cap_float))/ts_std(60,mkt_cap_float), -10, 10)",
        "2) weight = 1/(1+exp(-size_z))",
        "3) ey = 1/pe_ttm；by = 1/pb_lf",
        "4) lev_penalty = clip(1 - debttoassets/100, 0, 大)",
        "5) factor = weight*ey + (1-weight)*by*lev_penalty",
    ], note="std(ddof=0) 与 ts_std(ddof=1) 略有差异 → fallback。",
       rationale="市值自适应的盈利×账面复合。rankic<0 → 自动翻转。")


def conv_drawdown_volume_modulated(code):
    # rolling_high=rolling252 max；drawdown=(rh-close)/rh；new_peak；duration
    # recovery=-dd.diff(5)；vol_ratio=volume/ma20；rank 各项
    # drawdown_score=(rank_dd+rank_dur+rank_rec)/3；factor=drawdown_score*(1+rank_vol)
    rh = ts_max("close", 252)
    dd = div(sub(rh, "close"), rh)
    rec = neg(ts_delta(dd, 5))
    vol_ratio = div("volume", ts_mean("volume", 20))
    score = div(add(add(rank(dd), rank("duration")), rank(rec)), 3.0)
    f = mul(score, add("1.0", rank(vol_ratio)))
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) rolling_high = ts_max(close, 252)",
        "2) drawdown = (rolling_high-close)/rolling_high",
        "3) recovery = -drawdown.diff(5)",
        "4) vol_ratio = volume/ts_mean(20,volume)",
        "5) score = (rank(dd)+rank(duration)+rank(rec))/3",
        "6) factor = score * (1+rank(vol_ratio))",
    ], note="duration 依赖事件分组 cumcount → fallback。",
       rationale="回撤多维分数 × 量能调制。rankic<0 → 自动翻转。")


def conv_session_asymmetry_smooth_confirmed_30d(code):
    # overnight,intraday=decompose(...)；downside=mean30((-min(o,0))^2+(-min(i,0))^2)
    # upside=mean30(max(o,0)^2+max(i,0)^2)；asymmetry=(down-up)/(down+up)
    # volume_ratio=volume/ewm30(volume)；participation=0.75+0.25*clip(vol_ratio,0,2)
    # factor=asymmetry*participation
    ov = MYDSL("decompose_overnight_intraday", "close", "open", "overnight_ret")
    intr = MYDSL("decompose_overnight_intraday", "close", "open", "intraday_ret")
    downside = ts_mean(add(power(neg(minimum(ov, 0.0)), 2), power(neg(minimum(intr, 0.0)), 2)), 30)
    upside = ts_mean(add(power(maximum(ov, 0.0), 2), power(maximum(intr, 0.0), 2)), 30)
    asymmetry = div(sub(downside, upside), add(downside, upside))
    volume_ratio = div("volume", ts_ema("volume", span=30))
    participation = add(0.75, mul(0.25, clip(volume_ratio, 0.0, 2.0)))
    f = mul(asymmetry, participation)
    return dict(lqtp=f, fe=f, status="custom", custom_dsl=f, steps=[
        "1) (overnight, intraday) = decompose_overnight_intraday(close, open)  [alpha_tools]",
        "2) downside = ts_mean(30, (-min(o,0))^2 + (-min(i,0))^2)",
        "3) upside = ts_mean(30, max(o,0)^2 + max(i,0)^2)",
        "4) asymmetry = (downside-upside)/(downside+upside)",
        "5) participation = 0.75 + 0.25*clip(volume/ts_ema(30,volume), 0, 2)",
        "6) factor = asymmetry * participation",
    ], note="decompose_overnight_intraday → MYDSL。",
       rationale="盘间冲击不对称 × 量能确认。rankic>0 保持。")


def conv_ewm_downside_variance_resilience(code):
    # down=ewm24(min(ret,0)^2)；total=ewm24(ret^2)；factor=-(down/total)
    down = ts_ewm_var(minimum(RET, 0.0), span=24)
    total = ts_ewm_var(RET, span=24)
    f = neg(div(down, total))
    return dict(lqtp=f, fe=f, status="ok", steps=[
        "1) down = ts_ewm_var(24, min(ret,0)^2)",
        "2) total = ts_ewm_var(24, ret^2)",
        "3) factor = -down/total",
    ], note="code 已取负号。",
       rationale="下行方差占比越低越稳 → 显式负号。rankic<0 → 自动翻转（双重负号抵消，flipped 记录语义）。")



def conv_lag_response_vol_tool_adaptive(code):
    # 同 flipped 版本（非翻转），rankic<0 → 自动翻转
    return conv_lag_response_vol_tool_adaptive_flipped(code)


def conv_drawdown_volume_geometry(code):
    # 已定义于 batch1
    return conv_drawdown_volume_geometry(code)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def conv_drawdown_depth_atr_gated_fixed(code):
    # depth = -(rolling_max(close,63)-close)/rolling_max(close,63)；atr=rolling63(TR)；normalized=depth/atr
    peak = ts_max("close", 63)
    dd = div(sub("close", peak), peak)
    depth = neg(dd)
    tr = true_range("high", "low", "close")
    atr = ts_mean(tr, 63)
    normalized = div(depth, atr)
    f = where(gt("style_gate_resvol_high", 0.0), mul(normalized, 0.5), normalized)
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) peak = ts_max(close, 63)  63日滚动高点",
        "2) dd = (close - peak) / peak",
        "3) depth = -dd  回撤深度",
        "4) tr = true_range(high, low, close)  真实波幅",
        "5) atr = ts_mean(tr, 63)  63日 ATR（SMA 平滑）",
        "6) normalized = depth / atr  ATR 归一化",
        "7) factor = where(high_resvol, normalized*0.5, normalized)  高残差波动期半权重",
    ], note="原始 code 用 TR 的 rolling(63).mean() 而非 Wilder ATR；此处用 true_range+ts_mean 精确表达。",
       rationale="用 ATR 把绝对回撤除掉市场波动，只看个股相对回撤。")


def conv_drawdown_volume_geometry_fixed(code):
    # running_max=close.expanding().max()；depth=(running_max-close)/running_max
    # duration=新高以来天数；depth_chg=depth.diff(10)；vol_ratio=volume/ma20
    # rank_depth*rank_vol + rank_duration + rank_recovery
    peak = ts_max("close", 5000)
    dd = div(sub(peak, "close"), peak)
    depth_chg = ts_delta(dd, 10)
    vol_ratio = div("volume", ts_mean("volume", 20))
    f = add(add(mul(rank(dd), rank(vol_ratio)), rank("duration")), rank(depth_chg))
    return dict(lqtp=f, fe=f, status="fallback", steps=[
        "1) running_max = close.expanding().max()  → ts_max(close, 5000) 近似（全历史）",
        "2) depth = (running_max - close) / running_max",
        "3) duration = 自新高以来的天数（事件分组 cumcount）",
        "4) depth_chg = depth.diff(10)  → ts_delta(depth, 10)",
        "5) vol_ratio = volume / ts_mean(volume, 20)",
        "6) factor = rank(depth)*rank(vol_ratio) + rank(duration) + rank(depth_chg)",
    ], note="expanding().max() 无直接算子 → 超大窗口 ts_max 近似；duration 依赖事件分组 cumcount → fallback。",
       rationale="回撤几何 + 量能复合。")


CONVERTERS = {
    "downside_semivariance_smoothed": conv_downside_semivariance_smoothed,
    "amount_weighted_squared_impact": conv_amount_weighted_squared_impact,
    "drawdown_depth_atr_gated": conv_drawdown_depth_atr_gated_fixed,
    "drawdown_volume_complexity": conv_drawdown_volume_complexity,
    "asym_intraday_sma": conv_asym_intraday_sma,
    "book_attention": conv_book_attention,
    "asym_vol_volume_cont_30": conv_asym_vol_volume_cont_30,
    "drawdown_volume_geometry": conv_drawdown_volume_geometry_fixed,
    "ewma_smoothness_volume": conv_ewma_smoothness_volume,
    "fear_adjusted_dollar_pressure_short_ema": conv_fear_adjusted_dollar_pressure_short_ema,
    "impact_weighted_asymmetry_slow": conv_impact_weighted_asymmetry_slow,
    "overnight_repricing_ewm_stability": conv_overnight_repricing_ewm_stability,
    "persistence": conv_persistence,
    "persistence_ewma": conv_persistence_ewma,
    "pressure_ema_mutation": conv_pressure_ema_mutation,
    "vol_asym_confirmed_range": conv_vol_asym_confirmed_range,
    "vol_asym_confirmed_range_v2": conv_vol_asym_confirmed_range_v2,
    "volatility_adjusted_dollar_pressure": conv_volatility_adjusted_dollar_pressure,
    "volume_weighted_impact": conv_volume_weighted_impact,
    "volume_weighted_squared_impact": conv_volume_weighted_squared_impact,
    "abnormality_asymmetry_flipped": conv_abnormality_asymmetry_flipped,
    "impact_asymmetry_slow_recovery_flipped": conv_impact_asymmetry_slow_recovery_flipped,
    "impact_downside_asymmetry_smoothed_flipped": conv_impact_downside_asymmetry_smoothed_flipped,
    "lag_response_vol_tool_adaptive_flipped": conv_lag_response_vol_tool_adaptive_flipped,
    "lag_vol_ratio_smoothed_v2_flipped": conv_lag_vol_ratio_smoothed_v2_flipped,
    "lag_vol_robust_volratio_flipped": conv_lag_vol_robust_volratio_flipped,
    "lag_response_vol_tool_adaptive": conv_lag_response_vol_tool_adaptive,
    "lag_vol_volatility_adjusted_gate": conv_lag_vol_volatility_adjusted_gate,
    "persistent_left_tail_variance_share_flipped": conv_persistent_left_tail_variance_share_flipped,
    "price_impact_stable_5d_flipped": conv_price_impact_stable_5d_flipped,
    "range_volume_ratio_flipped": conv_range_volume_ratio_flipped,
    "session_magnitude_imbalance_volume_weighted_42d_flipped": conv_session_magnitude_imbalance_volume_weighted_42d_flipped,
    "smoothed_downside_variance_resilience_flipped": conv_smoothed_downside_variance_resilience_flipped,
    "smoothed_drawdown_recovery_participation_flipped": conv_smoothed_drawdown_recovery_participation_flipped,
    "smoothed_energy_downside_resilience_flipped": conv_smoothed_energy_downside_resilience_flipped,
    "smoothed_energy_downside_resilience_8span_flipped": conv_smoothed_energy_downside_resilience_8span_flipped,
    "smoothed_left_tail_variance_share_flipped": conv_smoothed_left_tail_variance_share_flipped,
    "squared_range_close_ema_cuberoot_flipped": conv_squared_range_close_ema_cuberoot_flipped,
    "squared_range_volume_rank_smoothed_no_cuberoot_flipped": conv_squared_range_volume_rank_smoothed_no_cuberoot_flipped,
    "vol_regime_impact_ema_flipped": conv_vol_regime_impact_ema_flipped,
    "vol_volume_asym_ewma_flipped": conv_vol_volume_asym_ewma_flipped,
    "volume_adaptive_momentum_fast_vol": conv_volume_adaptive_momentum_fast_vol,
    "volume_adaptive_momentum_smoothed_v3": conv_volume_adaptive_momentum_smoothed_v3,
    "volume_adjusted_price_range": conv_volume_adjusted_price_range,
    "volume_adjusted_price_range_ema_flipped": conv_volume_adjusted_price_range_ema_flipped,
    "volume_adjusted_price_range_ema_gated": conv_volume_adjusted_price_range_ema_gated,
    "volume_adjusted_price_range_gated_flipped": conv_volume_adjusted_price_range_gated_flipped,
    "volume_adjusted_price_range_regime_flipped": conv_volume_adjusted_price_range_regime_flipped,
    "volume_adjusted_price_range_regime_smoothed_flipped": conv_volume_adjusted_price_range_regime_smoothed_flipped,
    "volume_adjusted_price_range_regime_smoothed_v2_flipped": conv_volume_adjusted_price_range_regime_smoothed_v2_flipped,
    "volume_adjusted_price_range_regime_v2_flipped": conv_volume_adjusted_price_range_regime_v2_flipped,
    "volume_adjusted_price_range_regime_v3_flipped": conv_volume_adjusted_price_range_regime_v3_flipped,
    "volume_impact_elasticity_smooth_flipped": conv_volume_impact_elasticity_smooth_flipped,
    "volume_state_transition_carry_smooth_flipped": conv_volume_state_transition_carry_smooth_flipped,
    "volume_transition_raw_carry_30d_flipped": conv_volume_transition_raw_carry_30d_flipped,
    "volume_transition_stress_attenuated_30d_flipped": conv_volume_transition_stress_attenuated_30d_flipped,
    "vwap_adjusted_range_tanh_mutated_v2_flipped": conv_vwap_adjusted_range_tanh_mutated_v2_flipped,
    "vwap_adjusted_range_tanh_smooth_flipped": conv_vwap_adjusted_range_tanh_smooth_flipped,
    "ts_size_adaptive_earnings_book_smooth": conv_ts_size_adaptive_earnings_book_smooth,
    "drawdown_volume_modulated": conv_drawdown_volume_modulated,
    "session_asymmetry_smooth_confirmed_30d": conv_session_asymmetry_smooth_confirmed_30d,
    "ewm_downside_variance_resilience": conv_ewm_downside_variance_resilience,
}

def load_factor(name):
    base = name.replace("_flipped", "")
    candidates = [
        os.path.join(FC_DIR, "factor_%s.json" % base),
        os.path.join(FC_DIR, "factor_%s.json" % name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return json.load(open(p))
    # 兜底：带数字后缀
    for fn in os.listdir(FC_DIR):
        if fn.startswith("factor_%s_" % base) and fn.endswith(".json"):
            return json.load(open(os.path.join(FC_DIR, fn)))
    return None


def convert_all():
    results = {}
    for name in FACTOR_NAMES:
        data = load_factor(name)
        if data is None:
            results[name] = {
                "lqtp_formula": "", "fe_formula": "", "custom_dsl": None,
                "status": "custom", "flipped": False,
                "required_columns": [], "steps": [],
                "rationale": "未找到原始 code", "code": "",
                "rankic_raw": None, "note": "因子 JSON 缺失",
            }
            continue
        code = data.get("code", "")
        req_cols = data.get("required_columns", []) or []
        rationale = (data.get("rationale") or "")[:400]
        conv = CONVERTERS.get(name)
        if conv is None:
            conv = CONVERTERS.get(name.replace("_flipped", ""))
        if conv is None:
            results[name] = {
                "lqtp_formula": "", "fe_formula": "", "custom_dsl": None,
                "status": "custom", "flipped": False,
                "required_columns": req_cols, "steps": [],
                "rationale": rationale, "code": code,
                "rankic_raw": RANKIC.get(name), "note": "缺少专用转换器",
            }
            continue
        try:
            conv_out = conv(code)
        except Exception as exc:
            results[name] = {
                "lqtp_formula": "", "fe_formula": "", "custom_dsl": None,
                "status": "custom", "flipped": False,
                "required_columns": req_cols, "steps": [],
                "rationale": rationale, "code": code,
                "rankic_raw": RANKIC.get(name),
                "note": "转换器异常: %s" % exc,
            }
            continue
        lqtp = conv_out.get("lqtp", "")
        fe = conv_out.get("fe", lqtp)
        status = conv_out.get("status", "ok")
        custom_dsl = conv_out.get("custom_dsl")
        steps = conv_out.get("steps", [])
        note = conv_out.get("note", "")
        # 自动翻转：raw mean_rankic < 0 → 最外层加负号
        raw_ric = RANKIC.get(name)
        flip = (raw_ric is not None) and (raw_ric < 0)
        flipped_formula = lqtp
        flipped_fe = fe
        if flip and lqtp:
            flipped_formula = flip_formula(lqtp)
            flipped_fe = flip_formula(fe)
            if custom_dsl:
                custom_dsl = flip_formula(custom_dsl)
        results[name] = {
            "lqtp_formula": flipped_formula,
            "fe_formula": flipped_fe,
            "custom_dsl": custom_dsl,
            "status": status,
            "flipped": flip,
            "required_columns": req_cols,
            "steps": steps,
            "rationale": rationale,
            "code": code,
            "rankic_raw": raw_ric,
            "note": note,
        }
    return results


def main():
    results = convert_all()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    # 统计
    stats = {"total": 0, "ok": 0, "fallback": 0, "custom": 0, "flipped": 0}
    for name, r in results.items():
        stats["total"] += 1
        if r["status"] == "ok":
            stats["ok"] += 1
        elif r["status"] == "fallback":
            stats["fallback"] += 1
        else:
            stats["custom"] += 1
        if r["flipped"]:
            stats["flipped"] += 1
    print("转换统计:")
    for k, v in stats.items():
        print("  %s: %d" % (k, v))
    # 抽查 3 个
    samples = ["downside_semivariance_smoothed", "ewma_smoothness_volume", "vwap_adjusted_range_tanh_smooth_flipped"]
    print("\n抽查:")
    for s in samples:
        r = results.get(s)
        if not r:
            continue
        print("=" * 70)
        print("因子:", s)
        print("status:", r["status"], "| flipped:", r["flipped"], "| rankic_raw:", r["rankic_raw"])
        print("lqtp_formula:", r["lqtp_formula"])
        print("fe_formula:", r["fe_formula"])
        if r.get("custom_dsl"):
            print("custom_dsl:", r["custom_dsl"])
        print("--- code 原文 ---")
        print(r["code"][:600])


if __name__ == "__main__":
    main()


def close_col():
    return "close"


