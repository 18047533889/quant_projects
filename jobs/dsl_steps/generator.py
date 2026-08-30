# -*- coding: utf-8 -*-
"""DSL 计算步骤生成器核心实现。

输入：因子名 / dsl / python code / is_flipped。
输出：HTML 片段（<div class="card">…</div>），含「计算步骤」有序列表 + 说明框。

三步策略：
  1. 手写步骤（factor_manual_61 61 条 / formula_lqtp.json 62 条）
  2. 结构化翻译（pandas code 语句级，能覆盖绝大多数 fallback DSL）
  3. 算子正则式（对 DSL 中出现的算子做逐个释义，兜底）
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROJECT = Path("/home/sunhaiwei/quant_projects")
_CN_FONT = "/home/sunhaiwei/.fonts/NotoSansSC-Regular.otf"

# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------
_LQTP_ALL = Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json")
_LQTP_61 = Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp.json")
_MANUAL = Path("/home/sunhaiwei/quant_projects/scripts/archive/jobs/factor_manual_61.py")

_manual_cache: dict | None = None
_lqtp61_cache: dict | None = None
_all_cache: list | None = None


def _load_all() -> list:
    global _all_cache
    if _all_cache is None:
        _all_cache = json.loads(_LQTP_ALL.read_text(encoding="utf-8"))
    return _all_cache


def _load_lqtp61() -> dict:
    global _lqtp61_cache
    if _lqtp61_cache is None:
        _lqtp61_cache = json.loads(_LQTP_61.read_text(encoding="utf-8"))
    return _lqtp61_cache


def _load_manual() -> dict:
    """动态导入 factor_manual_61.FACTOR_MANUAL，避免硬依赖。"""
    global _manual_cache
    if _manual_cache is None:
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("_fm61", _MANUAL)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _manual_cache = dict(getattr(mod, "FACTOR_MANUAL", {}) or {})
        except Exception:
            _manual_cache = {}
    return _manual_cache


def _get_record(page: str) -> dict | None:
    """从 formula_lqtp_all.json 找 page 的记录。"""
    for e in _load_all():
        if e.get("page_name") == page:
            return e
    return None


# ---------------------------------------------------------------------------
# 第一层：手写步骤
# ---------------------------------------------------------------------------
def _manual_steps(page: str) -> list[str] | None:
    """优先取 factor_manual_61（按 base 名匹配 _flipped 后缀）。"""
    manual = _load_manual()
    base = page[:-8] if page.endswith("_flipped") else page
    rec = manual.get(page) or manual.get(base)
    if rec and rec.get("steps"):
        return rec["steps"]
    return None


def _lqtp61_steps(page: str) -> list[str] | None:
    """次优取 formula_lqtp.json 的 steps（62 条，base 名匹配 flipped）。"""
    d61 = _load_lqtp61()
    base = page[:-8] if page.endswith("_flipped") else page
    rec = d61.get(page) or d61.get(base)
    if rec and rec.get("steps"):
        return rec["steps"]
    return None


# ---------------------------------------------------------------------------
# 字段 / 算子释义表（陌生算子才解释）
# ---------------------------------------------------------------------------
BASE_FIELDS = {"open", "high", "low", "close", "volume", "amount", "vwap",
               "pre_close", "adj_factor", "is_suspend", "high_limit", "low_limit"}

# 常见中文语义 → 简单算子（不需要展开成步骤，但需要解释的）
_FIELD_GLOSS = {
    "ret": "日收益，(close - 前一日close) / 前一日close",
    "ret_5": "5 日收益，(close - 5日前close) / 5日前close",
    "vol_ratio": "量比，volume / N日均量，>1 放量、<1 缩量",
    "vol_ma20": "20 日均量，ts_mean(volume, 20)",
    "vol_ma40": "40 日均量，ts_mean(volume, 40)",
    "log_volume": "成交量取自然对数，压缩极端值",
    "log_close": "收盘价取自然对数",
    "log_return": "对数收益，log(close / 前日close)",
    "drawdown": "回撤，(close - N日最高close) / N日最高close",
    "overnight_ret": "隔夜收益，(open - 前日close) / 前日close",
    "intraday_ret": "日内收益，(close - open) / open",
    "downside_var": "下行方差，min(ret,0)^2 的滚动均值",
    "total_var": "总方差，ret^2 的滚动均值",
    "down_vol": "下行波动率，ret<0 部分的滚动标准差",
    "up_vol": "上行波动率，ret>0 部分的滚动标准差",
    "sigma_down": "下行波动率（同 down_vol）",
    "sigma_up": "上行波动率（同 up_vol）",
    "ewma_up_vol": "上行波动 EMA 平滑（成交量/振幅上行侧的指数均值）",
    "ewma_down_vol": "下行波动 EMA 平滑（下行侧的指数均值）",
    "trailing_vol": "滞后波动率，前 N 期的波动率（避免未来信息）",
    "trailing_max": "滞后滚动最高，前 N 期最高价",
    "lag_ret": "滞后收益，N 期前的收益",
    "lag_vol": "滞后量，N 期前的成交量",
    "momentum": "动量，N 期收益",
    "persistence": "持续性，信号延续度（时间序列自相关方向）",
    "directional_efficiency": "方向效率，|净位移| / 路径长度",
    "range_pct": "日内振幅率，(high - low) / close",
    "typical_price": "典型价，(high + low + close) / 3",
    "depth": "回撤深度，(N日高点 - close) / N日高点",
    "duration": "回撤持续天数，自 N 日高点以来的天数",
    "recovery": "恢复强度，回撤后反弹幅度 / 回撤深度",
    "atr_20": "ATR 平均真实波幅(20日)",
    "score": "中间信号，复合计算的过程值",
    "smoothed": "平滑后的值（EMA/滚动平滑）",
    "normalized": "归一化后的值",
    "zscore": "Z-score，(x - mean) / std",
    "resvol_high": "高残差波动期标志",
    "high_resvol": "高残差波动期标志（同 resvol_high）",
    "low_volume_regime": "低量能状态标志",
    "is_high_volume": "高量能日标志",
    "is_low_volume": "低量能日标志",
    "vol_dev": "量能偏离度，(volume - 量均线) / 量均线",
    "avg_down_range": "下行日平均振幅，(high-low) 在下跌日的均值",
    "avg_up_range": "上行日平均振幅，(high-low) 在上涨日的均值",
    "free_turn": "自由换手率",
    "turnover": "换手率",
    "pb_lf": "市净率（最新披露），PB = price / 每股净资产",
    "pe_ttm": "市盈率 TTM",
    "ps_ttm": "市销率 TTM",
    "eps_ttm": "每股收益 TTM",
    "roe_ttm": "ROE 净资产收益率 TTM",
    "debttoassets": "资产负债率，总负债 / 总资产",
    "style_gate_resvol_high": "高残差波动风格门控标志",
    "vol": "成交量（简写，同 volume）",
    "tr": "真实波幅，(high-low) 与 |high-前收|、|low-前收| 的最大值",
    "true_range": "真实波幅（同 tr）",
    "c": "收盘价（简写，同 close）",
    "o": "开盘价（简写，同 open）",
    "h": "最高价（简写，同 high）",
    "l": "最低价（简写，同 low）",
    "v": "成交量（简写，同 volume）",
    "close_price": "收盘价（别名，同 close）",
    "open_price": "开盘价（别名，同 open）",
    "high_price": "最高价（别名，同 high）",
    "low_price": "最低价（别名，同 low）",
    "amount": "成交额（元）",
    "raw": "原始信号（未平滑/未标准化）",
}

_OP_GLOSS = {
    "ts_mean": "滚动均值：过去 N 天的平均值",
    "ts_std": "滚动标准差：过去 N 天的波动程度",
    "ts_ema": "指数加权移动平均：近期权重更大的均值",
    "ts_ewm_std": "指数加权滚动标准差：近期权重更大的波动",
    "ewm_mean": "指数加权移动平均：近期权重更大的均值",
    "ewm_std": "指数加权滚动标准差",
    "ts_delay": "滞后 N 期：取 N 天前的值（避免未来信息）",
    "delay": "滞后 N 期（同 ts_delay）",
    "ts_pct": "收益率：x[t]/x[t-N] - 1",
    "ts_rank": "过去 N 天内的百分位排名 (0~1)",
    "ts_quantile": "过去 N 天的分位数",
    "ts_max": "过去 N 天的滚动最大值",
    "ts_min": "过去 N 天的滚动最小值",
    "ts_sum": "过去 N 天的滚动求和",
    "ts_median": "过去 N 天的滚动中位数",
    "ts_skew": "过去 N 天的滚动偏度（分布不对称程度）",
    "ts_kurt": "过去 N 天的滚动峰度（尾部厚度）",
    "ts_argmax": "过去 N 天内最大值的位置",
    "ts_argmin": "过去 N 天内最小值的位置",
    "ts_delta": "N 期差分：x[t] - x[t-N]",
    "ts_corr": "滚动相关系数：两个序列过去 N 天的相关",
    "ts_cov": "滚动协方差",
    "ts_regression_slope": "滚动回归斜率：过去 N 天线性回归的斜率",
    "ts_decay_linear": "线性衰减加权：越近权重越大",
    "ts_atr_wilder": "Wilder ATR：平均真实波幅",
    "ts_log_return": "对数收益率：log(x[t]/x[t-1])",
    "ts_zscore": "滚动 Z-score：(x - 均值) / 标准差",
    "zscore": "Z-score 标准化：(x - 均值) / 标准差",
    "rank": "截面分位排名 (0~1)",
    "cs_rank": "截面分位排名（同 rank）",
    "cs_zscore": "截面 Z-score：当日横截面标准化",
    "cs_weighted_mean": "截面加权平均",
    "cs_pct_rank": "截面百分位排名",
    "group_weighted_mean": "分组加权平均",
    "group_mean": "分组均值",
    "group_rank": "分组内排名",
    "group_zscore": "分组内 Z-score",
    "winsorize": "极值裁剪：把超出分位阈值的值拉回阈值",
    "clip": "数值裁剪到 [下限, 上限]",
    "where": "条件取值：满足条件取 X，否则取 Y",
    "abs": "绝对值",
    "log": "自然对数 ln(x)",
    "sqrt": "平方根",
    "exp": "指数 e^x",
    "pow": "幂运算 x^n",
    "power": "幂运算（同 pow）",
    "sign": "符号函数：正数→1，负数→-1",
    "tanh": "双曲正切：把值压缩到 (-1, 1)",
    "sigmoid": "Sigmoid：把值压缩到 (0, 1)",
    "min": "取最小值",
    "max": "取最大值",
    "mean": "均值",
    "sum": "求和",
    "std": "标准差",
    "rolling_mean": "滚动均值（同 ts_mean）",
    "rolling_std": "滚动标准差（同 ts_std）",
    "rolling_max": "滚动最大值（同 ts_max）",
    "rolling_sum": "滚动求和（同 ts_sum）",
    "rolling_vwap": "滚动 VWAP：成交额加权均价",
    "rank_corr": "滚动秩相关",
    "true_range": "真实波幅：(high-low) 与 |high-前收|、|low-前收| 取最大",
    "corr": "相关系数（同 ts_corr）",
    "correlation": "相关系数（同 ts_corr）",
    "ADX": "平均趋向指数（趋势强度）",
    "overnight_return": "隔夜收益：(open - 前收) / 前收",
    "overnight_volatility": "隔夜波动率",
    "vwap_to_close_return": "VWAP 到收盘的收益",
    "volume_momentum": "量能动量",
    "volume_zscore": "量能 Z-score",
    "returns": "收益率（同 ts_pct）",
    "shift": "滞后/超前（同 ts_delay）",
    "fillna": "缺失值填充",
    "replace": "值替换",
    "pct_change": "变化率：x[t]/x[t-1] - 1",
    "ewm": "指数加权移动平均",
    "sma": "简单移动平均",
    "ema": "指数移动平均",
    "SMA": "简单移动平均（同 ts_mean）",
    "EMA": "指数移动平均（同 ts_ema）",
    "ATR": "平均真实波幅",
    "STDDEV": "滚动标准差",
    "rank_ts": "时间序列排名",
    "rank_trailing": "滚动窗口排名",
    "m_zscore": "滚动 Z-score（同 ts_zscore）",
    "m_skew": "滚动偏度（同 ts_skew）",
    "cummax": "累计最大值（历史最高）",
    "skew": "偏度",
    "std_20": "20 日滚动标准差",
    "mean_20": "20 日滚动均值",
    "sum_20": "20 日滚动求和",
    "max_20": "20 日滚动最大值",
    "median_20": "20 日滚动中位数",
    "vol_ratio_ema": "量比 EMA 平滑",
    "robust_z": "稳健 Z-score",
    "consensus": "一致性指标",
    "daily_last": "日内最后一个值",
    "daily_std": "日内标准差",
    "daily_skew": "日内偏度",
    "daily_kurt": "日内峰度",
    "classify_volume_regime": "量能状态分类（高/低量能）",
    "decompose_overnight_intraday": "隔夜+日内收益拆解",
    "daily_corr": "日内（或逐日截面）相关系数",
    "delta": "差分：x[t] - x[t-N]",
    "get": "数据列引用",
    "ifnan": "若为 NaN 则取给定值",
    "trailing_max": "滞后滚动最高",
    "rolling_skew": "滚动偏度",
    "rolling_zscore": "滚动 Z-score",
    "trailing_max": "滞后滚动最高",
    "lag": "滞后（同 ts_delay）",
    "lag_ret": "滞后收益",
    "ret": "日收益",
    "vol_ratio": "量比",
    "momentum": "动量",
    "persistence": "持续性",
    "directional_efficiency": "方向效率",
    "downside_ratio": "下行占比",
    "downside_stress_ratio": "下行压力比",
    "drawdown": "回撤",
    "rank_corr": "秩相关",
    "normalized": "归一化",
    "smoothed": "平滑",
    "score": "中间信号",
    "amount_weighted_mean": "成交额加权平均",
    "cs_rank": "截面排名",
    "ts_atr": "平均真实波幅",
    "ts_rsi": "RSI 相对强弱指标",
    "EWMA_up_vol": "上行波动 EMA",
    "EWMA_down_vol": "下行波动 EMA",
    "EWMA": "指数加权移动平均",
    "ewma": "指数加权移动平均",
    "MA": "移动平均",
    "MA_volume": "成交量均线",
    "SMA_21": "21 日简单均线",
    "SMA10": "10 日简单均线",
    "SMA20": "20 日简单均线",
    "SMA90": "90 日简单均线",
    "SMA_10": "10 日简单均线",
    "SMA_5": "5 日简单均线",
    "CV": "变异系数：标准差 / 均值（波动相对水平）",
    "rolling": "滚动窗口",
}

# 语法级 token（不解释为算子）
_SKIP_OPS = {"if", "else", "then", "where"}


def _is_base_field(name: str) -> bool:
    return name in BASE_FIELDS or name in {"open", "high", "low", "close", "volume"}


def _gloss_field(name: str) -> str | None:
    return _FIELD_GLOSS.get(name)


def _gloss_op(name: str) -> str | None:
    return _OP_GLOSS.get(name)


# ---------------------------------------------------------------------------
# 第二层：结构化翻译（pandas code 语句级）
# ---------------------------------------------------------------------------
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_FINAL_RE = re.compile(r"return\s+df_copy\s*\[\s*['\"]([^'\"]+)['\"]\s*\]")


def _simplify_expr(expr: str) -> str:
    """把 pandas 代码压成一行可读文本。"""
    t = expr.strip()
    t = re.sub(r"df_copy\s*\[", "", t)
    t = re.sub(r"\]", "", t)
    t = t.replace("np.nan", "空")
    # 去掉列名引号：'volume' → volume
    t = re.sub(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", r"\1", t)
    t = re.sub(r"\.replace\s*\([^)]*\)", "", t)
    t = re.sub(r"\.fillna\s*\([^)]*\)", "", t)
    t = re.sub(r"\.rolling\s*\(\s*(\d+)", r".滚动窗口(\1", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" ,")


def _extract_code_steps(code: str) -> list[str] | None:
    """从 Python code 逐行解析赋值语句 → 计算步骤。"""
    if not code:
        return None
    lines = code.split("\n")
    steps: list[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("def ") or s.startswith("#") or s.startswith("return"):
            continue
        if "=" not in s:
            continue
        m = _ASSIGN_RE.match(s)
        if not m:
            continue
        var, expr = m.group(1), m.group(2)
        if var.startswith("df_copy"):
            continue
        if re.match(r"^(import|from)\s", s):
            continue
        # 跳过 = 单值（如 span = 10）这类无信息量的赋值
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", expr.strip()):
            continue
        # 跳过纯重命名：safe_range = range_ 或 factor = factor
        if expr.strip() == var or (re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", expr.strip()) and expr.strip() in {var, "range_", "factor"}):
            continue
        # 简化
        simp = _simplify_expr(expr)
        if len(simp) < 2 or simp in {"copy()"}:
            continue
        steps.append(f"计算 {var}：{simp}")
    if not steps:
        return None
    return steps


# ---------------------------------------------------------------------------
# 第三层：算子正则式（兜底）
# ---------------------------------------------------------------------------
_OP_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\(")


def _dsl_op_steps(dsl: str) -> list[str] | None:
    """对 DSL 中出现的算子逐个释义。"""
    if not dsl or len(dsl) < 3:
        return None
    ops: list[str] = []
    seen = set()
    for m in _OP_CALL_RE.finditer(dsl):
        op = m.group(1)
        if op in seen or op in _SKIP_OPS:
            continue
        seen.add(op)
        gloss = _gloss_op(op)
        if gloss is None:
            continue
        ops.append(f"算子 {op}：{gloss}")
    if not ops:
        return None
    return ops


# ---------------------------------------------------------------------------
# 汇总步骤（保证顺序：手写 → 结构化 → 算子兜底）
# ---------------------------------------------------------------------------
def resolve_steps(page: str, dsl: str = "", code: str = "") -> tuple[list[str], str]:
    """返回 (steps, note)。steps 为空表示没有可展示的步骤。"""
    note = ""

    # 1) 手写步骤
    ms = _manual_steps(page)
    if ms:
        return ms, "手写步骤（factor_manual_61 / formula_lqtp 人工核对）"

    # 2) 结构化 code
    if code:
        cs = _extract_code_steps(code)
        if cs:
            note = "由原始 Python 代码逐句翻译（字段/算子释义见下方）"
            return cs, note

    # 3) DSL 算子释义
    if dsl:
        if dsl.strip() == "evoalpha_dsl_see_land_evoalpha14_build_formulas":
            return [
                "该因子为本周新挖（EvoAlpha14），公式注册在 jobs/land_evoalpha14.build_formulas() 的 DSL 表达式。",
                "计算步骤由 factor_engine 落值管线执行，详情见 jobs/land_evoalpha14.py 中对应因子的表达式。",
            ], "EvoAlpha14 本周新挖因子（见 jobs/land_evoalpha14.py）"
        os_ = _dsl_op_steps(dsl)
        if os_:
            note = "由 DSL 表达式中的算子自动生成（逐算子释义）"
            return os_, note

    return [], ""


# ---------------------------------------------------------------------------
# HTML 组装
# ---------------------------------------------------------------------------
def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_steps_block(page: str, dsl: str = "", code: str = "", is_flipped: bool = False) -> str:
    """构建「计算步骤」卡片 HTML。返回空串表示无步骤。"""
    steps, note = resolve_steps(page, dsl, code)
    if not steps:
        return ""

    li = "".join(f"<li>{_esc(s)}</li>" for s in steps)
    flip_note = ""
    if is_flipped:
        flip_note = (
            '<div style="margin-top:10px;padding:8px 12px;background:#fef9c3;border-left:3px solid #eab308;'
            'border-radius:6px;font-size:0.8rem;color:#854d0e">'
            "⚠ 该因子原始 IC 为负，已整体取反（×(-1)）调正方向；下列步骤展示的是原始公式的计算过程。</div>"
        )
    note_html = ""
    if note:
        note_html = (
            '<div style="margin-top:10px;font-size:0.78rem;color:var(--muted)">'
            f"来源：{_esc(note)}</div>"
        )

    return (
        '\n<!-- 计算步骤 -->\n'
        '<div class="card">\n'
        '<h2>📝 计算步骤（一步一步解释）</h2>\n'
        '<ol style="line-height:1.9;padding-left:24px;font-size:0.92rem">'
        + li +
        '</ol>\n'
        + flip_note
        + note_html +
        '</div>\n'
    )
