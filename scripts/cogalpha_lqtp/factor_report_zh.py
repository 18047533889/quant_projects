#!/usr/bin/env python3
"""Plain-Chinese copy for factor detail HTML: formula steps, operators, chart guides."""
from __future__ import annotations

import html as html_lib
import re
from typing import Any

# 算子中文说明（详情页「公式里的算子」小节复用）
OPERATOR_ZH: dict[str, str] = {
    "ts_pct": "n 日收益率：x 相对 n 天前的涨跌幅，即 x/delay(x,n) − 1",
    "ts_mean": "过去 n 个交易日的算术滑动平均",
    "ema": "指数加权平均，越近的日期权重越大",
    "ts_std": "过去 n 日样本标准差，衡量波动大小",
    "ts_delta": "当前值减去 n 天前的值",
    "delay": "取 n 个交易日之前的数值（滞后）",
    "ts_max": "过去 n 日内的最高值",
    "ts_sum": "过去 n 日累加求和",
    "rank": "同一交易日全市场截面排序，映射为 0~1 分位",
    "zscore": "(值 − 均值) / 标准差，用于标准化",
    "safe_div": "安全除法：分母为 0 或缺失时返回空，避免除零",
    "where": "条件选择：条件成立取第一项，否则取第二项",
    "log": "自然对数，常用于比值或压缩尺度",
    "abs": "绝对值",
    "tanh": "双曲正切，把数值压缩到 (−1, 1) 区间",
    "cap": "下限截断，防止 log(0) 等数值问题",
    "ATR": "平均真实波幅：n 日真实波幅的滑动平均，衡量价格波动",
    "ROC": "n 日变化率，同 ts_pct",
    "volume": "成交量",
    "high-low": "当日振幅 = 最高价 − 最低价",
}

try:
    from scripts.cogalpha_lqtp.weekly_dug_factor_guides import WEEKLY_DUG_FACTOR_GUIDES
except ImportError:  # script-dir import
    from weekly_dug_factor_guides import WEEKLY_DUG_FACTOR_GUIDES  # type: ignore

WEEK3_FACTOR_GUIDES: dict[str, dict[str, Any]] = {
    "factor_persistence_ewma": {
        "theme": "收益路径平滑度（指数加权）",
        "summary": "衡量最近几天涨跌幅是否连贯：相邻两日收益率变化越小，走势越「顺滑」，因子值越高。",
        "steps": [
            "计算日收益率 r = close 相对昨收的涨跌幅（ts_pct(close,1)）",
            "计算相邻两日收益率之差的绝对值 |r − 昨日r|（刻画方向是否频繁反转）",
            "对上述序列做 10 日指数加权平均 ema（近期权重更大）",
            "最后取负号：变化越小 → 因子值越高",
        ],
        "direction": "因子值高：近期收益路径较平滑、方向较一致；因子值低：涨跌切换频繁、噪声大。",
        "operators": ["ts_pct", "ts_delta", "abs", "ema"],
    },
    "factor_vol_regime_vol_ratio": {
        "theme": "高成交量下的振幅收缩",
        "summary": "在放量日子里，若短期振幅相对长期收窄，认为有资金介入后的「整理」，再经截面排序。",
        "steps": [
            "5 日平均振幅 = ts_mean(high−low, 5)；20 日平均振幅 = ts_mean(high−low, 20)",
            "计算 −(5日振幅/20日振幅)：短期相对长期收窄时为正",
            "门控：仅当 volume > 30 日均量 × 1.5 时保留信号",
            "全市场截面 rank，得到 0~1 分位",
        ],
        "direction": "因子值高：放量且振幅相对收缩；因子值低：放量但波动仍在放大或量能不足。",
        "operators": ["ts_mean", "rank", "where", "high-low", "volume"],
    },
    "factor_persistence": {
        "theme": "收益路径平滑度（简单平均）",
        "summary": "与 persistence_ewma 相同逻辑，但用等权 10 日滑动平均代替 ema，对近期突变不那么敏感。",
        "steps": [
            "日收益率 r = close 相对昨收的涨跌幅",
            "相邻两日收益率变化 |r − 昨日r|",
            "10 日滑动平均 ts_mean 后取负号",
        ],
        "direction": "因子值高：收益路径更连贯；因子值低：方向切换多。",
        "operators": ["ts_pct", "ts_delta", "abs", "ts_mean"],
    },
    "factor_volume_abnormality_liquidity_gated": {
        "theme": "成交量异常 × 流动性门控",
        "summary": "衡量成交量相对自身中位水平的偏离，但流动性太差的股票直接置零。",
        "steps": [
            "当日成交量 ÷ 20 日成交量中位数，得到相对放量/缩量程度",
            "若 20 日平均成交额低于流动性阈值，因子 = 0（过滤不可交易标的）",
            "否则输出成交量偏离得分",
        ],
        "direction": "因子值高：在可交易股票中明显放量；门控为 0 表示流动性不足被剔除。",
        "operators": ["ts_mean", "volume"],
    },
    "factor_lag_vol_volatility_adjusted_gate": {
        "theme": "滞后收益 × 对数成交量 × 波动调整（风格门控）",
        "summary": "5 日价格变化用成交量和 ATR 放大/缩小，高残差波动环境下加大平滑。",
        "steps": [
            "5 日收益率 = close/5日前close − 1",
            "对数(成交量相对 30 日均量的比值)",
            "ATR(20)/close 作为波动率缩放因子",
            "三者相乘后 EMA 平滑；高残差波动风格下用更长平滑窗口",
        ],
        "direction": "因子值高：滞后涨跌幅在放量、波动放大时被强化；门控会削弱噪声环境下的信号。",
        "operators": ["delay", "log", "ATR", "ema", "volume"],
    },
    "factor_lag_response_vol_tool_adaptive": {
        "theme": "量价滞后响应（自适应平滑）",
        "summary": "5 日收益 × log 量比 × ATR/close；高噪环境自动用更长 EMA。",
        "steps": [
            "5 日收益率 × log(成交量状态比值) × ATR(20)/close 得原始信号",
            "残差波动高：用 10 日 EMA；否则用 5 日 EMA",
        ],
        "direction": "因子值高：价量同向且波动配合的滞后动量；自适应平滑降低假信号。",
        "operators": ["delay", "log", "ATR", "ema", "volume"],
    },
    "factor_lag_volatility_regime_adapted": {
        "theme": "波动状态自适应的滞后量价",
        "summary": "核心仍是滞后收益 × log 量比 × ATR，但按波动状态切换 EMA 窗口。",
        "steps": [
            "原始信号 = 5日收益 × log(volume/EMA30) × ATR/close",
            "高波动市场：拉长 EMA；低波动市场：缩短 EMA",
        ],
        "direction": "因子值高：在不同波动状态下都较稳定的量价共振信号。",
        "operators": ["delay", "log", "ATR", "ema", "volume"],
    },
    "factor_lag_vol_volatility_adjusted": {
        "theme": "滞后收益 × 对数成交量 × ATR（DSL）",
        "summary": "Python 版的 DSL 表达：短周期收益经成交量和波动率加权后平滑。",
        "steps": [
            "5 日收益率 = close/delay(close,5) − 1",
            "乘以 log(volume/ema(volume,30))",
            "再乘以 ATR(20)/close",
            "整体 5 日 ema 平滑输出",
        ],
        "direction": "因子值高：滞后上涨且放量、波动配合；因子值低或为负则相反。",
        "operators": ["delay", "log", "ATR", "ema", "volume"],
    },
    "factor_vol_lag_ret_smoothed": {
        "theme": "量价滞后响应（平滑 DSL）",
        "summary": "5 日收益 × log 量比 × (1+ATR/close)，再 ema 平滑。",
        "steps": [
            "5 日收益 × log(volume/ema(volume,20))",
            "乘以 (1 + ATR/close)，高波动日信号放大",
            "5 日 ema 平滑",
        ],
        "direction": "因子值高：价量同向的滞后动量，且波动率有配合。",
        "operators": ["delay", "log", "ATR", "ema", "volume"],
    },
    "factor_adaptive_volume_smoothness": {
        "theme": "自适应成交量平滑度",
        "summary": "比较短期与长期成交量变异系数，识别成交是否突然变得不规律。",
        "steps": [
            "短期 CV / 长期 CV，衡量成交量「毛躁程度」",
            "高残差波动风格下加大该信号权重",
            "输出门控后的平滑度得分",
        ],
        "direction": "因子值高：成交节奏突然变得更不规律（在特定风格下被放大）。",
        "operators": ["ts_std", "ts_mean", "volume"],
    },
    "factor_drawdown_depth_duration_range_vol": {
        "theme": "回撤深度 × 非新高日占比",
        "summary": "结合相对 20 日高点的回撤深度，以及过去 20 日「非当日 20 日新高」的天数占比。",
        "steps": [
            "回撤深度 = 1 − (20日最高 − 现价)/20日最高",
            "非新高日占比 = 过去20日里 close < 当日20日最高的天数 / 20",
            "duration_penalty = 1 − 非新高日占比（非经典回撤持续天数）",
            "再乘振幅比、量比后截面 rank",
        ],
        "direction": "因子值高：浅回撤、近期较少长时间待在非新高区；因子值低：深回撤或长期磨在非高位。",
        "operators": ["ts_max", "ts_sum", "rank", "delay"],
    },
    "factor_asym_vol_cont_gate_30": {
        "theme": "下跌/上涨波动不对称 × 量比",
        "summary": "分别统计下跌日、上涨日的波动率，取 log 比，再用成交量连续加权。",
        "steps": [
            "下跌日收益率（涨跌幅<0）→ 60 日 ts_std 得下跌波动",
            "上涨日收益率（涨跌幅>0）→ 60 日 ts_std 得上涨波动",
            "log(下跌波动/上涨波动) × (volume/20日均量)",
            "zscore 标准化",
        ],
        "direction": "因子值高：下跌时波动相对更大（跌更乱）；因子值低：上涨波动主导。",
        "operators": ["where", "ts_std", "log", "safe_div", "zscore", "volume"],
    },
    "factor_asym_ratio_pure": {
        "theme": "涨跌日振幅不对称",
        "summary": "比较上涨日、下跌日的平均振幅（high−low），取 log 比后 rank。",
        "steps": [
            "下跌日：累计 (high−low)，除以下跌天数 → 下跌日平均振幅",
            "上涨日：同样计算 → 上涨日平均振幅",
            "log(下跌平均/上涨平均)，截面 rank",
        ],
        "direction": "因子值高：下跌日振幅相对更大；因子值低：上涨日振幅主导。",
        "operators": ["where", "ts_sum", "log", "rank", "high-low"],
    },
    "factor_asym_vol_volume_cont_30": {
        "theme": "波动不对称 × 连续量比（Python）",
        "summary": "与 asym_vol_cont_gate 类似，但窗口 30 日、量比截断在 [0,3]。",
        "steps": [
            "下跌日、上涨日分别算 30 日滚动波动率",
            "log(下跌波动/上涨波动)",
            "乘以 min(volume/20日均量, 3)",
        ],
        "direction": "因子值高：跌时波动更大且有一定放量配合。",
        "operators": ["ts_std", "log", "volume"],
    },
    "factor_drawdown_vol_persistence": {
        "theme": "回撤幅度 ÷ 波动率",
        "summary": "相对 60 日高点的回撤，经 20 日波动率标准化，再与 ATR 项结合。",
        "steps": [
            "回撤 = close/ts_max(close,60) − 1",
            "除以 ts_std(日收益率, 20) 做风险调整",
            "与 ATR 相关项相乘",
        ],
        "direction": "因子值高：深回撤且波动放大；因子值低：浅回撤或低波动。",
        "operators": ["ts_max", "ts_std", "ts_pct", "ATR"],
    },
    "factor_vol_gated_momentum_v2": {
        "theme": "成交量门控动量",
        "summary": "20 日价格动量，仅在成交量放大时保留，并用振幅过滤噪声。",
        "steps": [
            "计算 20 日价格动量（收益率）",
            "成交量高于 20 日均量一定倍数时保留，否则削弱",
            "用振幅/波动进一步过滤",
        ],
        "direction": "因子值高：有量配合的上涨动量；因子值低：无量反弹或下跌动量。",
        "operators": ["ts_pct", "ts_mean", "volume", "high-low"],
    },
    "factor_range_asym_log_vol": {
        "theme": "涨跌日振幅不对称 × 成交量（长窗口）",
        "summary": "log(上涨日平均振幅/下跌日平均振幅) 再乘以 log 量比。",
        "steps": [
            "上涨日、下跌日分别求平均振幅 (high−low)",
            "log(上涨平均/下跌平均)",
            "乘以 log(volume/20日均量)",
        ],
        "direction": "因子值高：上涨日振幅更大且有一定放量。",
        "operators": ["log", "high-low", "volume"],
    },
    "factor_resvol_volume_momentum": {
        "theme": "价格动量 × 相对成交量",
        "summary": "20 日涨跌幅乘以成交量相对 20 日 EMA 的比值。",
        "steps": [
            "ROC(close,20) = close/20日前close − 1",
            "乘以 volume/ema(volume,20)",
        ],
        "direction": "因子值高：涨且放量，或跌且放量（方向由 ROC 符号决定）。",
        "operators": ["ROC", "ema", "safe_div", "volume"],
    },
    "factor_volume_price_divergence_cross_mut": {
        "theme": "量价背离（5 日）",
        "summary": "价涨量缩或价跌量增时信号强，tanh 压缩极端值。",
        "steps": [
            "5 日价格变化 = close − 5日前close",
            "乘以 volume/ema(volume,20)",
            "取负号后 tanh 压缩到 (−1,1)",
        ],
        "direction": "因子值高：价量背离（涨缩量的反向逻辑经负号处理）；适合捕捉反转类信号。",
        "operators": ["delay", "tanh", "ema", "volume"],
    },
    "factor_range_asym_log_short": {
        "theme": "涨跌日振幅不对称（短窗口 log 比）",
        "summary": "10 日窗口内，分别统计收阳/收阴日的平均振幅，取 log 比。",
        "steps": [
            "收阳：close > 昨收；收阴：反之",
            "10 日内上涨日振幅之和/上涨天数 vs 下跌日同理",
            "log(上涨平均振幅/下跌平均振幅)，最少 5 日有效",
        ],
        "direction": "因子值高：上涨日振幅相对更大；短窗口，对近期结构更敏感。",
        "operators": ["delay", "log", "high-low"],
    },
}

CHART_GUIDES: dict[str, str] = {
    "rankic": (
        "横轴为日期，纵轴为当日 RankIC（Spearman 秩相关）。"
        "每个点 = 当日全市场因子排序与次日收盘收益的相关系数。"
        "线在 0 上方居多说明因子多数日子有效；大幅下穿 0 的区间需结合市场环境看。"
        "上方卡片：Mean RankIC 为全程平均；RankICIR = 均值/标准差，越大越稳定；"
        "胜率为 RankIC>0 的交易日占比。"
    ),
    "decile": (
        "横轴 G1~G10 为按因子值从低到高的十个分组（G1 因子最低，G10 最高）。"
        "纵轴为各组累计收益（复利净值 − 1）。"
        "理想情况：G10 明显高于 G1，且中间组大致单调递增，说明因子排序有分层能力。"
        "若 G1 与 G10 差距小或交叉，说明分层弱。"
    ),
    "ls": (
        "多空累计净值曲线：每日做多 G10、做空 G1 的净收益（已扣假设换手成本）复利累积。"
        "曲线向上 = 高因子组长期跑赢低因子组；向下 = 反向。"
        "注意：这是研究用十分位多空，不是下方 TopK 可交易组合。"
    ),
    "topk": (
        "TopK 组合净值曲线（起点归一化为 1）。"
        "规则：每日选因子值最高 10% 股票（最多 50 只），次日开盘价等权买入。"
        "曲线向上且回撤可控 = 头部选股策略有效；"
        "若 RankIC 高但 TopK 弱，可能是中间分组贡献大或头部极端值不稳定。"
    ),
}


def _esc(text: str) -> str:
    return html_lib.escape(text or "", quote=True)


def _detect_operators_from_dsl(dsl: str) -> list[str]:
    found: list[str] = []
    text = dsl or ""
    for key in OPERATOR_ZH:
        if key in ("volume", "high-low"):
            continue
        if re.search(rf"\b{re.escape(key)}\s*\(", text):
            found.append(key)
    if "high" in text and "low" in text:
        found.append("high-low")
    if "volume" in text:
        found.append("volume")
    return list(dict.fromkeys(found))


def get_factor_guide(
    factor_name: str,
    *,
    ann: dict[str, Any] | None = None,
    dsl: str = "",
    python_code: str = "",
) -> dict[str, Any]:
    """Return structured Chinese guide; curated week/dug factors first, else fallback."""
    if factor_name in WEEK3_FACTOR_GUIDES:
        g = dict(WEEK3_FACTOR_GUIDES[factor_name])
        g.setdefault("formula_display", (ann or {}).get("formula_display") or dsl.strip())
        return g
    if factor_name in WEEKLY_DUG_FACTOR_GUIDES:
        g = dict(WEEKLY_DUG_FACTOR_GUIDES[factor_name])
        g.setdefault("formula_display", (ann or {}).get("formula_display") or dsl.strip())
        return g

    ann = ann or {}
    primary = (ann.get("formula_display") or dsl or "").strip()
    py = (python_code or "").strip()
    theme = ann.get("theme") or "量价因子"
    ops = _detect_operators_from_dsl(primary or py)
    steps = []
    if primary and len(primary) < 200:
        steps.append(f"按公式计算：{primary}")
    elif primary:
        steps.append(f"按公式计算（见下方代码）：{primary[:120]}…")
    elif py:
        steps.append("见下方 Python 源码逐步计算")
    else:
        steps.append("源侧仅交付中性化面板数值，无可展示公式/代码；评估直接使用面板落值")
    steps.append("每个交易日 T 仅使用 T 日及以前的 OHLCV，不使用未来数据")

    if primary or py:
        summary = f"属于{theme}，具体逻辑见公式与代码。"
    else:
        summary = f"属于{theme}；源侧未公开公式，仅有中性化面板可供评估。"

    return {
        "theme": theme,
        "summary": summary,
        "steps": steps,
        "direction": "因子值越高/越低的经济含义需结合公式方向与 RankIC 符号判断（本页 RankIC 为正向检验）。",
        "operators": ops[:8],
        "formula_display": primary,
    }


def render_operator_list_html(operators: list[str]) -> str:
    if not operators:
        return ""
    rows = []
    for op in operators:
        desc = OPERATOR_ZH.get(op, "见通用算子表")
        rows.append(f"<li><code>{_esc(op)}</code>：{_esc(desc)}</li>")
    return f"<ul class='op-list'>{''.join(rows)}</ul>"


def _lookahead_badge_html(risk: str) -> str:
    colors = {
        "low": ("#dcfce7", "#166534", "低"),
        "medium": ("#fef9c3", "#854d0e", "中"),
        "high": ("#fee2e2", "#991b1b", "高"),
        "pending_fix": ("#fce7f3", "#9d174d", "待修复"),
    }
    bg, fg, label = colors.get(risk, ("#f3f4f6", "#374151", risk or "?"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;'
        f'font-size:12px;font-weight:600">未来函数风险·{label}</span>'
    )


def render_interpretation_block(
    factor_name: str,
    *,
    ann: dict[str, Any] | None = None,
    dsl: str = "",
    python_code: str = "",
    lookahead_risk: str = "",
    lookahead_judgment: str = "",
) -> str:
    guide = get_factor_guide(factor_name, ann=ann, dsl=dsl, python_code=python_code)
    steps_html = "".join(f"<li>{_esc(s)}</li>" for s in guide.get("steps", []))
    ops_html = render_operator_list_html(guide.get("operators") or [])
    risk_badge = _lookahead_badge_html(lookahead_risk) if lookahead_risk else ""

    return f"""
  <div class="card">
    <h2>因子说明</h2>
    <p class="theme-line"><b>{_esc(guide.get('theme', ''))}</b></p>
    <p>{_esc(guide.get('summary', ''))}</p>
    <h3>计算步骤</h3>
    <ol class="steps">{steps_html}</ol>
    <h3>因子值怎么理解</h3>
    <p>{_esc(guide.get('direction', ''))}</p>
    {f'<h3>公式里用到的算子</h3>{ops_html}' if ops_html else ''}
    {f'<p class="note">{risk_badge} <b>未来数据审查：</b>{_esc(lookahead_judgment)}</p>' if lookahead_judgment else ''}
  </div>"""


def render_chart_guide(key: str) -> str:
    text = CHART_GUIDES.get(key, "")
    return f'<p class="chart-guide"><b>怎么看图：</b>{_esc(text)}</p>' if text else ""


def signal_note_zh(note: str) -> str:
    """Translate common English signal lag notes embedded in analysis payload."""
    if not note:
        return "信号对齐：T 日收盘后可得因子 → 与 T+1 日前瞻收益做 RankIC。"
    if sum(1 for c in note if "\u4e00" <= c <= "\u9fff") >= 8:
        return note
    return (
        "T 日收盘后因子可得 → 与 T+1 日前瞻收益做 RankIC；"
        "多空净收益假设单边换手 40% 及双边手续费；TopK 回测另用开盘价成交。"
    )
