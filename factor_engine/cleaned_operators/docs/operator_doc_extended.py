# -*- coding: utf-8 -*-
"""算子释义扩展表（2026-09-21）。

背景
----
``operator_doc_semantics.py`` 的显式表（``_EXPLICIT``）覆盖的是通用代数 / 时序 /
截面算子；``intraday_*``（分钟→日频聚合）与 ``intra_*``（同一批算子的挖掘侧别名，
注册规则见 ``factor_engine/api/intraday_daily.py``：``name.replace("intraday_", "intra_", 1)``）
一族在注册表里没有 catalog 条目，于是释义落到兜底模板，报告页只能显示
"算子详细释义尚待注册表/实现核验"。

本模块把这一族以及其余零星缺口（``col`` / ``field`` / ``nonfinite_to_num`` /
``ts_regression_slope``）补成显式 ``OpDoc``。

撰写依据（不推测）
------------------
* ``factor_engine/cleaned_operators/microstructure/intraday_agg.py`` —— 实现本体
  （如 ``IntraRealizedVariance`` = 日内对数收益平方和；``IntraPathEfficiency`` =
  |净位移| / 路径长度）。
* ``factor_engine/tests/operators/test_operator_expansion_gap.py`` —— 行为断言
  （``test_intraday_realized_variance_matches_manual``、
  ``test_intraday_path_efficiency_single_direction_is_one`` 等），是"可核验"的口径。
* ``factor_engine/api/intraday_daily.py`` —— 每个特征名与默认参数
  （``bar_minutes=5``、``min_coverage=0.80``、``min_bars>=2``、
  ``timestamp_convention="bar_end"``、``cutoff_time="session_close"``）。
* ``factor_engine/cleaned_operators/polars_native/intraday_final.py`` —— 进阶变体
  的 ``OperatorMetadata.description``。

统一口径（所有 ``intraday_*`` 共享）
------------------------------------
1. 输入是**分钟面板**（``StockMinuteBar`` / ``StockMinuteBarAdj``），先按
   (标的, 交易日) 会话切分，再聚合成**当日一个值**。
2. 会话按 A 股官方 240 根 5 分钟格（上午 120 + 下午 120，午休无 bar）计算；
   ``min_coverage=0.80``（进阶聚合器要求 ≥0.90）——**有效 slot 覆盖不足或出现
   非有限样本时，该 (标的, 日) 输出 NaN**，不插值、不向前填充。
3. 段端点默认 ``EndpointPolicy.EXACT``（11:30 / 13:01 / 15:00 必须存在），
   否则段收益为 NaN；显式传 ``endpoint_policy="recent_valid"`` 才退化为
   首个/末个有效 bar（R26-025/027）。
4. 时区：COS 分钟数据是 UTC，段边界按 ``Asia/Shanghai`` 墙钟判定。

本表只写"含义 / 算法 / 公式"；参数默认值与覆盖率门槛按上面第 2、3 条，
具体边界条件以实现为准。
"""
from __future__ import annotations

from factor_engine.cleaned_operators.docs.operator_doc_semantics import OpDoc

#: 挖掘侧别名前缀映射：``intra_X`` 与 ``intraday_X`` 是同一算子的两种写法。
_ALIAS_PREFIXES: tuple[tuple[str, str], ...] = (
    ("intraday_", "intra_"),
    ("intra_", "intraday_"),
)


def normalize_extended_name(name: str) -> str:
    """把 ``intra_X`` / ``intraday_X`` 归一到本表登记的键。

    两代命名互为别名（``intraday_daily.py`` 机械替换），因此释义可共用；
    未命中返回原样，绝不改写其他算子名。
    """
    text = str(name or "").strip()
    if text in EXTENDED_DOCS:
        return text
    for src, dst in _ALIAS_PREFIXES:
        if text.startswith(src):
            candidate = dst + text[len(src):]
            if candidate in EXTENDED_DOCS:
                return candidate
    return text


# ---------------------------------------------------------------------------
# 日内已实现矩族（realized moments）
# ---------------------------------------------------------------------------
_REALIZED: dict[str, OpDoc] = {
    "intraday_realized_variance": OpDoc(
        "日内已实现方差 RV（当日分钟对数收益的平方和）。",
        "取当日会话内每根 bar 的对数收益 r_i = ln(C_i / C_{i-1})（相邻有效 bar），"
        "对其平方求和；覆盖不足或含非有限样本当天为 NaN。",
        r"RV_t = \sum_{i=1}^{N_t} r_{i}^{2},\quad r_i=\ln\frac{C_i}{C_{i-1}}",
    ),
    "intraday_realized_vol": OpDoc(
        "日内已实现波动率（RV 的平方根）。",
        "先算当日 RV（对数收益平方和），再开平方；量纲回到收益尺度。",
        r"\sigma_t = \sqrt{RV_t} = \sqrt{\sum_{i} r_i^{2}}",
    ),
    "intraday_realized_skew": OpDoc(
        "日内已实现偏度（收益分布的不对称方向）。",
        "以日内对数收益的三阶幂变差对 RV 做标准化，捕捉当日是「急涨缓跌」还是"
        "「急跌缓涨」（正=右偏/急涨）。",
        r"RS_t = \frac{\sqrt{N_t}\sum_i r_i^{3}}{RV_t^{3/2}}",
    ),
    "intraday_realized_kurtosis": OpDoc(
        "日内已实现峰度（极端 bar 的相对厚度）。",
        "以日内对数收益的四阶幂变差对 RV 平方做标准化；值越大表示当日收益"
        "越集中在少数极端 bar（尖峰厚尾）。",
        r"RK_t = \frac{N_t\sum_i r_i^{4}}{RV_t^{2}}",
    ),
    "intraday_realized_quarticity": OpDoc(
        "日内已实现四次幂变差 RQ（RV 的估计方差来源）。",
        "日内对数收益四次幂的缩放求和，常用于构造 RV 的置信区间"
        "（RV 的渐近方差 ∝ 2·RQ）。",
        r"RQ_t = \frac{N_t}{3}\sum_{i} r_i^{4}",
    ),
    "intraday_upside_semivariance": OpDoc(
        "日内上行半方差（只累计正收益的平方）。",
        "对数收益 r_i > 0 的部分平方求和；刻画上行方向的波动贡献。",
        r"RS^{+}_t = \sum_{i} r_i^{2}\,\mathbf{1}\{r_i>0\}",
    ),
    "intraday_downside_semivariance": OpDoc(
        "日内下行半方差（只累计负收益的平方）。",
        "对数收益 r_i < 0 的部分平方求和；是「下行风险」的日内度量。",
        r"RS^{-}_t = \sum_{i} r_i^{2}\,\mathbf{1}\{r_i<0\}",
    ),
    "intraday_bipower_variation": OpDoc(
        "日内双幂变差 BV（对跳跃稳健的连续波动估计）。",
        "相邻两根 bar 绝对收益的乘积求和，再乘 π/2 校正；跳跃（孤立的极端 bar）"
        "不会像 RV 那样被平方放大，故 RV − BV 常用于识别跳跃。",
        r"BV_t = \frac{\pi}{2}\sum_{i=2}^{N_t}|r_{i-1}|\,|r_i|",
    ),
    "intraday_jump_variation": OpDoc(
        "日内跳跃变差（RV 中不能被连续波动解释的部分）。",
        "以 RV − BV 的非负部分定义；为 0 表示当日价格路径基本连续、无显著跳跃。",
        r"JV_t = \max\bigl(RV_t - BV_t,\ 0\bigr)",
    ),
    "intraday_jump_ratio": OpDoc(
        "跳跃占比（跳跃变差 / 已实现方差）。",
        "把当日 RV 拆成连续成分与跳跃成分后，跳跃成分所占比例；"
        "越大表示当日波动越集中（越「跳跃式」）。",
        r"JR_t = \frac{RV_t - BV_t}{RV_t}",
    ),
    "intraday_signed_jump": OpDoc(
        "有符号跳跃（跳跃的方向性合成）。",
        "在跳跃识别阈值（默认 threshold=0.0）之上，按超出部分收益的符号"
        "合成一个有符号量：正=向上跳，负=向下跳。阈值口径以实现为准。",
        r"SJ_t = \sum_{i} r_i\,\mathbf{1}\{|r_i|>\theta\}",
    ),
    "intraday_max_abs_return": OpDoc(
        "日内最大单 bar 绝对收益。",
        "对当日所有分钟对数收益取绝对值后取最大；度量最剧烈的一根 bar。",
        r"\max_i |r_i|",
    ),
    "intraday_tail_return_sum": OpDoc(
        "尾部分位之上的收益之和。",
        "取 |r_i| 的 q 分位（默认 q=0.95）作为阈值，把超过阈值的 bar 收益求和；"
        "刻画「少数极端 bar 贡献了多少收益」。",
        r"\sum_{i} r_i\,\mathbf{1}\{|r_i|\ge Q_q(|r|)\}",
    ),
    "intraday_jump_count": OpDoc(
        "日内跳跃 bar 计数。",
        "统计 |r_i| 超过阈值（默认 threshold=0.01，即 1%）的 bar 个数；"
        "是跳跃的「频次」口径（RV/BV 只度量幅度）。",
        r"\#\{i:\ |r_i|>\theta\}",
    ),
}

# ---------------------------------------------------------------------------
# 时段 / 区间收益
# ---------------------------------------------------------------------------
_SEGMENT: dict[str, OpDoc] = {
    "intraday_return": OpDoc(
        "日内收益（收盘 / 开盘 − 1）。",
        "挖掘侧别名：注册表把 ``intraday_return`` 解析为 ``open_close_return``，"
        "即当日 open 收盘价与 open 开盘价之比减 1，与分钟数据无关。",
        r"R_t = \frac{Close_t}{Open_t}-1",
    ),
    "intraday_open_to_close_return": OpDoc(
        "日内收益（开盘到收盘）。",
        "等同 ``intraday_return`` / ``open_close_return``：当日收盘 / 当日开盘 − 1。",
        r"R_t = \frac{Close_t}{Open_t}-1",
    ),
    "intraday_first_nmin_return": OpDoc(
        "开盘后前 N 分钟收益（默认 minutes=30）。",
        "取开盘首根 bar 的价格与开盘后第 N 分钟（默认 30）那根 bar 的价格做比值减 1；"
        "端点必须存在，否则 NaN。",
        r"R^{(N)}_t = \frac{C_{09{:}31+N}}{C_{09{:}31}}-1",
    ),
    "intraday_last_nmin_return": OpDoc(
        "收盘前 N 分钟收益（默认 minutes=30）。",
        "取收盘前第 N 分钟那根 bar 的价格与当日收盘价做比值减 1；刻画尾盘拉升/砸盘。",
        r"R_{tail} = \frac{C_{15{:}00}}{C_{15{:}00-N}}-1",
    ),
    "intraday_morning_return": OpDoc(
        "上午段收益（09:31 → 11:30）。",
        "上午段末端点收盘 / 段初端点收盘 − 1；段端点默认要求精确存在。",
        r"R^{AM}_t = \frac{C_{11{:}30}}{C_{09{:}31}}-1",
    ),
    "intraday_afternoon_return": OpDoc(
        "下午段收益（13:01 → 15:00）。",
        "下午段末端点收盘 / 段初端点收盘 − 1。",
        r"R^{PM}_t = \frac{C_{15{:}00}}{C_{13{:}01}}-1",
    ),
    "intraday_morning_afternoon_reversal": OpDoc(
        "上下午收益反转度（日内方向切换幅度）。",
        "把当日拆成上午段与下午段，用两段收益的方向差刻画「上午涨、下午跌」这类"
        "日内反转的强度；同向时接近 0。具体合成以实现为准。",
        r"REV_t = R^{AM}_t - R^{PM}_t",
    ),
    "intraday_segment_return": OpDoc(
        "指定时段段收益（segment=morning / afternoon）。",
        "按 segment 参数取上午或下午段，计算段末端点收盘 / 段初端点收盘 − 1；"
        "端点缺省策略默认 EXACT（缺 11:30/13:01/15:00 则 NaN）。",
        r"R^{seg}_t = \frac{C_{end(seg)}}{C_{start(seg)}}-1",
    ),
    "intraday_segment_volume_share": OpDoc(
        "指定时段成交量占全天比例。",
        "该时段成交量 / 当日总成交量；用于度量「成交集中在哪个时段」。",
        r"w^{seg}_t = \frac{\sum_{i\in seg} V_i}{\sum_i V_i}",
    ),
    "intraday_segment_amount_share": OpDoc(
        "指定时段成交额占全天比例。",
        "该时段成交额（Σ价×量）/ 当日总成交额。",
        r"w^{seg,amt}_t = \frac{\sum_{i\in seg} A_i}{\sum_i A_i}",
    ),
    "intraday_segment_vwap_deviation": OpDoc(
        "段末收盘相对该段累计 VWAP 的偏离。",
        "先算该时段的成交量加权均价，再取段末收盘与之比较的相对偏离。",
        r"\frac{C_{end(seg)}}{\mathrm{VWAP}^{seg}}-1,\quad "
        r"\mathrm{VWAP}^{seg}=\frac{\sum_{i\in seg} P_i V_i}{\sum_{i\in seg} V_i}",
    ),
    "intraday_segment_realized_vol": OpDoc(
        "指定时段内的已实现波动率。",
        "只在该时段的 bar 上计算对数收益平方和，再开平方（即段内 RV 的平方根）。",
        r"\sigma^{seg}_t = \sqrt{\sum_{i\in seg} r_i^{2}}",
    ),
    "intraday_lunch_gap_return": OpDoc(
        "午间跳空收益（下午首 bar 开盘 / 上午末 bar 收盘 − 1）。",
        "跨午休的价格断层：下午第一根 bar 的开盘价对上午最后一根 bar 的收盘价；"
        "已由单测固定该口径（``test_intraday_segment_handles_utc_index``）。",
        r"R^{lunch}_t = \frac{Open^{PM}_{first}}{Close^{AM}_{last}}-1",
    ),
    "intraday_closing_return": OpDoc(
        "尾盘收益（收盘前 N 分钟，默认 30）。",
        "收盘前 N 分钟段的段收益；是「收盘前异动」的直接度量。",
        r"R^{close,N}_t = \frac{C_{15{:}00}}{C_{15{:}00-N}}-1",
    ),
    "intraday_closing_ramp": OpDoc(
        "尾盘拉升强度（收盘前活跃度/涨幅的加速）。",
        "比较收盘前 N 分钟与更早时段的收益或成交活跃度，刻画尾盘被「抬价」的程度；"
        "与 closing_return 的区别在于它衡量「加速」，实现口径以实现为准。",
        r"\mathrm{ramp}_t = f\bigl(R^{close,N}_t,\ R^{prior}_t\bigr)",
    ),
}

# ---------------------------------------------------------------------------
# 日内路径 / 形态
# ---------------------------------------------------------------------------
_PATH: dict[str, OpDoc] = {
    "intraday_trend_slope": OpDoc(
        "日内趋势斜率（价格对时间的最小二乘斜率）。",
        "把当日分钟收盘价对 bar 序号（时间索引）做 OLS 回归，取斜率；"
        "正=日内整体上行。",
        r"C_i \approx a + b\,i,\quad b=\frac{\sum(i-\bar{i})(C_i-\bar{C})}"
        r"{\sum(i-\bar{i})^2}",
    ),
    "intraday_trend_r2": OpDoc(
        "日内趋势拟合优度 R²。",
        "上述趋势回归的决定系数：越接近 1 表示日内是「单边直线行情」，"
        "越低表示震荡。",
        r"R^2 = 1-\frac{\sum(C_i-\hat{C}_i)^2}{\sum(C_i-\bar{C})^2}",
    ),
    "intraday_path_length": OpDoc(
        "日内价格路径长度（累计绝对变动）。",
        "对相邻分钟收盘价的绝对变动求和，度量「走了多少路」（不含方向）。",
        r"L_t = \sum_{i=2}^{N_t}\bigl|C_i - C_{i-1}\bigr|",
    ),
    "intraday_path_efficiency": OpDoc(
        "日内路径效率（净位移 / 路径长度）。",
        "净位移取绝对值后除以路径长度；单边行情 = 1（已由单测固定），"
        "来回震荡 → 0。是「趋势 vs 噪音」的日内度量。",
        r"E_t = \frac{\bigl|C_{N}-C_{1}\bigr|}{\sum_{i=2}^{N}|C_i-C_{i-1}|}\in[0,1]",
    ),
    "intraday_reversal_count": OpDoc(
        "日内收益方向反转次数。",
        "对分钟收益的符号序列统计方向切换（+→− 或 −→+）的次数；"
        "次数越高表示日内来回震荡越频繁。",
        r"\#\{i:\ \mathrm{sgn}(r_i)\ne \mathrm{sgn}(r_{i-1}),\ r_i r_{i-1}\ne 0\}",
    ),
    "intraday_return_autocorr": OpDoc(
        "日内分钟收益的自相关（默认 lag=1）。",
        "在当日 bar 序列内计算 r_i 与 r_{i-lag} 的相关系数；"
        "负值表示 bar 间反转、正值表示短期延续。",
        r"\rho_{\ell} = \mathrm{corr}\bigl(r_i,\ r_{i-\ell}\bigr),\ \ell=\text{lag}",
    ),
    "intraday_max_drawdown": OpDoc(
        "日内最大回撤。",
        "在当日分钟价格路径上，取「此前最高价 − 当前价」的最大相对幅度。",
        r"MDD_t = \max_i \frac{\max_{j\le i} C_j - C_i}{\max_{j\le i} C_j}",
    ),
    "intraday_max_runup": OpDoc(
        "日内最大上行幅度。",
        "在当日分钟价格路径上，取「当前价 − 此前最低价」的最大相对幅度。",
        r"MRU_t = \max_i \frac{C_i - \min_{j\le i} C_j}{\min_{j\le i} C_j}",
    ),
    "intraday_time_of_high": OpDoc(
        "日内最高价出现时刻。",
        "取当日最高价所在 bar 的官方 slot 序号（按 240 格归一）；"
        "接近 0 = 开盘冲高，接近 1 = 尾盘冲高。",
        r"\tau^{high}_t = \frac{\arg\max_i C_i}{N_t}\in[0,1]",
    ),
    "intraday_time_of_low": OpDoc(
        "日内最低价出现时刻。",
        "取当日最低价所在 bar 的官方 slot 序号（按 240 格归一）。",
        r"\tau^{low}_t = \frac{\arg\min_i C_i}{N_t}\in[0,1]",
    ),
    "intraday_close_location": OpDoc(
        "收盘位置（收盘价在日内高低区间中的相对位置）。",
        "收盘价减最低价，除以当日最高价减最低价；1 = 收在最高点（强势），"
        "0 = 收在最低点。",
        r"CL_t = \frac{C_{close}-L_t}{H_t-L_t}\in[0,1]",
    ),
    "intraday_opening_range": OpDoc(
        "开盘区间宽度（开盘前 N 分钟，默认 30）。",
        "取开盘后 N 分钟内的最高价与最低价之差；常与 opening_range_position 搭配"
        "做区间突破类因子。",
        r"OR_t = H^{(N)}_{open} - L^{(N)}_{open}",
    ),
    "intraday_opening_range_position": OpDoc(
        "价格相对开盘区间的突破位置。",
        "把当前/收盘价映射到开盘区间 [L, H] 的相对位置：>1 表示向上突破区间，"
        "<0 表示向下突破。",
        r"ORP_t = \frac{C - L^{(N)}_{open}}{H^{(N)}_{open}-L^{(N)}_{open}}",
    ),
    "intraday_opening_drive": OpDoc(
        "开盘驱动强度（开盘后 N 分钟的方向性推进）。",
        "以开盘后 N 分钟（默认 30）的价格位移（相对开盘价）度量开盘买/卖压；"
        "配合成交量口径时衡量「放量推进」与否，具体合成以实现为准。",
        r"OD_t = \frac{C_{09{:}31+N}}{C_{09{:}31}}-1",
    ),
    "intraday_gap_continuation": OpDoc(
        "跳空的日内延续度。",
        "度量开盘跳空后价格是否继续沿跳空方向走（延续）而非回补；"
        "与 gap_fill_ratio 互为反面。",
        r"GC_t = \mathrm{sgn}(\mathrm{gap})\cdot\frac{C_{close}-C_{open}}{|gap|}",
    ),
    "intraday_gap_fill_ratio": OpDoc(
        "跳空的回补比例。",
        "跳空幅度中被日内价格回补掉的比例：1 = 完全回补（跳空被吃回），"
        "0 = 完全未回补。分母为 0（无跳空）时该日不作数。",
        r"GFR_t = \frac{|\mathrm{gap}|-\bigl|C_{close}-C_{open}\bigr|}{|\mathrm{gap}|}"
        r"\ \text{（口径以实现为准）}",
    ),
    "intraday_extreme_bar_return": OpDoc(
        "最极端 bar 的收益（side=max/min）。",
        "按 side 参数取当日收益最大或最小的那根 bar 的收益；"
        "区别于 max_abs_return（比的是绝对值）。",
        r"\max_i r_i\ \ \text{或}\ \ \min_i r_i",
    ),
}

# ---------------------------------------------------------------------------
# 涨跌停 / 交易状态
# ---------------------------------------------------------------------------
_LIMIT: dict[str, OpDoc] = {
    "intraday_limit_up_touch_fraction": OpDoc(
        "盘中触及涨停的 slot 占比。",
        "统计当日价格触及涨停价的官方 slot 数 ÷ 有效 slot 数；"
        "度量「是否 / 多长时间封在涨停」。",
        r"f^{up}_t = \frac{\#\{i: P_i \ge \mathrm{HighLimit}\}}{N_t}",
    ),
    "intraday_limit_down_touch_fraction": OpDoc(
        "盘中触及跌停的 slot 占比。",
        "统计当日价格触及跌停价的官方 slot 数 ÷ 有效 slot 数。",
        r"f^{down}_t = \frac{\#\{i: P_i \le \mathrm{LowLimit}\}}{N_t}",
    ),
    "intraday_limit_up_close": OpDoc(
        "收盘是否封在涨停（1/0）。",
        "当日收盘价等于涨停价则为 1，否则 0；是最常用的「涨停封板」标签。",
        r"\mathbf{1}\{C_{close}=\mathrm{HighLimit}\}",
    ),
    "intraday_limit_down_close": OpDoc(
        "收盘是否封在跌停（1/0）。",
        "当日收盘价等于跌停价则为 1，否则 0。",
        r"\mathbf{1}\{C_{close}=\mathrm{LowLimit}\}",
    ),
    "intraday_limit_first_hit_time": OpDoc(
        "首次触及涨/跌停的时刻。",
        "当日第一次触及涨停（或跌停）价所在 bar 的官方 slot 序号，"
        "按 240 格归一：越早触板通常意味着买盘越强。",
        r"\frac{\min\{i: P_i \ge \mathrm{HighLimit}\}}{N_t}",
    ),
    "intraday_limit_duration": OpDoc(
        "封板持续时长（处于涨/跌停状态的 slot 数）。",
        "统计当日价格处于涨停（或跌停）状态的官方 slot 数，再按 240 格归一；"
        "与 touch_fraction 的区别在于是否要求连续计状态，口径以实现为准。",
        r"d_t = \#\{i: P_i \ge \mathrm{HighLimit}\}",
    ),
    "intraday_limit_reopen_count": OpDoc(
        "涨跌停打开次数。",
        "统计当日「先封住、后打开」（离开涨停/跌停状态）的次数；"
        "反映封板牢固程度（打开次数越多越不牢）。",
        r"\#\{i:\ \text{封板态}_{i-1}\wedge \text{非封板态}_{i}\}",
    ),
}

# ---------------------------------------------------------------------------
# 日内 VWAP 族
# ---------------------------------------------------------------------------
_VWAP: dict[str, OpDoc] = {
    "intraday_vwap": OpDoc(
        "日内成交量加权均价 VWAP。",
        "当日 Σ(价格 × 成交量) / Σ成交量；是机构成本线的日内口径。",
        r"\mathrm{VWAP}_t = \frac{\sum_i P_i V_i}{\sum_i V_i}",
    ),
    "intraday_close_to_vwap": OpDoc(
        "收盘价相对 VWAP 的偏离率。",
        "收盘价 / 日内 VWAP − 1；>0 表示当日收在成本线之上（买盘占优）。",
        r"\frac{C_{close}}{\mathrm{VWAP}_t}-1",
    ),
    "intraday_high_to_vwap": OpDoc(
        "日内最高价相对 VWAP 的上偏幅度。",
        "最高价 / 日内 VWAP − 1；刻画向上的日内乖离。",
        r"\frac{H_t}{\mathrm{VWAP}_t}-1",
    ),
    "intraday_low_to_vwap": OpDoc(
        "日内最低价相对 VWAP 的下偏幅度。",
        "最低价 / 日内 VWAP − 1；通常为负值，刻画向下的日内乖离。",
        r"\frac{L_t}{\mathrm{VWAP}_t}-1",
    ),
    "intraday_vwap_slope": OpDoc(
        "日内 VWAP 斜率。",
        "把逐 bar 的累计 VWAP 对 bar 序号做 OLS 回归取斜率；"
        "正 = 成本线被逐步抬高（典型的上行推进）。",
        r"\mathrm{VWAP}_i \approx a+b\,i,\quad b=\text{slope}",
    ),
    "intraday_vwap_deviation_mean": OpDoc(
        "价格对 VWAP 的偏离均值。",
        "逐 bar 计算 (P_i / VWAP_i − 1)，再取当日均值；衡量价格「贴着成本线走」"
        "还是长期偏离。",
        r"\overline{dev}_t = \frac{1}{N_t}\sum_i\Bigl(\frac{P_i}{\mathrm{VWAP}_i}-1\Bigr)",
    ),
    "intraday_vwap_deviation_std": OpDoc(
        "价格对 VWAP 偏离的标准差。",
        "对逐 bar 的 (P_i / VWAP_i − 1) 序列取标准差；度量围绕成本线的震荡幅度。",
        r"\mathrm{std}_i\Bigl(\frac{P_i}{\mathrm{VWAP}_i}-1\Bigr)",
    ),
    "intraday_vwap_cross_count": OpDoc(
        "价格穿越 VWAP 的次数。",
        "统计价格由下向上、或由上向下穿越 VWAP 的 bar 数；"
        "越小越「单边」，越大越「多空纠结」。",
        r"\#\{i:\ \mathrm{sgn}(P_{i}-\mathrm{VWAP}_{i})\ne"
        r"\mathrm{sgn}(P_{i-1}-\mathrm{VWAP}_{i-1})\}",
    ),
    "intraday_vwap_above_ratio": OpDoc(
        "价格位于 VWAP 之上的 slot 占比。",
        "统计 P_i ≥ VWAP_i 的官方 slot 数 ÷ 有效 slot 数；"
        "接近 1 表示当日几乎全程在成本线之上，是较强的买方信号。",
        r"r^{above}_t = \frac{\#\{i: P_i \ge \mathrm{VWAP}_i\}}{N_t}",
    ),
}

# ---------------------------------------------------------------------------
# 日内量能结构
# ---------------------------------------------------------------------------
_VOLUME: dict[str, OpDoc] = {
    "intraday_volume_first_share": OpDoc(
        "开盘前 30 分钟成交量占全天比例。",
        "开盘后前 N 分钟（默认 30）成交量 / 当日总成交量；"
        "度量「成交是否集中在开盘」。",
        r"w^{first}_t = \frac{\sum_{i\in[09{:}31,09{:}31+N)} V_i}{\sum_i V_i}",
    ),
    "intraday_volume_last_share": OpDoc(
        "收盘前 30 分钟成交量占全天比例。",
        "收盘前 N 分钟（默认 30）成交量 / 当日总成交量。",
        r"w^{last}_t = \frac{\sum_{i\in[15{:}00-N,15{:}00]} V_i}{\sum_i V_i}",
    ),
    "intraday_volume_peak_time": OpDoc(
        "成交量峰值出现时刻。",
        "当日成交量最大的 bar 的官方 slot 序号（按 240 格归一）；"
        "刻画「放量时刻」在开盘、盘中还是尾盘。",
        r"\tau^{vol}_t = \frac{\arg\max_i V_i}{N_t}\in[0,1]",
    ),
    "intraday_volume_hhi": OpDoc(
        "成交量分布赫芬达尔指数（集中度）。",
        "把各 bar 成交量按当日常量占比 s_i = V_i / ΣV 后求平方和；"
        "越接近 1 表示成交越集中在少数 bar。",
        r"\mathrm{HHI}_t = \sum_i s_i^{2},\quad s_i=\frac{V_i}{\sum_j V_j}",
    ),
    "intraday_volume_entropy": OpDoc(
        "成交量分布香农熵。",
        "对成交量占比 s_i 计算 −Σ s_i ln s_i；越高表示成交在时间上越均匀，"
        "越低表示越集中（与 HHI 反向）。",
        r"H_t = -\sum_i s_i\ln s_i",
    ),
    "intraday_volume_profile_skew": OpDoc(
        "成交量日内分布偏度。",
        "把成交量按 bar 序号看作一条日内曲线，计算其三阶标准化矩；"
        "正 = 放量偏向后半场（尾盘放量），负 = 偏向开盘。",
        r"\mathrm{skew}_i(V_i)",
    ),
    "intraday_volume_profile_slope": OpDoc(
        "成交量日内分布斜率。",
        "把逐 bar 成交量对 bar 序号做 OLS 回归取斜率；"
        "正 = 越到后面越放量。",
        r"V_i \approx a+b\,i,\quad b=\text{slope}",
    ),
    "intraday_turnover_hhi": OpDoc(
        "成交额（换手）分布集中度 HHI。",
        "以各 bar 成交额占比求平方和；口径同 volume_hhi，但用金额而非股数，"
        "对高价股更可比。",
        r"\sum_i \Bigl(\frac{A_i}{\sum_j A_j}\Bigr)^{2}",
    ),
    "intraday_turnover_entropy": OpDoc(
        "成交额（换手）分布熵。",
        "以各 bar 成交额占比计算香农熵；越高表示金额在时间上越均匀分布。",
        r"-\sum_i \tilde{s}_i\ln\tilde{s}_i,\quad \tilde{s}_i=\frac{A_i}{\sum_j A_j}",
    ),
    "intraday_turnover_per_volatility": OpDoc(
        "单位波动对应的换手（「换手/波动」比）。",
        "当日换手（或成交额比例）除以日内波动率；"
        "高值表示「用较大成交换来较小波动」（成交密集但价格不敏感）。",
        r"\frac{\mathrm{Turnover}_t}{\sigma_t}",
    ),
}

# ---------------------------------------------------------------------------
# 量价关系 / 流动性与冲击
# ---------------------------------------------------------------------------
_FLOW: dict[str, OpDoc] = {
    "intraday_return_volume_corr": OpDoc(
        "分钟收益与成交量的相关系数。",
        "在当日 bar 序列内计算 r_i 与 V_i 的相关系数；"
        "正值表示「涨放量、跌缩量」。",
        r"\rho_t = \mathrm{corr}_i\bigl(r_i,\ V_i\bigr)",
    ),
    "intraday_abs_return_volume_corr": OpDoc(
        "分钟 |收益| 与成交量的相关系数。",
        "计算 |r_i| 与 V_i 的相关系数，度量「波动与量能同步」程度；"
        "是量价同步性的常用代理。",
        r"\mathrm{corr}_i\bigl(|r_i|,\ V_i\bigr)",
    ),
    "intraday_return_activity_corr": OpDoc(
        "收益与交易活跃度的相关系数。",
        "与 return_volume_corr 同类，但活跃度口径可配置"
        "（``activity='volume'``，也可换成笔数等）；刻画收益与活跃度的方向关系。",
        r"\mathrm{corr}_i\bigl(r_i,\ \mathrm{Activity}_i\bigr)",
    ),
    "intraday_signed_volume_imbalance": OpDoc(
        "有符号成交量不平衡。",
        "按 bar 收益方向给成交量赋号（涨为正、跌为负）后求和，"
        "再以当日总量归一；正值表示放量上涨占优。",
        r"\mathrm{SVI}_t = \frac{\sum_i \mathrm{sgn}(r_i)\,V_i}{\sum_i V_i}",
    ),
    "intraday_average_trade_price": OpDoc(
        "当日平均成交价（成交额 / 成交量）。",
        "当日成交额 ÷ 成交量；与 VWAP 的差别在于复权/口径细节，"
        "通常用于与 VWAP 交叉校验。",
        r"\bar{P}_t = \frac{\sum_i A_i}{\sum_i V_i}",
    ),
    "intraday_active_volume_share": OpDoc(
        "主动成交量占比。",
        "按价格上行/下行方向区分的「主动买/主动卖」成交量占比"
        "（上涨 bar 量 / 总量），度量买盘主动性；实现口径以实现为准。",
        r"w^{active}_t = \frac{\sum_{i: r_i>0} V_i}{\sum_i V_i}",
    ),
    "intraday_volume_weighted_return": OpDoc(
        "成交量加权日内收益。",
        "以各 bar 成交量占比为权重对分钟收益加权求和，"
        "强调放量 bar 的收益贡献（放量上涨更「算数」）。",
        r"\mathrm{VWR}_t = \sum_i \frac{V_i}{\sum_j V_j}\,r_i",
    ),
    "intraday_price_impact": OpDoc(
        "日内价格冲击（单位成交引起的价格变动）。",
        "把当日价格变动幅度与成交量（或成交额）相比，度量「成交对价格的推动效率」。",
        r"\mathrm{PI}_t = \frac{\Delta P_t / P_t}{V_t}\ \ \text{（口径以实现为准）}",
    ),
    "intraday_amihud": OpDoc(
        "日内 Amihud 非流动性。",
        "逐 bar 计算 |收益| / 成交额，再取当日均值；值越大表示单位成交额造成的"
        "价格变动越大（越不流动）。",
        r"\mathrm{ILLIQ}_t = \frac{1}{N_t}\sum_i \frac{|r_i|}{A_i}",
    ),
    "intraday_kyle_lambda_proxy": OpDoc(
        "Kyle λ 代理（价格对净买压的敏感度）。",
        "把逐 bar 价格变动对「有符号成交量」做回归，取斜率；"
        "λ 越大表示同样买压推价越多（流动性越差）。",
        r"\Delta p_i = \lambda\, \mathrm{sgn}(r_i)V_i + \varepsilon_i",
    ),
}

# ---------------------------------------------------------------------------
# 日内形态异常度（相对历史）
# ---------------------------------------------------------------------------
_PROFILE: dict[str, OpDoc] = {
    "intraday_profile_deviation": OpDoc(
        "日内形态相对历史均值的偏离。",
        "把当日逐 bar 的形态（价格或量）与历史同 slot 的平均形态比较，"
        "取整体偏离度；用于识别「今天走法不寻常」。",
        r"\frac{1}{N}\sum_i \bigl(P_i - \bar{P}^{hist}_i\bigr)",
    ),
    "intraday_profile_zscore": OpDoc(
        "日内形态离群度（Z 分数）。",
        "以历史同 slot 形态的均值与标准差为基准，把当日形态标准化；"
        "绝对值越大越异常。",
        r"Z_i = \frac{P_i-\bar{P}^{hist}_i}{\sigma^{hist}_i}",
    ),
    "intraday_abnormal_volume_profile": OpDoc(
        "成交量形态异常度。",
        "当日逐 bar 成交量形态相对历史（同 slot）均值的偏离度；"
        "度量「量能分布是否异常」，常用于事前事件识别。",
        r"\mathrm{Dev}^{vol}_t = f\bigl(V_i-\bar{V}^{hist}_i\bigr)",
    ),
    "intraday_abnormal_return_profile": OpDoc(
        "收益形态异常度。",
        "当日逐 bar 收益形态相对历史（同 slot）均值的偏离度。",
        r"\mathrm{Dev}^{ret}_t = f\bigl(r_i-\bar{r}^{hist}_i\bigr)",
    ),
    "intraday_abnormal_vol_profile": OpDoc(
        "波动形态异常度。",
        "当日逐 bar 波动（|收益|）形态相对历史（同 slot）均值的偏离度；"
        "刻画「波动的时间分布」是否与往常不同。",
        r"\mathrm{Dev}^{abvol}_t = f\bigl(|r_i|-\overline{|r|}^{hist}_i\bigr)",
    ),
}

# ---------------------------------------------------------------------------
# 零星缺口（非日内）
# ---------------------------------------------------------------------------
_MISC: dict[str, OpDoc] = {
    "ts_regression_slope": OpDoc(
        "滚动窗口 OLS 斜率（第二个输入对第一个输入的回归斜率）。",
        "在长度 d 的窗口内，以**第一个输入为自变量**、**第二个输入为因变量**做"
        "最小二乘回归，返回斜率 β = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²；"
        "窗口内两者同时有限的样本数须 ≥ 3、且自变量方差 > 0，否则该 bar 为 NaN。"
        "（实现：``_numpy_kernels.ts_regression_slope_``，R19-021/022 修正了旧版"
        "把 β 系统性放大 n/(n−1) 的 ddof 混用问题。）",
        r"\beta_t = \frac{\sum_{k}(x_{t-k}-\bar{x})(y_{t-k}-\bar{y})}"
        r"{\sum_{k}(x_{t-k}-\bar{x})^{2}}",
    ),
    "nonfinite_to_num": OpDoc(
        "把 NaN 与 ±Inf 一律替换为指定数值。",
        "非有限值（NaN、+Inf、−Inf）全部映射为 num（默认 0）；"
        "与 ``nan_to_num`` 的区别：后者只处理 NaN、±Inf 原样透传（R40 #192）。",
        r"\tilde{x} = \begin{cases}x & x\ \text{有限}\\ \mathrm{num} & \text{否则}\end{cases}",
    ),
    "col": OpDoc(
        "按列名读取数据字段。",
        "``col(name)`` 读取数据源逻辑字段名（如 ``close``、``volume``）对应的"
        "面板列，是因子公式的输入叶子；字段含义、单位与时点见「输入字段释义」表。",
        r"\mathrm{col}(\text{name})",
    ),
    "field": OpDoc(
        "按目录字段读取并带上字段契约。",
        "``field(name)`` 读取注册在字段目录中的字段，并把该字段的契约"
        "（单位、时点模型、空值策略、是否允许挖矿等）带入表达式，"
        "比 ``col`` 更严格：字段未登记即失败。",
        r"\mathrm{field}(\text{name})",
    ),
    "ts_rolling_beta": OpDoc(
        "滚动 Beta（协方差 / y 的方差）。",
        "在长度 window 的窗口内计算 cov(x, y) / var(y)（**注意分母是第二个输入 y 的方差**，"
        "实现原样如此：``technical/polars_phase2_indicators.py::ts_rolling_beta``）；"
        "窗口内需满足 ``min_samples=window``（不足则 NaN），var(y)=0 时输出 NULL。",
        r"\beta_t=\frac{\mathrm{Cov}_{w}(x,y)}{\mathrm{Var}_{w}(y)}",
    ),
    "ROLLING_BETA": OpDoc(
        "滚动 Beta（``ts_rolling_beta`` 的大写别名）。",
        "等价于 ``ts_rolling_beta(x, y, window)``：窗口内 cov(x, y) / var(y)。",
        r"\beta_t=\frac{\mathrm{Cov}_{w}(x,y)}{\mathrm{Var}_{w}(y)}",
    ),
    "ts_macd": OpDoc(
        "MACD 线（快慢 EMA 之差，即 DIF）。",
        "对每列价格计算 EMA(fast) − EMA(slow)，默认 fast=12、slow=26，"
        "要求 fast < slow（否则报错）。**该实现的返回值只有 DIF**：形参 signal（默认 9）"
        "未参与计算，也**不返回 DEA 与柱状差**；需要完整 MACD 请显式再算 EMA(DIF, signal)。",
        r"\mathrm{DIF}_t=\mathrm{EMA}_{fast}(x)_t-\mathrm{EMA}_{slow}(x)_t",
    ),
}


EXTENDED_DOCS: dict[str, OpDoc] = {
    **_REALIZED,
    **_SEGMENT,
    **_PATH,
    **_LIMIT,
    **_VWAP,
    **_VOLUME,
    **_FLOW,
    **_PROFILE,
    **_MISC,
}

#: 内部占位条目不应出现在对外文档里。
EXTENDED_DOCS = {
    k: v for k, v in EXTENDED_DOCS.items() if not k.endswith("_placeholder")
}

__all__ = ["EXTENDED_DOCS", "OpDoc", "normalize_extended_name"]
