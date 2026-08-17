"""Curated Chinese explanations for this week's dug factors (53)."""
from __future__ import annotations

from typing import Any

WEEKLY_DUG_FACTOR_GUIDES: dict[str, dict[str, Any]] = {
    "cand_vol_surge_regime_gated": {
        "theme": "放量 + 高振幅日的日内收益（取负平滑）",
        "summary": "只在「振幅高于近20日中位」的日子，用日内涨跌幅×相对放量强度做信号，再取负并做20日EMA。本质是压制「放量冲高振幅」的短线冲动，偏向反向。",
        "steps": [
            "日内收益 = close/open − 1",
            "量强度 = volume / 近20日成交量最大值",
            "振幅门控：仅当 (high−low) > 近20日振幅中位数时保留信号，否则为0",
            "三者相乘后做20日EMA，整体取负输出",
        ],
        "direction": "因子值高：近期较少出现「放量+大振幅」的冲高日，或冲高已被压低；因子值低：频繁放量冲高。",
        "operators": ["ema", "ts_max", "ts_quantile", "volume", "high-low"],
    },
    "ext_38cd4f77": {
        "theme": "振幅–成交量相关 × 收益波动",
        "summary": "看近20日「振幅与成交量是否同向变动」，再乘以同期收益波动率，截面排序。高值表示波动里量价联动强。",
        "steps": [
            "算 high−low 与 volume 的20日时序相关",
            "再乘 close 日收益的20日标准差",
            "全市场截面 rank",
        ],
        "direction": "因子值高：振幅与量同步、且价格波动大；因子值低：量价振幅脱钩或波动低迷。",
        "operators": ["ts_corr", "ts_std", "rank", "volume", "high-low"],
    },
    "cand_tanh_thrust_dynamic_smooth": {
        "theme": "价量推力（sigmoid压缩）+ 放量自适应平滑",
        "summary": "把5日价格变化与5日量变化各自压到(−1,1)，相乘得「价量同向推力」；放量时用短EMA、缩量时用长EMA，最后再乘放量权重并取负。",
        "steps": [
            "价推力 = 2·sigmoid(2·ts_pct(close,5)) − 1；量推力同理",
            "推力 = 价推力 × 量推力",
            "放量权重 w = sigmoid(2·(volume/EMA20 − 1))",
            "输出 −[ w·EMA5(推力) + (1−w)·EMA20(推力) ]",
        ],
        "direction": "因子值高：价量同向推力弱或被取负后偏空侧；解读时务必结合页面 RankIC 符号。",
        "operators": ["sigmoid", "ema", "ts_pct", "safe_div", "volume"],
    },
    "ext_90298bd1": {
        "theme": "收盘相对VWAP偏离的波动",
        "summary": "用近10日 (close−vwap)/vwap 的标准差衡量「收盘相对成交均价」的抖动程度，再截面排序。",
        "steps": [
            "日偏离 = (close − vwap) / vwap",
            "取10日标准差",
            "截面 rank",
        ],
        "direction": "因子值高：收盘相对VWAP摆动大（定价噪音/博弈强）；因子值低：收盘贴近成交均价。",
        "operators": ["ts_std", "rank", "vwap"],
    },
    "cand_vol_regime_trend_smoothed": {
        "theme": "价趋势 × 量比 × 振幅扩张（取负平滑）",
        "summary": "把「相对20日均价偏离 × 相对均量 × 相对中位振幅扩张」合成趋势强度，5日EMA后取负，偏反向处理过热放量扩张。",
        "steps": [
            "价项 = close/ts_mean(close,20) − 1",
            "量项 = volume/ts_mean(volume,20)",
            "振幅项 = (high−low)/中位振幅 − 1",
            "三者相乘 → EMA5 → 取负",
        ],
        "direction": "因子值高：价量振幅扩张偏弱；因子值低：价格相对均线偏高且放量、振幅放大。",
        "operators": ["ema", "ts_mean", "ts_quantile", "safe_div", "volume"],
    },
    "cand_herding_momentum_stress": {
        "theme": "羊群压力：日收益 × 短量比 × 振幅率",
        "summary": "把日涨跌、5日相对放量、振幅/收盘叠乘，短EMA后截断到[−3,3]再取负，刻画「拥挤追涨杀跌」压力。",
        "steps": [
            "日收益 × (volume/5日均量) × ((high−low)/close)",
            "5日EMA，cap 到 [−3,3]",
            "整体取负",
        ],
        "direction": "因子值高：近期羊群冲量偏弱；因子值低：价量振幅共振的追涨压力大。",
        "operators": ["ema", "ts_pct", "ts_mean", "cap", "volume"],
    },
    "cand_vw_ew_spread_responsive": {
        "theme": "成交量加权收益 vs 等权收益差",
        "summary": "比较「按成交量加权的收益」与「等权收益」的差（均用约19日EMA近似原 ewm）。差大说明大成交日子主导方向。",
        "steps": [
            "量加权收益 ≈ EMA(日收益×volume) / EMA(volume)",
            "等权收益 ≈ EMA(日收益)",
            "输出二者之差",
        ],
        "direction": "因子值高：放量日涨跌主导、量价同向更强；因子值低或为负：缩量日主导或量价背离。",
        "operators": ["ema", "ts_pct", "safe_div", "volume"],
    },
    "ext_2b3d9cff": {
        "theme": "50日区间位置 / (高点 + 波动)",
        "summary": "收盘相对50日最低的抬升幅度，除以「50日最高 + 50日收益波动」，再rank。既看位置，又用波动抬高分母防虚高。",
        "steps": [
            "分子 = close − ts_min(low,50)",
            "分母 = ts_max(high,50) + 日收益50日标准差",
            "比值截面 rank",
        ],
        "direction": "因子值高：相对低点抬升明显且波动不过分夸大；因子值低：靠近区间底部或波动很大。",
        "operators": ["ts_min", "ts_max", "ts_std", "rank"],
    },
    "ext_a313f83b": {
        "theme": "短窗口随机指标 × 成交量波动",
        "summary": "10日 %K 式位置（收盘在高低区间中的位置）乘以成交量10日标准差，强调「短线位置 + 量不稳定」。",
        "steps": [
            "位置 = (close−min_low10) / (max_high10−min_low10)",
            "乘 ts_std(volume,10)",
            "截面 rank",
        ],
        "direction": "因子值高：短线偏高位且量波动大；因子值低：低位或量平稳。",
        "operators": ["ts_min", "ts_max", "ts_std", "rank", "volume"],
    },
    "cand_overnight_intraday_divergence_simplified": {
        "theme": "日内收益 vs 隔夜收益的平滑差",
        "summary": "比较平滑后的「日内(close/open)」与「隔夜(open/昨收)」：日内强、隔夜弱则因子偏正，刻画日内/隔夜资金分歧。",
        "steps": [
            "日内 = EMA5(close/open − 1)",
            "隔夜 = EMA5(open/delay(close,1) − 1)",
            "输出 日内 − 隔夜",
        ],
        "direction": "因子值高：日内偏强、隔夜偏弱；因子值低：隔夜跳空主导或日内回吐。",
        "operators": ["ema", "safe_div", "delay"],
    },
    "ext_cc3b5fb0": {
        "theme": "隔夜–日内差的波动标准化",
        "summary": "用 EMA5(日内) − SMA10(隔夜) 得到分歧，再除以该差的20日波动，得到类似z-score的稳定信号。",
        "steps": [
            "diff = EMA5(close/open−1) − SMA10(open/昨收−1)",
            "除以 ts_std(diff,20)（加微小常数防除零）",
        ],
        "direction": "因子值高：日内相对隔夜偏强且经波动标准化后仍突出；负值则相反。",
        "operators": ["ema", "ts_mean", "ts_std", "delay"],
    },
    "ext_7cd58c86": {
        "theme": "价量20日相关",
        "summary": "直接取 close 与 volume 近20日相关系数，衡量价格与成交量同向程度。",
        "steps": [
            "计算 ts_corr(close, volume, 20)",
        ],
        "direction": "因子值高：涨跌与放量同向；因子值低/负：价涨量缩或价跌量增。",
        "operators": ["ts_corr", "volume"],
    },
    "ext_287712d7": {
        "theme": "平滑动量 / 长波动（风险调整）",
        "summary": "20日EMA日收益除以60日收益波动，类似信息比率式动量，再截面rank。",
        "steps": [
            "分子 = EMA20(日收益)",
            "分母 = 日收益60日标准差",
            "比值 rank",
        ],
        "direction": "因子值高：单位风险下的平滑上涨更强；因子值低：弱动量或高波动稀释。",
        "operators": ["ema", "ts_std", "rank", "delay"],
    },
    "ext_62f1f060": {
        "theme": "价格与成交量算术均值（占位/弱信号）",
        "summary": "公式仅为 (close+volume)/2，几乎无经济结构，更像探矿残留或对照项；解读以实证RankIC为准，不宜强行讲故事。",
        "steps": [
            "对每个交易日计算 (close + volume) / 2",
        ],
        "direction": "经济含义薄弱；高/低主要反映价格与成交量量纲混合后的排序，实战慎用。",
        "operators": ["volume"],
    },
    "cand_volume_surge_persistence": {
        "theme": "极端放量日占比（取负）",
        "summary": "标记「成交量 > 2×近20日量中位」的日子，对标记做10日EMA，再取负——惩罚持续极端放量。",
        "steps": [
            "若 volume > 2×ts_quantile(volume,20,0.5) 记1，否则0",
            "对标记做EMA10，整体取负",
        ],
        "direction": "因子值高：近期很少极端放量；因子值低：放量日频繁出现。",
        "operators": ["ema", "where", "ts_quantile", "volume"],
    },
    "cand_gap_reversal_intensity": {
        "theme": "低开高走（缺口反转）强度",
        "summary": "隔夜下跌幅度与当日收阳幅度相乘（只保留低开+收阳组合），近5日求和，衡量「跳空低开后收回」的强度。",
        "steps": [
            "隔夜收益 = open/昨收 − 1；仅保留负值部分的绝对值贡献",
            "日内收益 = close/open − 1；仅保留正值",
            "日得分 = (−隔夜负收益) × 日内正收益",
            "近5日求和",
        ],
        "direction": "因子值高：近几日多次「低开后强力收回」；因子值低：少有缺口反转。",
        "operators": ["ts_sum", "where", "safe_div", "delay"],
    },
    "cand_crash_vol_spike_volregime_boost": {
        "theme": "短长振幅比 × 放量加码（取负）",
        "summary": "用 log(高/低) 的5日均 / 50日均 衡量振幅骤升；若量比>1.5再放大约80%，整体取负，偏「恐慌放量扩张」的反向。",
        "steps": [
            "短振幅 = 均值5(log(high/low))；长振幅 = 均值50",
            "比值 × (1 + 0.8×放量指示)",
            "整体取负",
        ],
        "direction": "因子值高：近期振幅相对平静；因子值低：短窗振幅相对长窗飙升且可能放量。",
        "operators": ["ts_mean", "log", "safe_div", "where", "ema", "volume"],
    },
    "ext_cc11c428": {
        "theme": "中短动量 / 中长波动",
        "summary": "EMA12(日收益)/Std50(日收益) 的风险调整动量，再rank。",
        "steps": [
            "分子 EMA12 日收益；分母 50日收益标准差；截面 rank",
        ],
        "direction": "因子值高：风险调整后偏强；因子值低：弱或高波动。",
        "operators": ["ema", "ts_std", "rank"],
    },
    "ext_73a60264": {
        "theme": "5日涨跌 × 短长量比",
        "summary": "5日价格变化率乘以「5日均量/20日均量」，再截面RANK，经典放量动量。",
        "steps": [
            "ts_pct(close,5) × (均量5 / 均量20)",
            "截面 RANK",
        ],
        "direction": "因子值高：上涨且短量相对长量放大；下跌放量则为负向。",
        "operators": ["ts_pct", "ts_mean", "rank", "volume"],
    },
    "ext_b52617c5": {
        "theme": "40日均价偏离 / 区间宽度",
        "summary": "收盘相对40日均价的偏离，除以同期高低区间宽度，类似标准化位置。",
        "steps": [
            "(close − mean40) / (max_high40 − min_low40)",
            "截面 rank",
        ],
        "direction": "因子值高：相对均线偏高且相对区间不极端宽；因子值低：偏弱。",
        "operators": ["ts_mean", "ts_max", "ts_min", "rank"],
    },
    "cand_fusion_upcap_volregime": {
        "theme": "上涨日成交占比 × 量状态加码",
        "summary": "近20日「上涨日成交量之和 / 总成交量」衡量资金偏向上涨日；再按相对EMA20量比做轻度加码。",
        "steps": [
            "上涨日量占比 = Σ(volume|涨) / Σ(volume)，窗口20",
            "乘以 (1 + 0.5×(volume/EMA20volume − 1))",
        ],
        "direction": "因子值高：成交更集中在上涨日，且当前偏放量；因子值低：下跌日吸走更多成交。",
        "operators": ["ts_sum", "where", "ts_pct", "safe_div", "ema", "volume"],
    },
    "ext_131e5281": {
        "theme": "相对EMA50的价格z分",
        "summary": "(close − EMA50) / Std20(close)，中期趋势相对短期价格波动的标准化。",
        "steps": [
            "偏离 = close − ema(close,50)",
            "除以 ts_std(close,20)，再 rank",
        ],
        "direction": "因子值高：价格显著高于中期均线；因子值低：低于均线或波动大稀释信号。",
        "operators": ["ema", "ts_std", "rank"],
    },
    "ext_d6c071a1": {
        "theme": "实体波动 + 当日实体（取负rank）",
        "summary": "近10日 |close−open| 的标准差加上当日 (close−open)，截面rank后取负。高实体波动+收阳会被压低。",
        "steps": [
            "ts_std(|close−open|,10) + (close−open)",
            "截面 rank，再乘 −1",
        ],
        "direction": "因子值高：实体波动小或当日偏阴；因子值低：实体抖动大且偏阳。注意：含截面rank，FE默认拒跑。",
        "operators": ["ts_std", "rank"],
    },
    "ext_d53e006f": {
        "theme": "20日价格z-score",
        "summary": "经典 (close−mean20)/std20，衡量相对近期均值的标准化偏离。",
        "steps": [
            "(close − ts_mean(close,20)) / ts_std(close,20)",
            "截面 rank",
        ],
        "direction": "因子值高：相对20日均价显著偏高；因子值低：偏低。",
        "operators": ["ts_mean", "ts_std", "rank"],
    },
    "cand_volume_confirmed_intraday_momentum": {
        "theme": "放量确认的日内动量",
        "summary": "日内收益 × 成交量相对20日的z分，再10日滚动均值。只有「放量的日内方向」会被强化。",
        "steps": [
            "日内收益 = (close−open)/open",
            "量z = (volume − 均量20) / 量标准差20",
            "乘积做10日滚动均值",
        ],
        "direction": "因子值高：放量配合的日内上涨持续；因子值低/负：放量下跌或缩量日内涨。",
        "operators": ["ts_mean", "ts_std", "volume"],
    },
    "ext_909472af": {
        "theme": "30日区间位置 × 量比",
        "summary": "收盘在30日高低区间中的位置，乘以 volume/均量30，再rank。",
        "steps": [
            "位置 = (close−min_low30)/(max_high30−min_low30)",
            "× volume/ts_mean(volume,30)，rank",
        ],
        "direction": "因子值高：偏高位且相对放量；因子值低：低位或缩量。",
        "operators": ["ts_min", "ts_max", "ts_mean", "rank", "volume"],
    },
    "cand_gpdev_volregime_ema10": {
        "theme": "几何中枢偏离 × 量状态",
        "summary": "收盘相对 √(high·low) 的偏离，乘以 (量比−1)，再EMA10。价格偏离几何中枢且放量时信号更强。",
        "steps": [
            "偏离 = close/√(high·low) − 1",
            "× (volume/EMA20volume − 1)",
            "EMA10 平滑",
        ],
        "direction": "因子值高：收盘高于几何中枢且偏放量；缩量或低于中枢则偏低/负。",
        "operators": ["ema", "safe_div", "volume"],
    },
    "ext_1ef16bdf": {
        "theme": "隔夜 vs 日内的相对差（归一）",
        "summary": "(隔夜收益 − 日内收益) / (|隔夜|+|日内|)，衡量两者谁主导且做幅度归一。",
        "steps": [
            "隔夜 = open/昨收 − 1；日内 = close/open − 1",
            "分子 = 隔夜 − 日内；分母 = |隔夜|+|日内|+ε",
        ],
        "direction": "因子值高：隔夜相对日内更强；因子值低：日内主导或隔夜弱。",
        "operators": ["delay"],
    },
    "ext_8d6b4d53": {
        "theme": "20日区间位置 × 量比",
        "summary": "与909472af同结构，窗口改为20日。",
        "steps": [
            "位置 = (close−min20)/(max20−min20)",
            "× volume/均量20，截面 rank",
        ],
        "direction": "因子值高：20日偏高位且放量；因子值低：低位或缩量。",
        "operators": ["ts_min", "ts_max", "ts_mean", "rank", "volume"],
    },
    "ext_7f827abe": {
        "theme": "相对WMA20的价格z分",
        "summary": "(close − WMA20) / Std20(close)，加权均线版均值回归/趋势位置。",
        "steps": [
            "偏离 = close − WMA(close,20)",
            "除以 ts_std(close,20)，rank",
        ],
        "direction": "因子值高：显著高于加权均线；因子值低：低于均线。",
        "operators": ["WMA", "ts_std", "rank"],
    },
    "ext_b512e792": {
        "theme": "5日涨跌 / 长波动（取负）",
        "summary": "−Δclose_5 / Std60(日收益)，把中期波动标准化后的短动量取反，偏反转。",
        "steps": [
            "ts_delta(close,5) / 日收益60日标准差",
            "取负后截面 rank",
        ],
        "direction": "因子值高：近5日偏弱（反转视角看好）；因子值低：近5日偏强。",
        "operators": ["ts_delta", "ts_std", "rank"],
    },
    "ext_50b318f3": {
        "theme": "极短平滑动量 / 短波动",
        "summary": "EMA5(日收益)/Std20(日收益)，超短风险调整动量。",
        "steps": [
            "EMA5 日收益 ÷ 20日收益标准差，rank",
        ],
        "direction": "因子值高：短线风险调整偏强；因子值低：偏弱。",
        "operators": ["ema", "ts_std", "rank"],
    },
    "ext_760efb3b": {
        "theme": "5日涨跌 / 20日收益波动",
        "summary": "Δclose_5 经20日收益波动标准化后的动量rank。",
        "steps": [
            "ts_delta(close,5) / ts_std(日收益,20)，rank",
        ],
        "direction": "因子值高：单位波动下近5日上涨更强。",
        "operators": ["ts_delta", "ts_std", "rank"],
    },
    "cand_vol_asymmetry_diff_blend": {
        "theme": "涨跌波动不对称：短长窗按波动状态混合",
        "summary": "分别算5日/20日「下跌波动−上涨波动」；用ATR/close的63日分位决定更信短窗还是长窗，高波动时更信短不对称。",
        "steps": [
            "上涨日/下跌日分别估滚动波动，得 down_std − up_std（5日与20日）",
            "vol_rank = ATR14/close 在63日窗口的百分位",
            "输出 vol_rank×短不对称 + (1−vol_rank)×长不对称",
        ],
        "direction": "因子值高：下跌波动相对上涨更大（跌更乱）；低波动时更看长窗。",
        "operators": ["ts_std", "where", "ATR", "rank"],
    },
    "cand_liquidity_spread_vol_continuous_median": {
        "theme": "日内−隔夜价差（中位平滑）× 量比",
        "summary": "先做日内收益减隔夜收益，5日滚动中位数平滑，再乘截断到[0.5,2]的量比，连续调节流动性状态。",
        "steps": [
            "spread = 日内收益 − 隔夜收益",
            "5日滚动中位数平滑",
            "× clip(volume/均量相关量比, 0.5, 2)",
        ],
        "direction": "因子值高：日内强于隔夜且有一定放量配合；因子值低：隔夜主导或缩量。",
        "operators": ["ts_median", "cap", "volume"],
    },
    "ext_e1d40349": {
        "theme": "5日涨跌 / 振幅波动",
        "summary": "Δclose_5 除以 high−low 的20日标准差，用振幅波动做风险调整。",
        "steps": [
            "ts_delta(close,5) / ts_std(high−low,20)，rank",
        ],
        "direction": "因子值高：在振幅波动可控下近5日上涨更强。",
        "operators": ["ts_delta", "ts_std", "rank", "high-low"],
    },
    "cand_gap_vol_cluster_adaptive_w30": {
        "theme": "隔夜–日内差×量比，叠加波动聚集惩罚",
        "summary": "核心是 (日内−隔夜)×量比 经sigmoid压缩；再乘一项与「平方收益自相关」相关的聚集调节（约30日），波动聚集时调节信号强度。",
        "steps": [
            "core = (close/open−1) − (open/昨收−1)，再×量比并压缩到(−1,1)",
            "波动聚集项：−corr(平方收益, 滞后平方收益, 30) 经sigmoid",
            "输出 core × (1 + 0.5×聚集项)",
        ],
        "direction": "因子值高：日内相对隔夜偏强且量配合，并受波动聚集状态调节。",
        "operators": ["sigmoid", "ts_corr", "safe_div", "ema", "delay", "volume"],
    },
    "cand_pressure_tanh": {
        "theme": "尾盘压力偏离（取负）",
        "summary": "用 (close−low)/(high−low)×volume 近似「收在相对高位的成交压力」，对20日均值做z分，再sigmoid压到(−1,1)后取负。",
        "steps": [
            "压力 = 收盘在当日区间位置 × volume",
            "z = (压力 − mean20) / std20",
            "输出 −(2·sigmoid(2z)−1)",
        ],
        "direction": "因子值高：相对历史，收高位放量压力并不极端；因子值低：收高位放量显著高于常态。",
        "operators": ["sigmoid", "ts_mean", "ts_std", "volume"],
    },
    "cand_herding_pv_sync_sma": {
        "theme": "价量同号同步率（羊群）",
        "summary": "若日收益符号与成交量变化符号相同记+1、相反记−1、任一方为0记0，再20日简单平均。衡量价量方向是否「齐步走」。",
        "steps": [
            "sign(日收益) 与 sign(量变化)",
            "同号=+1，异号=−1，否则0",
            "20日滚动均值",
        ],
        "direction": "因子值高：近20日价涨量增/价跌量缩更常见；因子值低：价量经常背离。",
        "operators": ["ts_mean", "volume"],
    },
    "cand_geometric_deviation_vol_asym_tilt": {
        "theme": "几何偏离能量 × 上涨波动倾斜",
        "summary": "用 (2logC−logH−logL)/3 的几何偏离及其绝对值相乘得「偏离能量」，EMA5后，仅当上涨波动>下跌波动时再乘倾斜系数。",
        "steps": [
            "g = (2·log(close)−log(high)−log(low))/3",
            "能量 = EMA5(g · |g|)",
            "若上涨日波动 > 下跌日波动：乘 (1+倾斜比)，否则保持",
        ],
        "direction": "因子值高：收盘相对高低几何中枢偏离大且上涨波动占优；否则偏弱。",
        "operators": ["ema", "log", "ts_std", "where", "ts_pct"],
    },
    "ext_70b9e66b": {
        "theme": "5日涨跌 / √量波动（取负）",
        "summary": "−Δclose_5 / √Std20(volume)，用成交量波动开方做分母，取负偏反转。",
        "steps": [
            "ts_delta(close,5) / sqrt(ts_std(volume,20))",
            "取负后 rank",
        ],
        "direction": "因子值高：近5日偏弱（反转视角）；因子值低：近5日偏强。",
        "operators": ["ts_delta", "ts_std", "rank", "volume"],
    },
    "ext_bdd0fb42": {
        "theme": "对数空间的50日区间位置",
        "summary": "在log价格上算 (logC−logLmin)/(logHmax−logLmin)，长窗口相对位置，减轻价格水平差异。",
        "steps": [
            "分子 log(close)−log(ts_min(low,50))",
            "分母 log(ts_max(high,50))−log(ts_min(low,50))",
            "比值 rank",
        ],
        "direction": "因子值高：对数尺度上靠近50日高区；因子值低：靠近低区。",
        "operators": ["log", "ts_min", "ts_max", "rank"],
    },
    "cand_reversal_volregime_overnight_csz": {
        "theme": "波动扩张下的反转，隔夜跳空衰减",
        "summary": "−3日涨跌 × (短ATR/长ATR)；隔夜跳空相对自身波动越大，信号衰减越强，避免大缺口日误用反转。",
        "steps": [
            "raw = −ts_pct(close,3) × (均振幅5/均振幅20)",
            "隔夜z = |open/昨收−1| / 隔夜收益20日标准差",
            "输出 raw / (1+隔夜z)",
        ],
        "direction": "因子值高：近3日偏弱且振幅扩张、隔夜不过分夸张；大跳空日信号被压低。",
        "operators": ["ts_pct", "ts_mean", "safe_div", "ts_std", "delay"],
    },
    "cand_vol_persistence_momentum": {
        "theme": "5日动量 × 波动持续性",
        "summary": "5日收益乘以（平方收益20日自相关 − 0.5）。波动聚集强时放大动量符号，聚集弱时削弱甚至反号。",
        "steps": [
            "ret_5d = 5日涨跌幅",
            "vol_ac = corr(平方收益, 滞后1日平方收益, 20)",
            "输出 ret_5d × (vol_ac − 0.5)",
        ],
        "direction": "因子值高：上涨且波动持续性强；波动不持续时动量被打折。",
        "operators": ["ts_corr", "ts_pct"],
    },
    "ext_b67c0953": {
        "theme": "开收差 / 短收益波动",
        "summary": "(open−close)/Std10(日收益)，开盘相对收盘的日内回吐强度经短波动标准化。",
        "steps": [
            "(open − close) / ts_std(日收益,10)，rank",
        ],
        "direction": "因子值高：相对波动而言收盘弱于开盘（偏阴实体）；因子值低：收盘强于开盘。",
        "operators": ["ts_std", "rank"],
    },
    "ext_52c6c08d": {
        "theme": "风险调整5日动量（取负rank）",
        "summary": "−rank( Δclose_5 / (Std20(日收益)+0.001) )，标准化动量取反。含截面rank，FE默认拒跑。",
        "steps": [
            "Δclose_5 / (日收益20日标准差 + 0.001)",
            "截面 rank 后取负",
        ],
        "direction": "因子值高：近5日风险调整动量偏弱；因子值低：偏强。",
        "operators": ["ts_delta", "ts_std", "rank"],
    },
    "ext_981562e9": {
        "theme": "40日位置 / (区间宽 + 波动)",
        "summary": "相对40日低点的抬升，除以「区间宽度 + 收益波动」，再rank，防纯区间虚高。",
        "steps": [
            "分子 close−min_low40",
            "分母 (max_high−min_low) + Std40(日收益)",
            "比值 rank",
        ],
        "direction": "因子值高：相对低点抬升且风险项不过分大。",
        "operators": ["ts_min", "ts_max", "ts_std", "rank"],
    },
    "ext_773ab168": {
        "theme": "当日实体 / 实体波动",
        "summary": "(close−open)/Std10(close−open)，当日阳阴实体相对自身历史波动的标准化。",
        "steps": [
            "(close − open) / ts_std(close−open, 10)，rank",
        ],
        "direction": "因子值高：当日阳线相对近期实体波动偏大；阴线则为低/负。",
        "operators": ["ts_std", "rank"],
    },
    "ext_39e7b7a6": {
        "theme": "负实体 / 长价格波动",
        "summary": "−(close−open)/Std60(close)，收阴偏正、收阳偏负，经长波动标准化后rank。",
        "steps": [
            "−(close−open) / ts_std(close,60)，rank",
        ],
        "direction": "因子值高：相对长波动，当日偏收阴；因子值低：偏收阳。",
        "operators": ["ts_std", "rank"],
    },
    "ext_1490f2fe": {
        "theme": "相对30日低点 / (高低波动和)",
        "summary": "(close−min_low30)/(Std30(high)+Std30(low))，抬升幅度用高低各自波动之和做分母。",
        "steps": [
            "分子 close − ts_min(low,30)",
            "分母 ts_std(high,30)+ts_std(low,30)",
            "比值 rank",
        ],
        "direction": "因子值高：相对低点抬升且高低波动之和不大。",
        "operators": ["ts_min", "ts_std", "rank"],
    },
    "cand_vol_expansion_atr_ratio": {
        "theme": "阴阳方向 × 短长振幅扩张",
        "summary": "用 sign(close−open) 定方向，乘 sigmoid压缩后的「5日均振幅/60日均振幅 − 1」，振幅扩张时强化阴阳信号。",
        "steps": [
            "扩张 = 均振幅5 / 均振幅60 − 1，经 sigmoid(5·扩张)",
            "乘 sign(close−open)",
        ],
        "direction": "因子值高：收阳且短窗振幅相对长窗扩张；收阴扩张则为负向。",
        "operators": ["sigmoid", "ts_mean", "safe_div", "sign"],
    },
    "ext_2fffdf67": {
        "theme": "20日位置 / (区间宽 + 波动)",
        "summary": "与981562e9同结构，窗口20日。",
        "steps": [
            "(close−min_low20) / ((max−min) + Std20(日收益))，rank",
        ],
        "direction": "因子值高：相对20日低点抬升且分母风险项可控。",
        "operators": ["ts_min", "ts_max", "ts_std", "rank"],
    },
    "cand_mut_cro_asym_vol_ratio_med_20_minp3_vol6_ema5": {
        "theme": "阳阴振幅中位比 × 短量比（取负平滑）",
        "summary": "用20日中位数分别估「收阳日振幅」与「收阴日振幅」之比，再乘 volume/近6日量中位，EMA5后取负。",
        "steps": [
            "阳振幅中位 / 阴振幅中位（窗口20，where 分流）",
            "× volume / ts_quantile(volume,6,0.5)",
            "整体取负后 EMA5",
        ],
        "direction": "因子值高：阴线振幅相对更突出或量不强；因子值低：阳线振幅中位显著大于阴线且放量。",
        "operators": ["ema", "ts_quantile", "where", "volume", "high-low"],
    },
}


def assert_complete(names: list[str]) -> list[str]:
    missing = [n for n in names if n not in WEEKLY_DUG_FACTOR_GUIDES]
    return missing


# Permanent formula-sign-flip annotations (auto-maintained)
_FLIP_STEP = "最后整体取负：输出 = -(上述信号)，使因子与前瞻收益同向（RankIC>0）"
_FLIP_SUMMARY_PREFIX = "【公式已取负】原始信号与收益负相关，已在最外层加 -(…)。"
_FLIP_DIRECTION = {
    'ext_38cd4f77': '取负后：因子值高表示振幅–成交量相关偏弱或波动不突出；因子值低表示量价振幅同向且波动大。',
    'ext_90298bd1': '取负后：因子值高表示收盘相对 VWAP 更贴近、抖动小；因子值低表示偏离波动大。',
    'ext_2b3d9cff': '取负后：因子值高偏向区间相对偏弱/分母风险项更大；因子值低表示相对低点抬升更明显。',
    'ext_a313f83b': '取负后：因子值高偏向短线低位或量波动小；因子值低表示短线偏高位且量不稳。',
    'ext_cc3b5fb0': '取负后：因子值高表示日内相对隔夜并不占优（经波动标准化后）；负向/低值则日内更强。',
    'ext_7cd58c86': '取负后：因子值高表示价量相关偏弱或背离；因子值低表示价量同向相关更强。',
    'ext_287712d7': '取负后：因子值高表示风险调整后的平滑上涨偏弱；因子值低表示单位风险动量更强。',
    'ext_62f1f060': '取负后：对 (close+volume)/2 整体取负后截面排序；经济含义仍弱，以实证 IC 为准。',
    'ext_cc11c428': '取负后：因子值高表示中短风险调整动量偏弱；因子值低表示偏强。',
    'ext_73a60264': '取负后：因子值高表示 5 日涨跌×短长量比 的原始动量偏弱；因子值低表示放量动量更强。',
    'ext_b52617c5': '取负后：因子值高表示相对 40 日均价偏弱；因子值低表示相对均价偏高。',
    'ext_131e5281': '取负后：因子值高表示相对 EMA50 的 z 分偏低；因子值低表示显著高于中期均线。',
    'ext_d53e006f': '取负后：因子值高表示相对 20 日均价偏低；因子值低表示偏高（经典 z-score 取反）。',
    'ext_909472af': '取负后：因子值高偏向 30 日低位或缩量；因子值低表示偏高位且放量。',
    'ext_8d6b4d53': '取负后：因子值高偏向 20 日低位或缩量；因子值低表示偏高位且放量。',
    'ext_7f827abe': '取负后：因子值高表示相对 WMA20 偏弱；因子值低表示显著高于加权均线。',
    'ext_50b318f3': '取负后：因子值高表示极短风险调整动量偏弱；因子值低表示偏强。',
    'ext_760efb3b': '取负后：因子值高表示近 5 日风险调整上涨偏弱；因子值低表示偏强。',
    'ext_e1d40349': '取负后：因子值高表示近 5 日相对振幅波动的上涨偏弱；因子值低表示偏强。',
    'ext_bdd0fb42': '取负后：因子值高表示对数尺度上更靠近 50 日低区；因子值低表示靠近高区。',
    'ext_981562e9': '取负后：因子值高表示相对 40 日低点抬升偏弱或风险项大；因子值低表示抬升更明显。',
    'ext_773ab168': '取负后：因子值高表示当日阳线实体相对自身波动并不突出（偏阴/弱阳）；因子值低表示偏强阳。',
    'ext_1490f2fe': '取负后：因子值高表示相对 30 日低点抬升偏弱；因子值低表示抬升更明显。',
    'ext_2fffdf67': '取负后：因子值高表示相对 20 日低点抬升偏弱；因子值低表示抬升更明显。',
    'cand_vw_ew_spread_responsive': '取负后：因子值高表示量加权收益相对等权并不占优；因子值低表示放量日主导更强。',
    'cand_overnight_intraday_divergence_simplified': '取负后：因子值高表示隔夜相对日内更强；因子值低表示日内相对隔夜更强。',
    'cand_gap_reversal_intensity': '取负后：因子值高表示近期缺口反转（低开收回）并不强；因子值低表示反转强度大。',
    'cand_fusion_upcap_volregime': '取负后：因子值高表示上涨日成交占比/放量加码并不突出；因子值低表示上涨日吸量更强。',
    'cand_gpdev_volregime_ema10': '取负后：因子值高表示收盘相对几何中枢偏弱或缩量；因子值低表示偏高且放量。',
    'cand_gap_vol_cluster_adaptive_w30': '取负后：因子值高表示日内相对隔夜的量价差信号偏弱；因子值低表示该分歧信号更强。',
    'cand_geometric_deviation_vol_asym_tilt': '取负后：因子值高表示几何偏离能量/上涨波动倾斜并不强；因子值低表示偏离能量更大。',
    'cand_vol_expansion_atr_ratio': '取负后：因子值高表示阴阳方向×振幅扩张信号偏弱；因子值低表示扩张配合的方向信号更强。',

}
for _n, _dir in _FLIP_DIRECTION.items():
    _g = WEEKLY_DUG_FACTOR_GUIDES.get(_n)
    if not _g:
        continue
    _s = str(_g.get("summary") or "")
    for _p in ("【已取负】", "【公式已取负】"):
        if _s.startswith(_p):
            _s = _s[len(_p):]
    # drop pre-flip "高值/因子值高…" tails that contradict post-flip direction
    for _sep in ("。高值", "；高值", "。因子值高", "；因子值高"):
        if _sep in _s:
            _s = _s.split(_sep)[0]
            if not _s.endswith("。"):
                _s = _s + "。"
            break
    _g["summary"] = _FLIP_SUMMARY_PREFIX + _s
    _steps = [x for x in list(_g.get("steps") or []) if "整体取负" not in str(x) and "输出 = -(" not in str(x)]
    _steps.append(_FLIP_STEP)
    _g["steps"] = _steps
    _g["direction"] = _dir
    _th = str(_g.get("theme") or "")
    if "已取负" not in _th:
        _g["theme"] = _th + " · 已取负调正"
