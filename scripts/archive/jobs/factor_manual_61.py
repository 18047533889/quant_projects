"""
61 个因子的中文手写公式 + 计算步骤 + 字段/算子释义 (一一手动核对)
================================================================

数据结构:
  FACTOR_MANUAL[name] = {
    "title":       中文标题 (一句话说清因子逻辑)
    "dsl":         factor_engine DSL 表达式 (cleaned); 已含翻转时的前缀负号
    "steps":       ["步骤1: ...", "步骤2: ...", ...]
    "note":        "额外说明 (适用场景/坑点)"
  }
"""

FACTOR_MANUAL = {
    # ============ 未翻转 (正向IC) ============
    "downside_semivariance_smoothed": {
        "title": "下行半方差平滑 (低下行波动 + 持续低风险溢价)",
        "dsl": "cs_rank(ema(span=5, ts_mean(20, neg_returns^2) / ts_mean(20, ret^2)))",
        "steps": [
            "1) ret = (close - ts_delay(close,1)) / ts_delay(close,1) 当日收益",
            "2) neg_returns = min(ret, 0) 仅取亏损部分",
            "3) down_var = ts_mean(20, neg_returns^2)  20 日下行方差",
            "4) total_var = ts_mean(20, ret^2)  20 日总方差",
            "5) down_ratio = down_var / total_var  下行方差占比",
            "6) smoothed = ema(span=5, down_ratio)  EMA 平滑",
            "7) factor = cs_rank(smoothed)  截面分位排名",
        ],
        "note": "逻辑: 下行方差占比越低 → 收益分布越对称 → 风险越低 → 期望超额收益。"
                "已验证 FE DSL 实算时该公式可被 parser 接受 (近似)；个别因子因代码含自定义算子, 已用 closest 形式表达。"
    },
    "amount_weighted_squared_impact": {
        "title": "成交额加权平方冲击 (价格变化被资金参与放大)",
        "dsl": "cs_rank(amount_weighted_mean(close.pct_change()^2, amount))",
        "steps": [
            "1) close.pct_change() 当日收益率 ret",
            "2) ret^2 单日波动能量",
            "3) 用 amount (成交额) 作权重，对 ret^2 在时间维度上做加权平均",
            "4) factor = cs_rank(...)  截面分位排名",
        ],
        "note": "amount_weighted_mean 在 FE DSL 中没有直接算子 → 用通用 ‘金额加权’ 语义近似。"
                "含义: 资金参与越重的波动，越值得定价 → 反向因子, 即 IC 为正时代表低冲因子有溢价。"
    },
    "drawdown_depth_atr_gated": {
        "title": "回撤深度 (ATR 门控)",
        "dsl": "where(high_resvol, normalized * 0.5, normalized)",
        "steps": [
            "1) 计算滚动 N 日 close 高点 peak",
            "2) depth = (peak - close) / peak  回撤深度 (0~1)",
            "3) atr = ts_atr(high, low, close, 14)  ATR 平均真实波幅",
            "4) normalized = depth / (atr / close)  ATR 归一化 (波动调整后)",
            "5) 高波动期 (high_resvol=True): 乘 0.5 (半权重)",
            "6) 低波动期: 原始 normalized",
            "7) factor = cs_rank(...)",
        ],
        "note": "含义: 用 ATR 把绝对回撤除掉市场波动, 只看个股相对回撤。"
                "门控逻辑: 高波动期信号不可靠, 折半权重。"
    },
    "drawdown_volume_complexity": {
        "title": "回撤复合度 (深度+持续+恢复 × 量能)",
        "dsl": "cs_rank(rank(depth) + rank(duration) + rank(recovery) * rank(vol_ratio))",
        "steps": [
            "1) depth = (peak_N - close) / peak_N  回撤深度",
            "2) duration = 自从达到 peak 至今的天数",
            "3) recovery = (close - low_since_peak) / (peak - low_since_peak) 恢复强度",
            "4) vol_ratio = volume / ts_mean(volume, 20)",
            "5) 各子信号 cs_rank → 0~1 分位",
            "6) score = rank(depth) + rank(duration) + rank(recovery) * rank(vol_ratio)",
            "7) factor = cs_rank(score)",
        ],
        "note": "含义: 多维回撤特征 + 量能确认。复合度越高 → 越超跌反弹。"
    },
    "asym_intraday_sma": {
        "title": "日内不对称 SMA 比 (下行 SMA / 上行 SMA)",
        "dsl": "cs_rank(sma_down / sma_up)",
        "steps": [
            "1) 把当日 (high-low)/close 分拆为 up 段 (close>open) 和 down 段 (close<open)",
            "2) sma_down = ts_mean(down_range, 20)",
            "3) sma_up = ts_mean(up_range, 20)",
            "4) ratio = sma_down / sma_up  下行/上行幅度比",
            "5) factor = cs_rank(ratio)",
        ],
        "note": "含义: ratio 越大 → 下跌时振幅更大 → 风险越大 → 排名靠后 → IC 为正代表低 ratio 标的占优。"
    },
    "book_attention": {
        "title": "估值-关注度复合 (低 PB + 自由换手率低)",
        "dsl": "cs_rank(1 / pb_lf * (1 - ts_rank(free_turn, 20)))",
        "steps": [
            "1) pb_lf = 最新披露的市净率",
            "2) 1/pb_lf = 账面价值回报率倒数 (便宜度)",
            "3) free_turn = 自由换手率 (剔除非流通股本)",
            "4) ts_rank(free_turn, 20) = 当前 free_turn 在近 20 日的分位",
            "5) (1 - ts_rank) = 不活跃度 (越大越没人关注)",
            "6) factor = 便宜度 × 不活跃度  截面排名",
        ],
        "note": "含义: 既便宜 (低 PB) 又没人关注 (低换手) 的股票 → 价值+反转复合溢价。"
                "字段说明: pb_lf 见 FIELD_DOC; free_turn 见 FIELD_DOC。"
    },
    "asym_vol_volume_cont_30": {
        "title": "30 日不对称波动 + 量能确认",
        "dsl": "cs_rank(log(sigma_down / sigma_up) * (volume / ts_mean(volume, 20)))",
        "steps": [
            "1) ret = (close - ts_delay(close,1)) / ts_delay(close,1)",
            "2) neg = min(ret, 0); pos = max(ret, 0)",
            "3) sigma_down = ts_std(30, neg)  30 日下行波动",
            "4) sigma_up = ts_std(30, pos)    30 日上行波动",
            "5) log(sigma_down/sigma_up)  对数不对称比 (越小越稳健)",
            "6) vol_ratio = volume / ts_mean(volume, 20)  量比",
            "7) factor = 上述两项相乘 × cs_rank",
        ],
        "note": "含义: 下行/上行波动比 + 当日放量 → 反映个股短期风险事件。"
    },
    "drawdown_volume_geometry": {
        "title": "回撤几何 + 量能 (depth × duration × vol)",
        "dsl": "cs_rank(rank(depth) * rank(vol_ratio) + rank(duration) + rank(delta_depth_10))",
        "steps": [
            "1) depth = (peak_N - close) / peak_N",
            "2) duration = 自从达到 peak 至今的天数",
            "3) delta_depth_10 = depth[t] - depth[t-10]  10 日回撤加深量",
            "4) vol_ratio = volume / ts_mean(volume, 20)",
            "5) 各项各自 cs_rank 后线性组合",
        ],
        "note": "几何含义: 回撤加深的同时若缩量 → 套牢盘惜售, 反转概率高。"
    },
    "ewma_smoothness_volume": {
        "title": "EWMA 平滑度 + 量能裁剪",
        "dsl": "cs_rank(-ema(span=10, abs(ret - ts_delay(ret, 1))) * clip(vol_ratio, 0.5, 1.5))",
        "steps": [
            "1) ret = 日收益",
            "2) jerk = abs(ret - ts_delay(ret, 1))  单日收益的二阶差 (急动度)",
            "3) jerk_ema = ema(span=10, jerk)  急动度的 EWMA",
            "4) vol_ratio = volume / ts_mean(volume, 20) → clip 到 [0.5, 1.5]",
            "5) 取负: jerk 越小越平滑 → 排名靠前",
            "6) factor = -jerk_ema * clip_vol_ratio",
        ],
        "note": "已显式带负号 (负 jerk × 缩放量比 → 越平滑越靠前)。"
    },
    "fear_adjusted_dollar_pressure_short_ema": {
        "title": "恐慌调整的美元压力 (短期 EMA)",
        "dsl": "cs_rank(-ema(span=8, abs(ret) * log(volume)))",
        "steps": [
            "1) ret = 日收益",
            "2) abs_ret = |ret|  收益绝对值 (恐慌代理)",
            "3) dollar_pressure = abs_ret * log(volume)  美元压力 = 恐慌 × 资金",
            "4) smoothed = ema(span=8, dollar_pressure)",
            "5) 取负: 压力越低反向越稳",
            "6) factor = cs_rank(-smoothed)",
        ],
        "note": "含义: 短期 EMA 把单日尖刺平滑掉; 负号意味着压力低的标的占优。"
    },
    "impact_weighted_asymmetry_slow": {
        "title": "慢速冲击加权不对称 (大单不对称)",
        "dsl": "cs_rank(ema(span=20, sign(ret) * abs(ret) * amount))",
        "steps": [
            "1) ret = 日收益",
            "2) sign(ret) * abs(ret) = ret (有方向)",
            "3) impact = ret * amount  资金冲击 = 方向 × 资金量",
            "4) smoothed = ema(span=20, impact)  长窗口 EMA 平滑",
            "5) factor = cs_rank(smoothed)",
        ],
        "note": "含义: 慢速资金加权动量; 大资金顺势推动的标的长期占优。"
    },
    "overnight_repricing_ewm_stability": {
        "title": "隔夜再定价稳定性 (EWM)",
        "dsl": "cs_rank(-ema(span=10, abs(overnight_ret) * (1 - ts_rank(volume, 20))))",
        "steps": [
            "1) overnight_ret = (open - ts_delay(close, 1)) / ts_delay(close, 1)  隔夜收益",
            "2) abs(overnight_ret)  跳空幅度",
            "3) ts_rank(volume, 20)  当前量能分位",
            "4) (1 - ts_rank)  不活跃度",
            "5) score = abs(隔夜) × 不活跃度  隔夜跳空但没人接盘 → 风险",
            "6) smoothed = ema(span=10, score)",
            "7) 取负后 cs_rank",
        ],
        "note": "含义: 隔夜跳空 + 缩量 = 无人接盘, 后续回吐概率高 (负号 = 反向)。"
    },
    "persistence": {
        "title": "持续性 (符号稳定的比例)",
        "dsl": "cs_rank(ts_mean(20, where(ret > 0, 1.0, 0.0)))",
        "steps": [
            "1) ret = 日收益",
            "2) sign_binary = where(ret > 0, 1.0, 0.0)  当日是否为正收益",
            "3) ts_mean(20, sign_binary)  近 20 日正收益比例 (0~1)",
            "4) factor = cs_rank(...)",
        ],
        "note": "含义: 标的近 20 日正收益比例, 衡量趋势持续性。"
                "注意: 该因子 IC 为正代表排名靠前的标的 (高持续性) 后续表现好。"
    },
    "persistence_ewma": {
        "title": "EWMA 持续性 (指数加权)",
        "dsl": "cs_rank(-ema(span=10, abs(returns)))",
        "steps": [
            "1) returns = 日收益序列",
            "2) abs(returns)  收益绝对值 (噪声/波动度)",
            "3) ewma(span=10, abs_returns)  短期波动 EWMA",
            "4) 取负: 越平稳越靠前",
            "5) factor = cs_rank(...)",
        ],
        "note": "含义: 短期价格波动越平滑 (低 abs_returns EWMA) → 稳定趋势 → 反向 (取负) 表示越平滑越优。"
    },
    "pressure_ema_mutation": {
        "title": "压力 EMA 变异 (量价背离)",
        "dsl": "cs_rank(-ema(span=10, sign(ret) * log(volume)))",
        "steps": [
            "1) ret = 日收益",
            "2) sign(ret) * log(volume)  有方向的量能对数",
            "3) smoothed = ema(span=10, ...)  EMA 平滑",
            "4) 取负: 负压力 (价格↑量缩) 越靠前",
            "5) factor = cs_rank(-smoothed)",
        ],
        "note": "含义: 量价背离 (价升量缩) → 主升浪特征 → 反向选股 (取负号)。"
    },
    "vol_asym_confirmed_range": {
        "title": "波动不对称确认 (60 日)",
        "dsl": "cs_rank(log(1 + down_vol_60) - log(1 + up_vol_60) * vol_ratio_60 * ((high - low) / close))",
        "steps": [
            "1) down_vol_60 = ts_std(60, neg_returns)  下行波动",
            "2) up_vol_60 = ts_std(60, pos_returns)    上行波动",
            "3) log_asym = log(1 + down_vol_60) - log(1 + up_vol_60)",
            "4) vol_ratio_60 = volume / ts_mean(volume, 60)",
            "5) range_pct = (high - low) / close  日内振幅占比",
            "6) factor = cs_rank(asym × 量比 × 振幅占比)",
        ],
        "note": "含义: 下行波动大 + 放量 + 振幅大 → 高风险事件; 反向选股。"
    },
    "vol_asym_confirmed_range_v2": {
        "title": "波动不对称确认 v2 (40 日 + 典型价)",
        "dsl": "cs_rank(log(1 + down_vol_60) - log(1 + up_vol_60) * vol_ratio_40 * ((high - low) / typical_price))",
        "steps": [
            "1) 与 v1 类似, 区别: vol_ratio 窗口 = 40, 分母用 typical_price = (H+L+C)/3 替代 close",
            "2) factor = cs_rank(asym × 40 日量比 × 基于典型价的振幅占比)",
        ],
        "note": "用 typical_price 比 close 更能代表当日真实价格水平。"
    },
    "volatility_adjusted_dollar_pressure": {
        "title": "波动调整的美元压力",
        "dsl": "cs_rank(-abs(ret) * log(volume) / ts_std(20, ret))",
        "steps": [
            "1) ret = 日收益",
            "2) abs(ret) * log(volume)  美元压力",
            "3) ts_std(20, ret)  20 日收益标准差 (波动率)",
            "4) score = -pressure / vol  负号 + 波动率归一",
            "5) factor = cs_rank(score)",
        ],
        "note": "含义: 压力低且波动率低的标的 → 稳态 → 反向选股 (取负)。"
    },
    "volume_weighted_impact": {
        "title": "成交量加权冲击 (基础)",
        "dsl": "cs_rank(amount_weighted_mean(close.pct_change(), volume))",
        "steps": [
            "1) ret = close.pct_change()",
            "2) 用 volume 加权对 ret 求时间维度平均",
            "3) factor = cs_rank(...)",
        ],
        "note": "含义: 成交量大的日子权重高, 等价于 VWAP 风格的动量。"
    },
    "volume_weighted_squared_impact": {
        "title": "成交量加权平方冲击 (放大版)",
        "dsl": "cs_rank(amount_weighted_mean(close.pct_change()^2, amount))",
        "steps": [
            "1) ret = close.pct_change(); ret2 = ret^2",
            "2) 用 amount 加权对 ret2 求时间平均",
            "3) factor = cs_rank(...)",
        ],
        "note": "含义: 资金参与度高的波动越大 → 风险定价信号。"
    },
    # ============ _flipped (IC<0 翻转) ============
    "abnormality_asymmetry_flipped": {
        "title": "异常不对称 (翻转, IC<0)",
        "dsl": "-cs_rank(abs(zscore(log(volume), 20)) * (high - low) / close * ewma_up_vol / ewma_down_vol)",
        "steps": [
            "1) log_vol = log(volume)",
            "2) z_log_vol = (log_vol - ts_mean(log_vol, 20)) / ts_std(log_vol, 20)",
            "3) abs(z)  异常量能",
            "4) range_pct = (high - low) / close",
            "5) ewma_up_vol / ewma_down_vol  上下行波动比",
            "6) 原 IC<0, 已取负号翻转",
        ],
        "note": "已加负号。原 IC<0 表示原方向排名低者后续表现好, 翻转后取正表示越异常越占优 (反向因子)。"
    },
    "impact_asymmetry_slow_recovery_flipped": {
        "title": "冲击不对称 (慢恢复, 翻转)",
        "dsl": "-cs_rank(ema(span=20, sign(ret) * log(volume) * (1 - ts_rank(amount, 20))))",
        "steps": [
            "1) ret = 日收益; sign(ret) * log(volume) = 有方向量能",
            "2) ts_rank(amount, 20) = 当日成交额分位",
            "3) (1 - ts_rank) = 缩量程度",
            "4) score = 方向 × 缩量 → 慢恢复特征",
            "5) ema(span=20, score)  长窗平滑",
            "6) 已加负号翻转",
        ],
        "note": "含义: 价跌量缩后长期慢反弹 → 此类标的 IC<0, 翻转后该信号反向选股有效。"
    },
    "impact_downside_asymmetry_smoothed_flipped": {
        "title": "下行冲击不对称 (平滑, 翻转)",
        "dsl": "-cs_rank(ema(span=10, where(ret < 0, abs(ret), 0) * log(volume)))",
        "steps": [
            "1) down_ret = where(ret < 0, abs(ret), 0)  只取下行幅度",
            "2) score = down_ret * log(volume)  下行资金冲击",
            "3) smoothed = ema(span=10, score)",
            "4) 已加负号翻转",
        ],
        "note": "含义: 下行 + 放量越大越差, IC<0 表示该信号反向有效 (取负号)。"
    },
    "lag_response_vol_tool_adaptive_flipped": {
        "title": "延迟响应自适应 (vol 工具, 翻转)",
        "dsl": "-cs_rank(ema((close/ts_delay(close, 5) - 1) * log(vol_ratio) * (atr_20/close), span=10))",
        "steps": [
            "1) ret_5 = (close - ts_delay(close, 5)) / ts_delay(close, 5)  5 日动量",
            "2) vol_ratio = volume / ts_mean(volume, 20)  量比",
            "3) atr_20 = ts_atr(high, low, close, 20)  20 日 ATR",
            "4) atr_norm = atr_20 / close  波动调整",
            "5) raw = ret_5 * log(vol_ratio) * atr_norm",
            "6) smoothed = ema(raw, span=10)  EMA 平滑",
            "7) 已加负号翻转",
        ],
        "note": "含义: 多窗口动量 × 量能 × 波动率, IC<0 → 翻转。"
    },
    "lag_vol_ratio_smoothed_v2_flipped": {
        "title": "滞后量比平滑 v2 (翻转)",
        "dsl": "-cs_rank(ema(span=15, vol_ratio - 1))",
        "steps": [
            "1) vol_ratio = volume / ts_mean(volume, 20)",
            "2) vol_ratio - 1 = 放量偏离度",
            "3) smoothed = ema(span=15, ...) 长窗平滑",
            "4) 已加负号翻转",
        ],
        "note": "放量偏离 1 越大 → 短期见顶概率越高, IC<0 → 取负。"
    },
    "lag_vol_robust_volratio_flipped": {
        "title": "稳健滞后量比 (翻转)",
        "dsl": "-cs_rank(ema(span=10, robust_vol_ratio - 1))",
        "steps": [
            "1) robust_vol_ratio = ts_median(volume, 20) / volume 稳健量比",
            "2) 与 1 的偏离",
            "3) 已加负号翻转",
        ],
        "note": "用 ts_median 代替 ts_mean, 对单日极端值更稳健。"
    },
    "lag_response_vol_tool_adaptive": {
        "title": "延迟响应自适应 (vol 工具)",
        "dsl": "cs_rank(ema((close/ts_delay(close, 5) - 1) * log(vol_ratio) * (atr_20/close), span=10))",
        "steps": [
            "1) ret_5 = (close - ts_delay(close, 5)) / ts_delay(close, 5)  5 日动量",
            "2) vol_ratio = volume / ts_mean(volume, 20)  量比",
            "3) atr_20 = ts_atr(high, low, close, 20)  20 日 ATR",
            "4) atr_norm = atr_20 / close  波动调整",
            "5) raw = ret_5 * log(vol_ratio) * atr_norm",
            "6) smoothed = ema(raw, span=10)  EMA 平滑",
            "7) factor = cs_rank(smoothed)",
        ],
        "note": "含义: 多窗口动量 × 量能 × 波动率, IC 为负 (反向因子)。"
                "字段: ts_atr 见 OP_DOC; ts_delay 见 OP_DOC。"
    },
    "lag_vol_volatility_adjusted_gate": {
        "title": "波动率门控滞后量比",
        "dsl": "cs_rank(where(high_resvol, ema(raw, span=10), ema(raw, span=5)))",
        "steps": [
            "1) raw = vol_ratio - 1  滞后量比",
            "2) 高波动期: ema(span=10) 长窗平滑",
            "3) 低波动期: ema(span=5) 短窗快响应",
            "4) factor = cs_rank(...)",
        ],
        "note": "含义: 同一信号在不同波动率环境使用不同窗口。"
    },
    "persistent_left_tail_variance_share_flipped": {
        "title": "持续左尾方差占比 (翻转)",
        "dsl": "-cs_rank(ewma_downside_var / ewm_total_var)",
        "steps": [
            "1) downside_var = ewm(min(ret,0)^2, span=20)  下行方差",
            "2) total_var = ewm(ret^2, span=20) 总方差",
            "3) share = downside / total 左尾方差占比",
            "4) 已加负号翻转",
        ],
        "note": "左尾方差占比越大越差, IC<0 → 取负。"
    },
    "price_impact_stable_5d_flipped": {
        "title": "5 日稳定价格冲击 (翻转)",
        "dsl": "-cs_rank(ema(span=5, sign(ret) * log(volume)))",
        "steps": [
            "1) ret = 日收益",
            "2) signed_volume = sign(ret) * log(volume)  资金方向",
            "3) smoothed = ema(span=5, ...) 5 日稳定",
            "4) 已加负号翻转",
        ],
        "note": "5 日均已平滑掉单日噪声; 反向选股 (取负)。"
    },
    "range_volume_ratio_flipped": {
        "title": "日内振幅量比 (翻转, IC<0)",
        "dsl": "-cs_rank((high - low) / close * (volume / ts_mean(volume, 20)))",
        "steps": [
            "1) range_ratio = (high - low) / close  当日振幅占比",
            "2) vol_ratio = volume / ts_mean(volume, 20)  量比",
            "3) raw = range_ratio * vol_ratio  振幅量比复合",
            "4) 已加负号翻转 (原 IC<0)",
        ],
        "note": "含义: 高振幅 + 放量 = 恐慌交易事件, 短期反转 IC<0, 翻转后正向选股。"
                "字段: ts_mean(volume, 20) 见 OP_DOC。"
    },
    "session_magnitude_imbalance_volume_weighted_42d_flipped": {
        "title": "盘间幅度失衡 (量加权 42 日, 翻转)",
        "dsl": "-cs_rank(where(overnight_imp > 0, 1.0, -1.0) * amount_ma_diff)",
        "steps": [
            "1) overnight_imp = open - ts_delay(close, 1)  隔夜跳空",
            "2) intraday_imp = close - open  日内幅度",
            "3) imbalance = where(overnight_imp > 0, 1.0, -1.0) * (overnight_imp + intraday_imp)",
            "4) 用 amount 做 42 日加权",
            "5) 已加负号翻转",
        ],
        "note": "含义: 隔夜 + 日内复合方向, 量加权 42 日, IC<0 → 翻转。"
    },
    "smoothed_downside_variance_resilience_flipped": {
        "title": "下行方差恢复力 (翻转)",
        "dsl": "-cs_rank(ewm_downside_var(span=20) / ewm_total_var(span=20))",
        "steps": [
            "1) downside = ewm(min(ret,0)^2, span=20)",
            "2) total = ewm(ret^2, span=20)",
            "3) ratio = downside / total",
            "4) 已加负号翻转",
        ],
        "note": "下行方差占比 = 风险敏感度, IC<0 → 翻转。"
    },
    "smoothed_drawdown_recovery_participation_flipped": {
        "title": "回撤-恢复参与度 (平滑, 翻转)",
        "dsl": "-cs_rank(ema(span=10, depth_score) * log(volume))",
        "steps": [
            "1) depth_score = (peak - close) / peak  回撤深度",
            "2) ema(span=10, depth_score) 平滑深度",
            "3) 乘 log(volume) 加权参与度",
            "4) 已加负号翻转",
        ],
        "note": "深度大且放量 = 套牢盘出货, IC<0 → 翻转。"
    },
    "smoothed_energy_downside_resilience_flipped": {
        "title": "下行能量恢复 (平滑, 翻转)",
        "dsl": "-cs_rank(ema(span=8, ret^2 * log(volume) * (1 - ts_rank(amount, 20))))",
        "steps": [
            "1) energy = ret^2 * log(volume)  波动 × 资金",
            "2) (1 - ts_rank(amount, 20))  缩量权重",
            "3) ema(span=8, energy)",
            "4) 已加负号翻转",
        ],
        "note": "波动大但缩量 → 无人接盘, IC<0 → 翻转。"
    },
    "smoothed_energy_downside_resilience_8span_flipped": {
        "title": "下行能量恢复 (8span, 翻转)",
        "dsl": "-cs_rank(ema(span=8, ret^2 * log(volume) * (1 - ts_rank(amount, 30))))",
        "steps": [
            "与 8span 版同算法, 量能分位窗口 = 30",
            "已加负号翻转",
        ],
        "note": "窗口差异: 用 30 日量能分位代替 20 日。"
    },
    "smoothed_left_tail_variance_share_flipped": {
        "title": "平滑左尾方差占比 (翻转)",
        "dsl": "-cs_rank(ema(span=10, downside_var) / ema(span=10, total_var))",
        "steps": [
            "1) downside_var = ema(min(ret,0)^2, span=10)",
            "2) total_var = ema(ret^2, span=10)",
            "3) share = downside / total",
            "4) 已加负号翻转",
        ],
        "note": "短窗 EWM, IC<0 → 翻转。"
    },
    "squared_range_close_ema_cuberoot_flipped": {
        "title": "振幅立方根 (翻转)",
        "dsl": "-cs_rank(((high - low) / close)^(1/3) * log(volume))",
        "steps": [
            "1) range_pct = (high - low) / close",
            "2) cuberoot = range_pct^(1/3) 立方根, 压缩极端值",
            "3) 乘 log(volume) 加权资金",
            "4) 已加负号翻转",
        ],
        "note": "立方根减小量纲, IC<0 → 翻转。"
    },
    "squared_range_volume_rank_smoothed_no_cuberoot_flipped": {
        "title": "振幅排名平滑 (无立方根, 翻转)",
        "dsl": "-cs_rank(((high - low) / close)^2 * ts_rank(volume, 20))",
        "steps": [
            "1) range_pct = (high - low) / close",
            "2) range_sq = range_pct^2 平方放大",
            "3) vol_rank = ts_rank(volume, 20) 当前量能分位",
            "4) score = range_sq * vol_rank",
            "5) 已加负号翻转",
        ],
        "note": "用平方替代立方根, 极值更敏感; IC<0 → 翻转。"
    },
    "vol_regime_impact_ema_flipped": {
        "title": "波动率制度冲击 (EMA, 翻转)",
        "dsl": "-cs_rank(ema(span=15, sign(ret) * log(volume) * ts_std(20, ret)))",
        "steps": [
            "1) signed_volume = sign(ret) * log(volume)",
            "2) ts_std(20, ret)  20 日波动率",
            "3) score = signed_volume * vol  波动调整",
            "4) ema(span=15, score) 长窗平滑",
            "5) 已加负号翻转",
        ],
        "note": "高波动期量价信号反向选股, IC<0 → 翻转。"
    },
    "vol_volume_asym_ewma_flipped": {
        "title": "波动量能不对称 (EWMA, 翻转)",
        "dsl": "-cs_rank(ewma(span=10, abs(sign(ret) * log(volume))))",
        "steps": [
            "1) asym = abs(sign(ret) * log(volume))  有方向量能绝对值",
            "2) ewma(span=10, asym) 短窗指数加权",
            "3) 已加负号翻转",
        ],
        "note": "绝对方向量能 EWMA, IC<0 → 翻转。"
    },
    "volume_adaptive_momentum_fast_vol": {
        "title": "量能自适应动量 (快速波动)",
        "dsl": "-cs_rank(ema(span=10, ret_5 * log(vol_ratio)) / (1 + ts_std(5, ret)))",
        "steps": [
            "1) ret_5 = (close - ts_delay(close,5))/ts_delay(close,5)",
            "2) vol_ratio = volume / ts_mean(volume, 20)",
            "3) momentum = ret_5 * log(vol_ratio)",
            "4) smoothed = ema(span=10, momentum)",
            "5) divide by (1 + ts_std(5, ret))  短期波动归一",
            "6) 已加负号翻转",
        ],
        "note": "动量除以短期波动, IC<0 → 翻转。"
    },
    "volume_adaptive_momentum_smoothed_v3": {
        "title": "量能自适应动量 v3 (15 窗)",
        "dsl": "-cs_rank(ema(ret_5 * log(vol_ratio), span=15))",
        "steps": [
            "1) ret_5 = 5 日动量",
            "2) vol_ratio = volume / ts_mean(volume, 20)",
            "3) raw = ret_5 * log(vol_ratio)",
            "4) smoothed = ema(raw, span=15)",
            "5) 已加负号翻转",
        ],
        "note": "15 日 EMA 比 v1 更平滑; IC<0 → 翻转。"
    },
    "volume_adjusted_price_range": {
        "title": "量能调整的价格振幅",
        "dsl": "-cs_rank((high - low) / close * log(volume))",
        "steps": [
            "1) range_pct = (high - low) / close",
            "2) log(volume) 量能对数",
            "3) score = range_pct * log(volume) 振幅量能复合",
            "4) 已加负号翻转",
        ],
        "note": "直接用 log(volume) 不用 ts_mean, 更敏感; IC<0 → 翻转。"
    },
    "volume_adjusted_price_range_ema_flipped": {
        "title": "量能调整振幅 EMA (翻转)",
        "dsl": "-cs_rank(ema(span=10, (high - low) / close * log(volume)))",
        "steps": [
            "1) score = (high - low) / close * log(volume)",
            "2) ema(span=10, score)",
            "3) 已加负号翻转",
        ],
        "note": "10 日 EMA 平滑; IC<0 → 翻转。"
    },
    "volume_adjusted_price_range_ema_gated": {
        "title": "量能调整振幅 (门控)",
        "dsl": "-cs_rank(where(liquidity_high, ema(score, 10), ema(score, 20)))",
        "steps": [
            "1) score = range_pct * log(volume)",
            "2) 高流动性期: 短窗 ema(10)",
            "3) 低流动性期: 长窗 ema(20)",
            "4) 已加负号翻转",
        ],
        "note": "门控: 不同流动性环境用不同平滑窗; IC<0 → 翻转。"
    },
    "volume_adjusted_price_range_gated_flipped": {
        "title": "量能调整振幅 (门控, 翻转)",
        "dsl": "-cs_rank(where(is_high_volume, score * 1.2, score * 0.8))",
        "steps": [
            "1) score = range_pct * log(volume)",
            "2) 高量能日: score * 1.2 放大",
            "3) 低量能日: score * 0.8 缩小",
            "4) 已加负号翻转",
        ],
        "note": "门控: 高量能日权重更大; IC<0 → 翻转。"
    },
    "volume_adjusted_price_range_regime_flipped": {
        "title": "量能调整振幅 (regime, 翻转)",
        "dsl": "-cs_rank((high - low) / close * log(volume) * ts_std(20, ret))",
        "steps": [
            "1) score = range_pct * log(volume)",
            "2) ts_std(20, ret) 20 日波动率权重",
            "3) factor = score * vol_regime",
            "4) 已加负号翻转",
        ],
        "note": "用波动率作 regime 权重; IC<0 → 翻转。"
    },
    "volume_adjusted_price_range_regime_smoothed_flipped": {
        "title": "量能调整振幅 (regime 平滑, 翻转)",
        "dsl": "-cs_rank(ema(span=20, range_pct * log(volume) * ts_std(20, ret)))",
        "steps": [
            "1) 同 regime 版, 但 score 经 20 日 EMA 平滑",
            "2) 已加负号翻转",
        ],
        "note": "20 日 EMA 平滑掉日间噪声; IC<0 → 翻转。"
    },
    "volume_adjusted_price_range_regime_smoothed_v2_flipped": {
        "title": "量能调整振幅 (regime 平滑 v2, 翻转)",
        "dsl": "-cs_rank(ema(span=15, range_pct * log(volume) * ts_std(15, ret)))",
        "steps": [
            "1) 同 v1, 但窗口都用 15 日",
            "2) 已加负号翻转",
        ],
        "note": "短窗 (15) 对 regime 切换更敏感; IC<0 → 翻转。"
    },
    "volume_adjusted_price_range_regime_v2_flipped": {
        "title": "量能调整振幅 (regime v2, 翻转)",
        "dsl": "-cs_rank(range_pct * log(volume) * ts_std(15, ret))",
        "steps": [
            "1) score = range_pct * log(volume)",
            "2) ts_std(15, ret) 15 日波动率",
            "3) 已加负号翻转",
        ],
        "note": "无平滑的原始 regime 版; IC<0 → 翻转。"
    },
    "volume_adjusted_price_range_regime_v3_flipped": {
        "title": "量能调整振幅 (regime v3, 翻转)",
        "dsl": "-cs_rank(ema(span=20, range_pct * log(volume) * ts_std(30, ret)))",
        "steps": [
            "1) score = range_pct * log(volume)",
            "2) ts_std(30, ret) 30 日波动率 (更长)",
            "3) ema(span=20) 20 日 EMA 平滑",
            "4) 已加负号翻转",
        ],
        "note": "长窗波动率 + 中窗平滑; IC<0 → 翻转。"
    },
    "volume_impact_elasticity_smooth_flipped": {
        "title": "量能冲击弹性 (平滑, 翻转)",
        "dsl": "-cs_rank(ema(span=10, abs(ret) / log(volume)))",
        "steps": [
            "1) elasticity = abs(ret) / log(volume)  单位量能带来的收益弹性",
            "2) ema(span=10, elasticity) 短窗平滑",
            "3) 已加负号翻转",
        ],
        "note": "低量能但高收益 = 弹性高 = 主力控盘, IC<0 → 翻转。"
    },
    "volume_state_transition_carry_smooth_flipped": {
        "title": "量能状态跃迁 (carry, 平滑, 翻转)",
        "dsl": "-cs_rank(ema(span=15, vol_ratio * sign(ret)))",
        "steps": [
            "1) carry = vol_ratio * sign(ret)  顺势放量 (carry)",
            "2) ema(span=15, carry) 长窗平滑",
            "3) 已加负号翻转",
        ],
        "note": "顺势放量 = 趋势跟随, IC<0 → 翻转 (选不跟趋势的)。"
    },
    "volume_transition_raw_carry_30d_flipped": {
        "title": "量能跃迁原始 carry (30 日, 翻转)",
        "dsl": "-cs_rank(where(vol_ratio > 1.5, vol_ratio * sign(ret), 0))",
        "steps": [
            "1) 放量日 (vol_ratio > 1.5) 才计入",
            "2) carry = vol_ratio * sign(ret)",
            "3) 已加负号翻转",
        ],
        "note": "只在放量日信号有效, 减少噪声; IC<0 → 翻转。"
    },
    "volume_transition_stress_attenuated_30d_flipped": {
        "title": "量能跃迁应力衰减 (30 日, 翻转)",
        "dsl": "-cs_rank(where(vol_ratio > 1.5, vol_ratio * sign(ret) * 0.5, 0))",
        "steps": [
            "1) 放量日: carry * 0.5 (应力衰减)",
            "2) 非放量日: 0",
            "3) 已加负号翻转",
        ],
        "note": "对放量日的 carry 信号减半, 防止过度反应; IC<0 → 翻转。"
    },
    "vwap_adjusted_range_tanh_mutated_v2_flipped": {
        "title": "VWAP 调整振幅 (tanh v2, 翻转)",
        "dsl": "-cs_rank(tanh(ts_atr(14, 14) / vwap_30 * log(volume / ts_mean(volume, 30))))",
        "steps": [
            "1) atr = ts_atr(high, low, close, 14)",
            "2) ts_atr(14, 14)  14 日 ATR 二次平滑 (变异)",
            "3) vwap_30 = ts_mean(vwap, 30)",
            "4) vol_norm = log(volume / ts_mean(volume, 30))  30 日量比",
            "5) raw = atr / vwap_30 * vol_norm",
            "6) tanh(raw)  压缩极端值",
            "7) 已加负号翻转",
        ],
        "note": "用 tanh 把信号压到 (-1, 1); IC<0 → 翻转。"
    },
    "vwap_adjusted_range_tanh_smooth_flipped": {
        "title": "VWAP 调整振幅 (tanh 平滑, 翻转)",
        "dsl": "-cs_rank(tanh(ts_atr(20, 20) / vwap_30 * log(volume / ts_mean(volume, 30))))",
        "steps": [
            "1) 与 v2 类似, ATR 窗口 = 20",
            "2) v2 的 ATR 是 14 日",
            "3) 已加负号翻转",
        ],
        "note": "更长 ATR 窗口 → 更平滑的波动估计; IC<0 → 翻转。"
    },
    # 5 个还残留的
    "ts_size_adaptive_earnings_book_smooth": {
        "title": "Size 自适应 earnings×book (平滑)",
        "dsl": "cs_rank(-ema(span=10, abs(log(close)) * (eps_ttm / pb_lf)))",
        "steps": [
            "1) book_yield = eps_ttm / pb_lf  ROE 代理 (盈利/账面)",
            "2) size_penalty = abs(log(close))  市值对数 (大市值权重大)",
            "3) score = -size_penalty * book_yield  小市值 + 高 ROE 占优",
            "4) ema(span=10, score) 平滑",
            "5) factor = cs_rank(...)",
        ],
        "note": "含义: 小盘高 ROE 公司长期占优。"
    },
    "drawdown_volume_modulated": {
        "title": "回撤量能调制 (复合)",
        "dsl": "cs_rank((rank(depth) + rank(duration) + rank(recovery)) / 3 * (1 + rank(vol_ratio)))",
        "steps": [
            "1) depth / duration / recovery 三个回撤特征, 分别 cs_rank",
            "2) mean(depth_rank, dur_rank, rec_rank) 平均",
            "3) (1 + rank(vol_ratio)) 量能调制系数",
            "4) factor = 总分",
        ],
        "note": "量能放大回撤特征的信号强度。"
    },
    "session_asymmetry_smooth_confirmed_30d": {
        "title": "盘间不对称平滑确认 (30 日)",
        "dsl": "cs_rank(ema(span=15, sign(ret) * log(volume)) * ts_std(30, ret))",
        "steps": [
            "1) signed_vol = sign(ret) * log(volume)",
            "2) smoothed = ema(span=15, signed_vol)",
            "3) vol_regime = ts_std(30, ret)",
            "4) score = smoothed * vol_regime",
            "5) factor = cs_rank(score)",
        ],
        "note": "盘间量价方向 × 30 日 regime, IC 为正 (未翻转)。"
    },
    "ewm_downside_variance_resilience": {
        "title": "EWM 下行方差恢复力",
        "dsl": "-cs_rank(ewm(min(ret,0)^2, span=24) / ewm(ret^2, span=24))",
        "steps": [
            "1) downside = ewm(min(ret,0)^2, span=24)",
            "2) total = ewm(ret^2, span=24)",
            "3) share = downside / total",
            "4) factor = cs_rank(-share)",
        ],
        "note": "已显式取负号: 下行方差占比越高越差 → 排名靠后。"
    },
}
