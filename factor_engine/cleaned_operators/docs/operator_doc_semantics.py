# -*- coding: utf-8 -*-
"""算子文档语义与 LaTeX 公式（供 generate_operators_guide 使用）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class OpDoc:
    meaning: str
    compute: str
    latex: str


# ---------------------------------------------------------------------------
# 显式条目：复杂或易歧义算子
# ---------------------------------------------------------------------------
_EXPLICIT: dict[str, OpDoc] = {
    # --- 时序扩展 ---
    "ts_pct": OpDoc(
        "相对 d 期前的收益率（百分比变化）。",
        "对每个标的独立计算：当前值除以 d 期前的值再减 1；d=1 时等价于 returns/ pct_change。",
        r"x_{i,t} = \frac{X_{i,t}}{X_{i,t-d}} - 1",
    ),
    "ts_quantile": OpDoc(
        "滚动窗口内的分位数。",
        "在长度为 d 的窗口内，对 {X_{t-d+1},…,X_t} 取 q 分位点（线性插值，与 pandas quantile 一致）。",
        r"Q_q\bigl(\{X_{i,t-k}\}_{k=0}^{d-1}\bigr)",
    ),
    "ts_argmax": OpDoc(
        "窗口内最大值距当前 bar 的偏移（0 表示最新 bar 为最大值）。",
        "在窗口 {t-d+1,…,t} 内找 argmax，返回 t - t^*（t^* 为最大值所在时刻）。",
        r"\arg\max_{0\le k < d} X_{i,t-k}\;\;\text{（返回偏移量，0=最新）}",
    ),
    "ts_argmin": OpDoc(
        "窗口内最小值距当前 bar 的偏移（0 表示最新 bar 为最小值）。",
        "在窗口 {t-d+1,…,t} 内找 argmin，返回 t - t^*。",
        r"\arg\min_{0\le k < d} X_{i,t-k}\;\;\text{（返回偏移量，0=最新）}",
    ),
    "ts_topk_sum": OpDoc(
        "滚动窗口内最大的 k 个值的和。",
        "对每个窗口，将 d 个观测降序排列，取前 k 个求和。",
        r"\sum_{j=1}^{k} X_{i,t}^{(j)},\quad X_{i,t}^{(j)}\text{ 为窗口内第 }j\text{ 大值}",
    ),
    "ts_rank": OpDoc(
        "窗口内当前值的百分位排名。",
        "在长度为 d 的窗口内，将 X_t 与窗口内所有值比较，输出 [0,1] 的秩分位。",
        r"\frac{\#\{k: X_{i,t-k} \le X_{i,t}\} - 1}{d - 1}",
    ),
    "ts_zscore": OpDoc(
        "滚动 Z 分数（窗口内标准化）。",
        "在长度为 d 的窗口内计算均值 μ 与标准差 σ，再对当前值标准化。",
        r"Z_{i,t} = \frac{X_{i,t} - \mu_{i,t}^{(d)}}{\sigma_{i,t}^{(d)}}",
    ),
    "ts_moment": OpDoc(
        "滚动 k 阶中心矩。",
        "在窗口内先算均值 μ，再算 E[(X-μ)^k]。",
        r"M_k = \frac{1}{d}\sum_{j=0}^{d-1}\bigl(X_{i,t-j}-\mu\bigr)^k",
    ),
    "price_spread_deviation": OpDoc(
        "相对窗口均值的偏离率。",
        "当前值除以 d 期滚动均值再减 1。",
        r"\frac{X_{i,t}}{\bar{X}_{i,t}^{(d)}} - 1",
    ),
    "rank_corr": OpDoc(
        "秩相关系数（Spearman 型）。",
        "d=0：截面内对 rank(X)、rank(Y) 求 Pearson 相关；d>0：窗口内分别做秩变换后求相关。",
        r"\rho_s = \mathrm{corr}\bigl(\mathrm{rank}(X),\mathrm{rank}(Y)\bigr)",
    ),
    "ts_poly2_coeff": OpDoc(
        "窗口内对时间索引做二次拟合的二次项系数。",
        "在窗口内拟合 X ≈ a + bt + ct²，返回 c。",
        r"X_{i,t-j} \approx a + b\tau + c\tau^2,\quad \tau=0,\ldots,d-1",
    ),
    "ts_poly2_resid": OpDoc(
        "窗口内 Y 对 X 二次拟合的残差（最新 bar）。",
        "拟合 Y ≈ a + bX + cX²，输出最新观测的拟合残差。",
        r"\varepsilon_{i,t} = Y_{i,t} - \hat{Y}_{i,t},\quad \hat{Y}=a+bX+cX^2",
    ),
    "digital_count": OpDoc(
        "连续低波动 bar 的计数（满足阈值且连续长度 ≥ run）。",
        "若 |X_t/X_{t-1}-1| ≤ threshold 则计数 +1，否则归零；仅保留 ≥ run 的片段。",
        r"C_t = \begin{cases}C_{t-1}+1 & \left|\frac{X_t}{X_{t-1}}-1\right|\le\theta\\ 0 & \text{否则}\end{cases}",
    ),
    "ts_max_buildup": OpDoc(
        "窗口内持续创新高的次数。",
        "统计最近 d 期内 X 创历史新高的次数。",
        r"\sum_{s=t-d+1}^{t} \mathbf{1}\{X_{i,s} = \max_{u\le s} X_{i,u}\}",
    ),
    # --- 截面 ---
    "rank": OpDoc(
        "截面百分位排名。",
        "在同一交易日 t，对所有标的的 X_{i,t} 做秩变换，映射到 [0,1]。",
        r"R_{i,t} = \frac{\mathrm{rank}(X_{i,t}) - 1}{N_t - 1}",
    ),
    "zscore": OpDoc(
        "截面 Z 分数。",
        "每个交易日对全市场去均值、除以截面标准差。",
        r"Z_{i,t} = \frac{X_{i,t} - \bar{X}_{\cdot,t}}{\sigma_{\cdot,t}}",
    ),
    "normalize": OpDoc(
        "截面归一化（去均值）。",
        "每个交易日 X_{i,t} 减去截面均值。",
        r"\tilde{X}_{i,t} = X_{i,t} - \bar{X}_{\cdot,t}",
    ),
    "quantile": OpDoc(
        "截面分位数映射（高斯分位变换）。",
        "先 rank 到 (0,1)，再经 scipy 逆正态 CDF 映射到标准正态分位。",
        r"Q_{i,t} = \Phi^{-1}\!\bigl(\mathrm{rank}(X_{i,t})\bigr)",
    ),
    "scale": OpDoc(
        "截面缩放至单位 L1 范数。",
        "X / sum(|X|)（截面内）。",
        r"\tilde{X}_{i,t} = \frac{X_{i,t}}{\sum_j |X_{j,t}|}",
    ),
    "winsorize": OpDoc(
        "截面缩尾。",
        "将截面分布两端超出 [p, 1-p] 分位的值截断到分位点。",
        r"\tilde{X}_{i,t} = \mathrm{clip}\bigl(X_{i,t}, Q_p, Q_{1-p}\bigr)",
    ),
    "neutralize": OpDoc(
        "截面 OLS 中性化（对因子 Y 回归取残差）。",
        "每个交易日回归 X ~ Y，输出残差 ε。",
        r"X_{i,t} = \alpha_t + \beta_t Y_{i,t} + \varepsilon_{i,t}",
    ),
    "cs_resid": OpDoc(
        "截面线性回归残差。",
        "每个交易日 OLS：Y ~ X，输出 ε = Y - (α + βX)。",
        r"\varepsilon_{i,t} = Y_{i,t} - (\alpha_t + \beta_t X_{i,t})",
    ),
    "cs_regression": OpDoc(
        "截面回归（mode 控制输出：0=残差，1=β，2=拟合值）。",
        "每个交易日对 (X,Y) 做 OLS，按 mode 返回残差、斜率或拟合值。",
        r"Y_{i,t} = \alpha_t + \beta_t X_{i,t} + \varepsilon_{i,t}",
    ),
    "cs_demean": OpDoc(
        "截面去均值。",
        "每个交易日减去截面算术平均。",
        r"\tilde{X}_{i,t} = X_{i,t} - \frac{1}{N_t}\sum_j X_{j,t}",
    ),
    # --- 分组 ---
    "group_rank": OpDoc(
        "组内截面排名。",
        "每个 (t, group) 内对 X 做 [0,1] 秩分位。",
        r"R_{i,t} = \mathrm{rank}_{g(i)}(X_{i,t})",
    ),
    "group_neutralize": OpDoc(
        "组内去均值（行业中性）。",
        "每个 (t, group) 内 X - 组内均值。",
        r"\tilde{X}_{i,t} = X_{i,t} - \bar{X}_{g(i),t}",
    ),
    "group_zscore": OpDoc(
        "组内 Z 分数。",
        "每个 (t, group) 内标准化。",
        r"Z_{i,t} = \frac{X_{i,t} - \mu_{g(i),t}}{\sigma_{g(i),t}}",
    ),
    "group_mean": OpDoc(
        "组内均值（广播回原行）。",
        "每个 (t, group) 内算术平均，赋给组内每个标的。",
        r"\bar{X}_{g,t} = \frac{1}{|G|}\sum_{j\in G} X_{j,t}",
    ),
    "group_std": OpDoc(
        "组内标准差（广播）。",
        "每个 (t, group) 内样本标准差。",
        r"\sigma_{g,t} = \mathrm{std}_{j\in G}(X_{j,t})",
    ),
    "group_neutralize": OpDoc(
        "组内去均值（行业/分组中性化）。",
        "每个 (t, group) 内 X - 组内均值；别名 group_demean、neutralize。",
        r"\tilde{X}_{i,t} = X_{i,t} - \bar{X}_{g(i),t}",
    ),
    "group_demean": OpDoc(
        "组内去均值（同 group_neutralize）。",
        "每个 (t, group) 内 X - 组内均值。",
        r"\tilde{X}_{i,t} = X_{i,t} - \bar{X}_{g(i),t}",
    ),
    "industry_neutralize": OpDoc(
        "行业中性化（组内 demean）。",
        "按行业分组，减去组内截面均值。",
        r"\tilde{X}_{i,t} = X_{i,t} - \bar{X}_{\mathrm{ind}(i),t}",
    ),
    "size_neutralize": OpDoc(
        "市值中性化（对 log(cap) 截面回归残差）。",
        "每个交易日 Y 对 log(市值) 回归，取残差。",
        r"Y_{i,t} = \alpha_t + \beta_t \ln(\mathrm{cap}_{i,t}) + \varepsilon_{i,t}",
    ),
    # --- 元素级 ---
    "clip": OpDoc(
        "截断到区间 [lo, hi]；别名 cap、clamp。",
        "逐元素 clip，超出上下界的值截断。",
        r"\tilde{x} = \min\bigl(\max(x, lo), hi\bigr)",
    ),
    "cap": OpDoc(
        "截断到区间 [lo, hi]；别名 clip、clamp。",
        "逐元素 clip，超出上下界的值截断。",
        r"\tilde{x} = \min\bigl(\max(x, lo), hi\bigr)",
    ),
    "coalesce": OpDoc(
        "返回第一个非 NaN 值。",
        "从左到右扫描参数，取首个有限值。",
        r"\mathrm{coalesce}(a,b,\ldots) = \min\{v: v \text{ 非 NaN}\}",
    ),
    "signed_sqrt": OpDoc(
        "保号开方。",
        "sign(x)·√|x|。",
        r"\mathrm{sgn}(x)\sqrt{|x|}",
    ),
    "sigmoid": OpDoc(
        "Sigmoid 映射。",
        "1 / (1 + e^{-x})。",
        r"\sigma(x) = \frac{1}{1 + e^{-x}}",
    ),
    "protected_div": OpDoc(
        "安全除法。",
        "分母接近 0 或 NaN 时返回 NaN 或默认值，避免 inf。",
        r"\begin{cases}a/b & |b|>\epsilon\\ \mathrm{NaN} & \text{否则}\end{cases}",
    ),
    "protected_log": OpDoc(
        "安全对数。",
        "对非正数返回 NaN。",
        r"\ln(x)\;\text{若}\;x>0,\;\text{否则 NaN}",
    ),
    "protected_sqrt": OpDoc(
        "安全开方。",
        "对负数返回 NaN。",
        r"\sqrt{x}\;\text{若}\;x\ge 0",
    ),
    # --- 价量 ---
    "returns": OpDoc(
        "简单收益率（别名，canonical 为 ts_pct(x, 1)）。",
        "close/Ref(close,1) - 1，即相邻 bar 百分比变化。",
        r"r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1",
    ),
    "ts_ratio": OpDoc(
        "价格比率（相邻 bar）。",
        "P_t / P_{t-1}，即 1 + 简单收益率。",
        r"R_{i,t} = \frac{P_{i,t}}{P_{i,t-1}}",
    ),
    "ts_ema": OpDoc(
        "指数移动平均（EMA）。",
        "对序列做指数加权平滑，近期权重更大；span 参数以实现为准。",
        r"\mathrm{EMA}_t = \alpha X_t + (1-\alpha)\mathrm{EMA}_{t-1}",
    ),
    "log_returns": OpDoc(
        "对数收益率。",
        "ln(P_t / P_{t-1})。",
        r"r_{i,t} = \ln\frac{P_{i,t}}{P_{i,t-1}}",
    ),
    "volatility": OpDoc(
        "滚动年化波动率。",
        "窗口内收益率标准差 × √252（默认年化因子以实现为准）。",
        r"\sigma_{\mathrm{ann}} = \sqrt{252}\cdot \mathrm{std}(r_{t-d+1:t})",
    ),
    "sharpe_ratio": OpDoc(
        "滚动年化夏普比率。",
        "窗口内 mean(r)/std(r) × √252。",
        r"S = \sqrt{252}\,\frac{\bar{r}}{\sigma_r}",
    ),
    "max_drawdown": OpDoc(
        "最大回撤。",
        "窗口内从峰值到谷底的最大跌幅。",
        r"\mathrm{MDD} = \max_{\tau\in[t-d,t]}\frac{\mathrm{peak}_\tau - P_\tau}{\mathrm{peak}_\tau}",
    ),
    "cumulative_returns": OpDoc(
        "累计收益率。",
        "价格序列相对起点的总收益 (1+r) 连乘 - 1 或等价实现。",
        r"R_t^{\mathrm{cum}} = \prod_{s\le t}(1+r_s) - 1",
    ),
    "rolling_beta_to_market": OpDoc(
        "滚动市场 Beta。",
        "窗口内 Cov(r_i, r_m) / Var(r_m)。",
        r"\beta_{i,t} = \frac{\mathrm{Cov}(r_i,r_m)}{\mathrm{Var}(r_m)}",
    ),
    "downside_beta": OpDoc(
        "下行 Beta。",
        "仅在 r_m < 0 的子样本上估计 Beta。",
        r"\beta^- = \frac{\mathrm{Cov}(r_i,r_m\mid r_m<0)}{\mathrm{Var}(r_m\mid r_m<0)}",
    ),
    "tail_beta": OpDoc(
        "尾部 Beta。",
        "仅在 r_m 处于窗口内最低 q 分位（如 5%）的样本上估计 Beta。",
        r"\beta_{\mathrm{tail}} = \frac{\mathrm{Cov}(r_i,r_m\mid r_m\le Q_q)}{\mathrm{Var}(r_m\mid r_m\le Q_q)}",
    ),
    "residual_momentum_capm": OpDoc(
        "CAPM 残差动量。",
        "窗口内 r_i = α + β r_m + ε，输出 Σε。",
        r"\sum_{s=t-d+1}^{t}\hat\varepsilon_{i,s},\quad r_i=\alpha+\beta r_m+\varepsilon",
    ),
    "coskewness_to_market": OpDoc(
        "相对市场的协偏度。",
        "E[(r_i-μ_i)(r_m-μ_m)²] / (σ_i σ_m²)。",
        r"\frac{E\bigl[(r_i-\mu_i)(r_m-\mu_m)^2\bigr]}{\sigma_i\,\sigma_m^2}",
    ),
    "idio_vol": OpDoc(
        "特质波动率（CAPM 残差标准差）。",
        "窗口内 CAPM 回归残差的标准差。",
        r"\sigma_\varepsilon = \mathrm{std}(\hat\varepsilon)",
    ),
    "idio_skew": OpDoc(
        "特质偏度（CAPM 残差偏度）。",
        "窗口内 CAPM 残差的三阶标准化矩。",
        r"\mathrm{skew}(\hat\varepsilon)",
    ),
    # --- 基本面 ---
    "ttm": OpDoc(
        "滚动十二个月（TTM）累加。",
        "当前及前 3 个报告期累加（季频 4 期）。",
        r"\mathrm{TTM}_t = \sum_{k=0}^{3} X_{t-k}",
    ),
    "quarter": OpDoc(
        "累计值转单季度。",
        "X_t - X_{t-1}；首期为当期值。",
        r"Q_t = X_t - X_{t-1}",
    ),
    "yoy": OpDoc(
        "同比增速。",
        "X_t / X_{t-4} - 1（季频同比）。",
        r"Y_t = \frac{X_t}{X_{t-4}} - 1",
    ),
    "avg2": OpDoc(
        "当期与上期均值。",
        "(X_t + X_{t-1}) / 2。",
        r"\bar{X}_t = \frac{X_t + X_{t-1}}{2}",
    ),
    # --- 微观 ---
    "real_turnover_rate": OpDoc(
        "真实流通盘换手率。",
        "成交量除以有效流通股本。",
        r"\mathrm{TO}_t = \frac{\mathrm{volume}_t}{\mathrm{float}_t}",
    ),
    # --- 信号 ---
    "trade_when": OpDoc(
        "条件持仓信号。",
        "trigger 为真时输出 alpha，否则输出 exit_（或 hold 逻辑以实现为准）。",
        r"\alpha_t^{\mathrm{out}} = \begin{cases}\alpha_t & \mathrm{trigger}_t\\ \mathrm{exit}_t & \text{否则}\end{cases}",
    ),
    "if_else": OpDoc(
        "逐元素条件选择。",
        "condition 为真取 if_true，否则 if_false。",
        r"\begin{cases}a & c\\ b & \neg c\end{cases}",
    ),
    "hump_decay": OpDoc(
        "阈值衰减（抑制微小变化）。",
        "变化幅度小于 hump 时不更新，否则按规则衰减（见实现）。",
        r"\Delta x_t \leftarrow \begin{cases}0 & |\Delta|<h\\ \Delta & \text{否则}\end{cases}",
    ),
    # --- 技术指标（常用） ---
    "RSI": OpDoc(
        "相对强弱指数 RSI。",
        "窗口内上涨均值 / (上涨+下跌均值) 映射到 0–100。",
        r"\mathrm{RSI} = 100 - \frac{100}{1 + RS},\quad RS=\frac{\mathrm{EMA}(\Delta^+)}{\mathrm{EMA}(|\Delta^-|)}",
    ),
    "MACD": OpDoc(
        "MACD 柱（DIF - DEA）。",
        "快线 EMA - 慢线 EMA，再减信号线 EMA。",
        r"\mathrm{MACD} = \mathrm{EMA}_{f}(P) - \mathrm{EMA}_{s}(P) - \mathrm{EMA}_{sig}(\mathrm{DIF})",
    ),
    "MACD_line": OpDoc(
        "MACD DIF 线。",
        "快 EMA 减慢 EMA。",
        r"\mathrm{DIF} = \mathrm{EMA}_{12}(P) - \mathrm{EMA}_{26}(P)",
    ),
    "MACD_signal": OpDoc(
        "MACD 信号线 DEA。",
        "对 DIF 做 EMA。",
        r"\mathrm{DEA} = \mathrm{EMA}_9(\mathrm{DIF})",
    ),
    "MACD_hist": OpDoc(
        "MACD 柱状图。",
        "DIF - DEA。",
        r"\mathrm{HIST} = \mathrm{DIF} - \mathrm{DEA}",
    ),
    "SMA": OpDoc(
        "简单移动平均。",
        "窗口内算术平均。",
        r"\mathrm{SMA}_t = \frac{1}{d}\sum_{k=0}^{d-1} P_{t-k}",
    ),
    "WMA": OpDoc(
        "加权移动平均。",
        "线性递增权重，近期权重更大。",
        r"\mathrm{WMA}_t = \frac{\sum_{k=0}^{d-1}(d-k)P_{t-k}}{\sum_{k=1}^{d}k}",
    ),
    "EMA": OpDoc(
        "指数移动平均。",
        "EMA_t = α P_t + (1-α) EMA_{t-1}，α=2/(d+1)。",
        r"\mathrm{EMA}_t = \alpha P_t + (1-\alpha)\mathrm{EMA}_{t-1},\;\alpha=\frac{2}{d+1}",
    ),
    "ATR": OpDoc(
        "平均真实波幅 ATR。",
        "TR = max(H-L, |H-C_prev|, |L-C_prev|)，再 rolling mean。",
        r"\mathrm{ATR}_t = \mathrm{mean}(\mathrm{TR}_{t-d+1:t})",
    ),
    "ADX": OpDoc(
        "平均趋向指数 ADX。",
        "基于 +DI/-DI 与 DX 的平滑（TA-Lib 标准定义）。",
        r"\mathrm{ADX} = \mathrm{EMA}(\mathrm{DX}),\quad \mathrm{DX}=\frac{|\mathrm{DI}^+-\mathrm{DI}^-|}{\mathrm{DI}^++\mathrm{DI}^-}\times 100",
    ),
    "BollingerBands": OpDoc(
        "布林带中轨。",
        "SMA(P, d)。",
        r"\mathrm{BB}_{\mathrm{mid}} = \mathrm{SMA}(P,d)",
    ),
    "BollingerUpper": OpDoc(
        "布林带上轨。",
        "SMA + k·σ。",
        r"\mathrm{BB}_{\mathrm{up}} = \mathrm{SMA} + k\sigma",
    ),
    "BollingerLower": OpDoc(
        "布林带下轨。",
        "SMA - k·σ。",
        r"\mathrm{BB}_{\mathrm{low}} = \mathrm{SMA} - k\sigma",
    ),
    "StochasticK": OpDoc(
        "随机指标 %K。",
        "(C - L_d) / (H_d - L_d) × 100。",
        r"\%K = 100\cdot\frac{C_t - L_d}{H_d - L_d}",
    ),
    "StochasticD": OpDoc(
        "随机指标 %D。",
        "%K 的 SMA。",
        r"\%D = \mathrm{SMA}(\%K)",
    ),
    "WilliamsR": OpDoc(
        "威廉指标 %R。",
        "(H_d - C) / (H_d - L_d) × (-100)。",
        r"\%R = -100\cdot\frac{H_d - C_t}{H_d - L_d}",
    ),
    "CCI": OpDoc(
        "商品通道指数 CCI。",
        "(TP - SMA(TP)) / (0.015·MAD)。",
        r"\mathrm{CCI} = \frac{TP - \mathrm{SMA}(TP)}{0.015\cdot \mathrm{MAD}(TP)}",
    ),
    "MOM": OpDoc(
        "动量 MOM。",
        "P_t - P_{t-d}。",
        r"M_t = P_t - P_{t-d}",
    ),
    "ROC": OpDoc(
        "变动率 ROC。",
        "(P_t / P_{t-d} - 1) × 100。",
        r"\mathrm{ROC} = 100\cdot\left(\frac{P_t}{P_{t-d}}-1\right)",
    ),
    "TRIX": OpDoc(
        "三重 EMA 的 ROC。",
        "对三重 EMA 做 1 期变化率。",
        r"\mathrm{TRIX} = \frac{\mathrm{EMA}(\mathrm{EMA}(\mathrm{EMA}(P)))_t}{\mathrm{EMA}^3_{t-1}} - 1",
    ),
    "OBV": OpDoc(
        "能量潮 OBV。",
        "成交量按价格涨跌符号累加。",
        r"\mathrm{OBV}_t = \mathrm{OBV}_{t-1} + \mathrm{sgn}(\Delta P)\cdot V_t",
    ),
    "ts_decay_linear": OpDoc(
        "线性衰减加权求和。",
        "窗口内按 1,2,…,d 线性权重加权；别名 decay_linear、ts_decay。",
        r"\sum_{k=0}^{d-1}\frac{d-k}{\sum_{j=1}^{d}j} X_{t-k}",
    ),
    "decay_linear": OpDoc(
        "线性衰减加权求和（同 ts_decay_linear）。",
        "窗口内按 1,2,…,d 线性权重加权。",
        r"\sum_{k=0}^{d-1}\frac{d-k}{\sum_{j=1}^{d}j} X_{t-k}",
    ),
    "ts_decay": OpDoc(
        "指数衰减加权。",
        "近期观测权重更大（半衰或 span 参数以实现为准）。",
        r"\sum_{k} w_k X_{t-k},\quad w_k \propto e^{-\lambda k}",
    ),
    "ts_regression": OpDoc(
        "滚动 OLS 回归输出。",
        "窗口内 Y ~ X 回归，输出斜率/截距/残差/R²（参数控制）。",
        r"Y_{t-k} = \alpha + \beta X_{t-k} + \varepsilon,\quad k=0,\ldots,d-1",
    ),
    "ts_corr": OpDoc(
        "滚动 Pearson 相关系数。",
        "窗口内 corr(X, Y)。",
        r"\rho_{XY}^{(d)} = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}",
    ),
    "ts_cov": OpDoc(
        "滚动协方差。",
        "窗口内 Cov(X, Y)。",
        r"\mathrm{Cov}^{(d)}(X,Y) = \frac{1}{d-1}\sum(X-\bar{X})(Y-\bar{Y})",
    ),
    "ts_delta": OpDoc(
        "d 期差分。",
        "X_t - X_{t-d}。",
        r"\Delta_d X_t = X_{i,t} - X_{i,t-d}",
    ),
    "ts_delay": OpDoc(
        "滞后 d 期。",
        "X_{t-d}。",
        r"X_{i,t-d}",
    ),
    "Lead": OpDoc(
        "超前 d 期（未来值，慎用前视）。",
        "X_{t+d}；仅用于标签或特殊场景。",
        r"X_{i,t+d}",
    ),
    "prev": OpDoc(
        "上一 bar 值。",
        "X_{t-1}。",
        r"X_{i,t-1}",
    ),
    "next": OpDoc(
        "下一 bar 值（前视）。",
        "X_{t+1}。",
        r"X_{i,t+1}",
    ),
}


# ---------------------------------------------------------------------------
# 模式推断
# ---------------------------------------------------------------------------
def _ts_rolling(name: str) -> OpDoc | None:
    stat_map = {
        "ts_mean": ("算术平均", r"\bar{X}_{i,t} = \frac{1}{d}\sum_{k=0}^{d-1} X_{i,t-k}"),
        "ts_std": ("样本标准差", r"\sigma_{i,t}^{(d)} = \sqrt{\frac{1}{d-1}\sum_{k=0}^{d-1}(X_{i,t-k}-\bar{X})^2}"),
        "ts_var": ("样本方差", r"s_{i,t}^{2(d)} = \frac{1}{d-1}\sum_{k=0}^{d-1}(X_{i,t-k}-\bar{X})^2"),
        "ts_median": ("中位数", r"\mathrm{median}(X_{i,t-d+1:t})"),
        "ts_mad": ("MAD", r"\mathrm{MAD} = \mathrm{median}(|X-\mathrm{median}(X)|)"),
        "ts_beta": ("Beta", r"\beta = \mathrm{Cov}(X,Y)/\mathrm{Var}(Y)"),
        "ts_sum": ("求和", r"S_{i,t} = \sum_{k=0}^{d-1} X_{i,t-k}"),
        "ts_max": ("最大值", r"M_{i,t} = \max_{0\le k<d} X_{i,t-k}"),
        "ts_min": ("最小值", r"m_{i,t} = \min_{0\le k<d} X_{i,t-k}"),
        "ts_skew": ("偏度", r"\mathrm{skew}^{(d)}(X) = E\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]"),
        "ts_kurt": ("峰度", r"\mathrm{kurt}^{(d)}(X) = E\left[\left(\frac{X-\mu}{\sigma}\right)^4\right] - 3"),
        "ts_product": ("连乘", r"\prod_{k=0}^{d-1} X_{i,t-k}"),
    }
    if name in stat_map:
        label, latex = stat_map[name]
        return OpDoc(
            f"滚动窗口内的{label}。",
            f"对每个标的，在长度为 d 的窗口 {{t-d+1,…,t}} 上计算{label}；min_periods 规则以实现为准。",
            latex,
        )
    return None


def _elementwise(name: str) -> OpDoc | None:
    unary = {
        "abs": ("绝对值", r"|x|"),
        "neg": ("取负", r"-x"),
        "negate": ("取负", r"-x"),
        "sqrt": ("平方根", r"\sqrt{x}"),
        "sqrt_abs": ("绝对值开方", r"\sqrt{|x|}"),
        "square": ("平方", r"x^2"),
        "sqr": ("平方", r"x^2"),
        "cube": ("立方", r"x^3"),
        "exp": ("自然指数", r"e^x"),
        "exp_neg": ("负指数衰减", r"e^{-x}"),
        "log": ("自然对数", r"\ln x"),
        "log2": ("以 2 为底对数", r"\log_2 x"),
        "log10": ("常用对数", r"\log_{10} x"),
        "log_abs": ("绝对值对数", r"\ln|x|"),
        "sign": ("符号函数", r"\mathrm{sgn}(x)"),
        "reciprocal": ("倒数", r"1/x"),
        "inverse": ("倒数", r"1/x"),
        "inv": ("倒数", r"1/x"),
        "ceil": ("向上取整", r"\lceil x\rceil"),
        "floor": ("向下取整", r"\lfloor x\rfloor"),
        "round": ("四舍五入", r"\mathrm{round}(x)"),
        "fix": ("向零取整", r"\mathrm{trunc}(x)"),
        "sin": ("正弦", r"\sin x"),
        "cos": ("余弦", r"\cos x"),
        "tan": ("正切", r"\tan x"),
        "sinh": ("双曲正弦", r"\sinh x"),
        "cosh": ("双曲余弦", r"\cosh x"),
        "tanh": ("双曲正切", r"\tanh x"),
        "asin": ("反正弦", r"\arcsin x"),
        "acos": ("反余弦", r"\arccos x"),
        "atan": ("反正切", r"\arctan x"),
        "cbrt": ("立方根", r"\sqrt[3]{x}"),
        "real": ("复数实部", r"\Re(z)"),
        "imag": ("复数虚部", r"\Im(z)"),
    }
    binary = {
        "add": ("加法", r"a + b"),
        "subtract": ("减法", r"a - b"),
        "multiply": ("乘法", r"a \cdot b"),
        "divide": ("除法", r"a / b"),
        "power": ("幂", r"a^b"),
        "signed_power": ("保号幂", r"\mathrm{sgn}(a)\cdot|a|^b"),
        "fmax": ("NaN 感知 max", r"\max(a,b)"),
        "fmin": ("NaN 感知 min", r"\min(a,b)"),
        "flex_max": ("逐元素 max", r"\max(a,b)"),
        "flex_min": ("逐元素 min", r"\min(a,b)"),
    }
    compare = {
        "eq": ("等于", r"a = b"),
        "ne": ("不等于", r"a \ne b"),
        "gt": ("大于", r"a > b"),
        "ge": ("大于等于", r"a \ge b"),
        "lt": ("小于", r"a < b"),
        "le": ("小于等于", r"a \le b"),
        "and_": ("逻辑与", r"a \land b"),
        "or_": ("逻辑或", r"a \lor b"),
        "not_": ("逻辑非", r"\lnot a"),
    }
    if name in unary:
        label, latex = unary[name]
        return OpDoc(f"逐元素{label}。", f"对 panel 每个元素应用{label}。", latex)
    if name in binary:
        label, latex = binary[name]
        return OpDoc(f"逐元素{label}。", f"两个同形序列按位置{label}。", latex)
    if name in compare:
        label, latex = compare[name]
        return OpDoc(f"逐元素比较：{label}。", "两个序列按位置比较，输出布尔或 0/1。", latex)
    return None


def _cum_shift(name: str) -> OpDoc | None:
    cum_map = {
        "cum_sum": ("累计和", r"C_t = \sum_{s\le t} X_s"),
        "cum_prod": ("累计积", r"C_t = \prod_{s\le t} X_s"),
        "cum_max": ("累计最大值", r"C_t = \max_{s\le t} X_s"),
        "cum_min": ("累计最小值", r"C_t = \min_{s\le t} X_s"),
        "cum_avg": ("累计均值", r"C_t = \frac{1}{t}\sum_{s\le t} X_s"),
        "cum_std": ("累计标准差", r"C_t = \mathrm{std}(X_{1:t})"),
        "cum_delta": ("累计差分", r"C_t = X_t - X_1"),
        "cum_count": ("非 NaN 计数", r"C_t = \#\{s\le t: X_s \text{ 有效}\}"),
        "cum_rank": ("累计秩", r"对 X_{1:t} 做秩变换"),
        "cumulative_max": ("全样本累计最大", r"\max_{s\le t} X_s"),
        "cumulative_min": ("全样本累计最小", r"\min_{s\le t} X_s"),
        "cumulative_mean": ("全样本累计均值", r"\frac{1}{t}\sum_{s\le t} X_s"),
        "deltas": ("一阶差分", r"\Delta X_t = X_t - X_{t-1}"),
    }
    if name in cum_map:
        label, latex = cum_map[name]
        return OpDoc(f"时序{label}。", f"沿时间轴从样本起点累计计算{label}。", latex)
    return None


def _cross_sectional_row(name: str) -> OpDoc | None:
    if name.startswith("row_"):
        stat = name[4:]
        stat_map = {
            "avg": (r"\bar{X}_{\cdot,t} = \frac{1}{N}\sum_i X_{i,t}", "截面均值"),
            "sum": (r"S_{\cdot,t} = \sum_i X_{i,t}", "截面求和"),
            "max": (r"M_{\cdot,t} = \max_i X_{i,t}", "截面最大值"),
            "min": (r"m_{\cdot,t} = \min_i X_{i,t}", "截面最小值"),
            "std": (r"\sigma_{\cdot,t} = \mathrm{std}_i(X_{i,t})", "截面标准差"),
            "var": (r"s^2_{\cdot,t} = \mathrm{var}_i(X_{i,t})", "截面方差"),
            "median": (r"\mathrm{median}_i(X_{i,t})", "截面中位数"),
            "count": (r"N_t = \#\{i: X_{i,t}\ \text{有效}\}", "截面计数"),
            "prod": (r"\prod_i X_{i,t}", "截面乘积"),
            "skew": (r"\mathrm{skew}_i(X_{i,t})", "截面偏度"),
            "kurt": (r"\mathrm{kurt}_i(X_{i,t})", "截面峰度"),
            "corr": (r"\rho_{XY,t} = \mathrm{corr}_i(X,Y)", "截面相关系数"),
            "beta": (r"\beta_t = \mathrm{Cov}(X,Y)/\mathrm{Var}(Y)", "截面 Beta"),
        }
        if stat in stat_map:
            latex, label = stat_map[stat]
            return OpDoc(
                f"截面{label}（每个交易日一行）。",
                f"在每个 timestamp 对所有标的计算{label}。",
                latex,
            )
    return None


def _group_prefix(name: str) -> OpDoc | None:
    if not name.startswith("group_"):
        return None
    if name in _EXPLICIT:
        return _EXPLICIT[name]
    inner = name[6:]
    return OpDoc(
        f"分组截面 {inner}。",
        f"每个 (timestamp, group) 内计算 {inner}，再映射回各标的。",
        rf"\mathrm{{group\_}}{inner}(X_{{i,t}}, g(i))",
    )


def _stat_test(name: str) -> OpDoc | None:
    if "test" in name or name.startswith("ttest"):
        return OpDoc(
            "统计假设检验统计量或 p 值。",
            "在指定窗口或截面上执行标准统计检验（详见 scipy/statsmodels 实现）。",
            r"T = f(X;\,\theta),\quad p = P(T \ge T_{\mathrm{obs}}\mid H_0)",
        )
    return None


def _window_prefix(name: str) -> OpDoc | None:
    stat_map = {
        "mean": (r"\bar{X}_t = \frac{1}{w}\sum_{k=0}^{w-1} X_{t-k}", "算术平均"),
        "sum": (r"S_t = \sum_{k=0}^{w-1} X_{t-k}", "求和"),
        "max": (r"M_t = \max_{0\le k<w} X_{t-k}", "最大值"),
        "min": (r"m_t = \min_{0\le k<w} X_{t-k}", "最小值"),
        "std": (r"\sigma_t = \mathrm{std}(X_{t-w+1:t})", "标准差"),
    }
    if name.startswith("window_"):
        stat = name[7:]
        if stat in stat_map:
            latex, label = stat_map[stat]
            return OpDoc(
                f"固定窗口{label}。",
                f"长度为 window 的滚动{label}。",
                latex,
            )
    if name.startswith("expanding_"):
        stat = name[10:]
        exp_map = {
            "mean": (r"\bar{X}_t = \frac{1}{t}\sum_{s=1}^{t} X_s", "扩展均值"),
            "sum": (r"S_t = \sum_{s=1}^{t} X_s", "扩展求和"),
            "max": (r"M_t = \max_{s\le t} X_s", "扩展最大值"),
            "min": (r"m_t = \min_{s\le t} X_s", "扩展最小值"),
            "std": (r"\sigma_t = \mathrm{std}(X_{1:t})", "扩展标准差"),
            "rank": (r"R_t = \frac{\#\{s\le t: X_s\le X_t\}-1}{t-1}", "扩展秩分位"),
        }
        if stat in exp_map:
            latex, label = exp_map[stat]
            return OpDoc(
                f"{label}。",
                f"从样本起点到 t 的 expanding {stat}。",
                latex,
            )
    if name.startswith("ewm_"):
        stat = name[4:]
        ewm_map = {
            "mean": (r"\mathrm{EWM}_t = \alpha X_t + (1-\alpha)\mathrm{EWM}_{t-1}", "指数加权均值"),
            "std": (r"\mathrm{EWM\_std}_t", "指数加权标准差"),
            "var": (r"\mathrm{EWM\_var}_t", "指数加权方差"),
            "corr": (r"\rho_t^{\mathrm{ewm}} = \mathrm{EWM\_corr}(X,Y)", "指数加权相关"),
            "cov": (r"\mathrm{EWM\_cov}_t(X,Y)", "指数加权协方差"),
        }
        if stat in ewm_map:
            latex, label = ewm_map[stat]
            return OpDoc(
                f"{label}。",
                f"span/decay 参数的 EWM {stat}。",
                latex,
            )
    return None


def _m_prefix(name: str) -> OpDoc | None:
    if not name.startswith("m_"):
        return None
    inner = name[2:]
    templates = {
        "beta": ("滚动 Beta", r"\beta = \mathrm{Cov}(X,Y)/\mathrm{Var}(Y)"),
        "var": ("滚动方差", r"s^2 = \frac{1}{d-1}\sum(X-\bar{X})^2"),
        "mad": ("滚动 MAD", r"\mathrm{MAD} = \mathrm{median}(|X-\mathrm{median}(X)|)"),
        "median": ("滚动中位数", r"\mathrm{median}(X_{t-d+1:t})"),
        "zscore": ("滚动 Z 分数", r"Z = (X-\mu)/\sigma"),
        "top_n_avg": ("窗口 top-N 均值", r"\mathrm{mean}(X^{(1:N)})"),
        "top_n_std": ("窗口 top-N 标准差", r"\mathrm{std}(X^{(1:N)})"),
        "bottom_n_avg": ("窗口 bottom-N 均值", r"\mathrm{mean}(X^{(N:d)})"),
        "bottom_n_sum": ("窗口 bottom-N 求和", r"\sum X^{(N:d)}"),
    }
    if inner in templates:
        label, latex = templates[inner]
        return OpDoc(f"{label}。", f"在长度为 d 的窗口内计算 {label}。", latex)
    return OpDoc(f"滚动扩展统计 m_{inner}。", f"窗口内 {inner} 统计量。", rf"\mathrm{{m\_}}{inner}^{{(d)}}(X)")


def _fill_clean(name: str) -> OpDoc | None:
    clean = {
        "fillna": ("缺失值填充", "用指定值或方法填充 NaN", r"\tilde{x} = x \text{ 若有限，否则 fill}"),
        "ffill": ("前向填充", "用上一个有效值填充", r"\tilde{x}_t = x_{t'} \text{，} t'=\max\{s\le t: x_s \text{ 有效}\}"),
        "bfill": ("后向填充", "用下一个有效值填充", r"\tilde{x}_t = x_{t'} \text{，} t'=\min\{s\ge t: x_s \text{ 有效}\}"),
        "nan_to_num": ("NaN/inf 转数值", "NaN→0，inf→大数", r"\mathrm{nan\_to\_num}(x)"),
        "dropna": ("删除缺失", "剔除 NaN 行/列", r"\text{保留 } x \text{ 有限的位置"),
        "ifnan": ("NaN 替换", "x 为 NaN 时取默认值", r"\begin{cases}x & x \text{ 有效}\\ d & \text{否则}\end{cases}"),
        "winsorize_mean": ("缩尾后均值", "先 winsorize 再求均值", r"\mathrm{mean}(\mathrm{winsorize}(X))"),
    }
    if name in clean:
        m, c, l = clean[name]
        return OpDoc(m + "。", c + "。", l)
    return None


def _fallback(name: str, description: str, category: str) -> OpDoc:
    cat_zh = {
        "elementwise_math": "逐元素",
        "time_series": "时序滚动",
        "cross_sectional": "截面",
        "group_neutralization": "分组截面",
        "data_cleaning": "数据清洗",
        "statistics_regression": "统计/回归",
        "price_volume": "价量衍生",
        "technical_signal": "技术信号",
        "fundamental": "基本面",
        "intraday_microstructure": "微观结构",
    }.get(category, category)
    desc = description if description and description not in (
        "Basic runtime operator", "LQTP numpy implementation", "（暂无描述）"
    ) else f"{cat_zh}算子 `{name}`"
    return OpDoc(
        desc + "。",
        f"在 {cat_zh} 语义下对 panel 输入执行 `{name}`；具体边界条件（min_periods、NaN）以实现代码为准。",
        rf"\mathrm{{{name}}}(X)",
    )


def get_operator_doc(
    name: str,
    *,
    description: str = "",
    category: str = "other",
) -> OpDoc:
    """返回算子的含义、计算说明与 LaTeX 公式。"""
    from cleaned_operators.docs._operator_doc_batch import BATCH_DOCS

    if name in _EXPLICIT:
        return _EXPLICIT[name]
    if name in BATCH_DOCS:
        return BATCH_DOCS[name]
    for fn in (
        _ts_rolling,
        _elementwise,
        _cum_shift,
        _cross_sectional_row,
        _group_prefix,
        _stat_test,
        _window_prefix,
        _m_prefix,
        _fill_clean,
    ):
        doc = fn(name)
        if doc is not None:
            return doc
    return _fallback(name, description, category)
