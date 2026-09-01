"""Factor Survival Profile（任务书 §45 / Phase 10）。

把跨时期的 monthly RankIC 序列压缩成 SurvivalProfile（recent retention /
sign flip / rolling slope / q20 / max deterioration / descriptive half-life），
并按 §45.1 规则分类为 SurvivalLabel。

纯统计/规则代码：不做模型、不做指数拟合之外的任何估计；§45.3 明确
「不要强迫所有 factor 用指数衰减模型」——half-life 仅在指数拟合 R²>0.5
时给出，否则为 None。
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

from alphaprobe.contracts import SurvivalLabel, SurvivalProfile

__all__ = [
    "compute_survival_metrics",
    "classify_status",
    "RECENT_FRACTION",
    "RETENTION_HIGH",
    "RETENTION_LOW",
    "RECOVERY_THRESHOLD",
    "HALF_LIFE_MIN_R2",
]

#: 末段占比：recent = 最后 3 个月（或 30% 的最小段）
RECENT_FRACTION = 0.30
#: §45.1 阈值：retention > 0.8 → HEALTHY
RETENTION_HIGH = 0.8
#: §45.1 阈值：retention < 0.4 → DEGRADING
RETENTION_LOW = 0.4
#: §45.1 阈值：recent 段均值 < 0 → BROKEN；回升超过 0.6 → RECOVERED
RECOVERY_THRESHOLD = 0.6
#: §45.3：指数拟合 R² > 0.5 才给 descriptive half-life
HALF_LIFE_MIN_R2 = 0.5
#: 指数拟合最少观测数
_HALF_LIFE_MIN_N = 4


def _clip(v: float | None, lo: float, hi: float) -> float | None:
    if v is None or not math.isfinite(v):
        return None
    return min(max(v, lo), hi)


def _signed_retention(seq: Sequence[float]) -> float:
    """signed retention：分子为含符号和，分母为绝对值和。

    §45.1 判定负数段（BROKEN）需要「负贡献占比」：
    (Σ 正段) / (Σ|段|) 在符号翻转时会把一段 -0.06 的衰退如实反映为负 retention。
    """
    pos = sum(max(0.0, v) for v in seq)
    denom = sum(abs(v) for v in seq)
    if denom <= 0.0:
        return 0.0
    return pos / denom * 2.0 - 1.0


def compute_survival_metrics(
    monthly_rankic: Sequence[float] | list[float],
) -> SurvivalProfile:
    """§45.2：从 monthly RankIC 序列计算 SurvivalProfile。

    Parameters
    ----------
    monthly_rankic : list[float]
        月度 RankIC 序列（时间升序）。空序列返回 UNCLASSIFIED profile。

    Returns
    -------
    SurvivalProfile
        - recent_retention：末段(30% 或末 3 期)均值 / 前段均值的比值，clip 到
          [-1, 1]（负值表示最近段为负贡献，§45.1 判定负数段）；
        - sign_flip_rate：相邻月符号翻转比例（0~1）；
        - rolling_slope：对序列时间索引线性拟合斜率（标准化）;
        - q20_rolling_rankic：滚动窗口 20% 分位 RankIC（窗口 min(3, len)）;
        - max_deterioration：单调最坏累积回撤（任何位置之后的最大下降）;
        - descriptive_half_life：仅当指数拟合 R²>0.5 时给 ln2/λ。
    """
    seq = [float(x) for x in monthly_rankic]
    if not seq:
        return SurvivalProfile(factor_id="")

    n = len(seq)
    recent_n = max(3, round(n * RECENT_FRACTION)) if n >= 4 else n
    recent_seg = seq[-recent_n:]
    recent_mean = float(statistics.fmean(recent_seg)) if recent_seg else 0.0

    recent_retention: float | None
    if n >= 4:
        prior_seg = seq[:-recent_n]
        if prior_seg:
            prior_mean = float(statistics.fmean(prior_seg))
            # §45.2 末3月均值/前段均值，clip 到 [-1,1]（保持符号）
            recent_retention = _clip(
                recent_mean / max(abs(prior_mean), 1e-9), -1.0, 1.0
            )
        else:
            recent_retention = None
    else:
        recent_retention = None

    flips = sum(1 for a, b in zip(seq, seq[1:]) if (a >= 0) != (b >= 0))
    sign_flip_rate = flips / max(n - 1, 1)

    # rolling slope：时间索引线性拟合，标准化为 [0,1]（0=持平，>0.5=上升）
    rolling_slope = None
    if n >= 2:
        xs = list(range(n))
        mean_x = (n - 1) / 2.0
        mean_y = float(statistics.fmean(seq))
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom > 0:
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, seq)) / denom
            rolling_slope = _clip(0.5 + slope * 5.0, 0.0, 1.0)

    # q20 rolling rankic：滚动窗口 20% 分位
    q20_rolling_rankic = None
    if n >= 1:
        window = min(3, n)
        quantile = max(0.2, 1.0 / max(window, 1))
        q20_rolling_rankic = float(sorted(seq)[int((len(seq) - 1) * quantile)])

    # max deterioration：任何位置之后的最大下降
    max_deterioration = None
    if n >= 2:
        worst = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                drop = seq[i] - seq[j]
                if drop > worst:
                    worst = drop
        max_deterioration = float(worst)

    descriptive_half_life = _fit_exponential_half_life(seq)

    return SurvivalProfile(
        factor_id="",
        recent_retention=recent_retention,
        rolling_slope=rolling_slope,
        q20_rolling_rankic=q20_rolling_rankic,
        sign_flip_rate=sign_flip_rate,
        max_deterioration=max_deterioration,
        descriptive_half_life=descriptive_half_life,
        support_periods=n,
    )


def _fit_exponential_half_life(seq: Sequence[float]) -> float | None:
    """§45.3：IC_t ≈ IC_0 · e^{-λt} 线性化拟合；仅 R²>0.5 时给 ln2/λ。

    不足 4 期、IC_0 过小、或拟合质量差时返回 None（不强求指数模型）。
    """
    n = len(seq)
    if n < _HALF_LIFE_MIN_N:
        return None
    # 线性化：log|IC_t| = log|IC_0| - λ·t
    try:
        xs: list[float] = []
        ys: list[float] = []
        for t, v in enumerate(seq):
            a = abs(v)
            if a <= 1e-9:
                continue
            xs.append(float(t))
            ys.append(math.log(a))
        if len(xs) < _HALF_LIFE_MIN_N:
            return None
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom <= 0:
            return None
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        intercept = mean_y - slope * mean_x
        if intercept is None or slope is None:
            return None
        ic0 = math.exp(intercept)
        if not math.isfinite(ic0) or ic0 <= 1e-9:
            return None
        # 拟合优度 R²
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        if ss_tot <= 0:
            return None
        ss_res = sum(
            (y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)
        )
        r2 = 1.0 - ss_res / ss_tot
        if r2 < HALF_LIFE_MIN_R2:
            return None
        lam = -slope
        if lam <= 0:
            return None
        half_life = math.log(2.0) / lam
        if not math.isfinite(half_life) or half_life <= 0:
            return None
        return min(half_life, 1e6)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# §45.1 Survival Labels（规则分类，阈值全注释）
# ---------------------------------------------------------------------------

def _recent_positive_ratio(seq: Sequence[float]) -> float:
    """最近 30% 段内正 RankIC 占比（[0,1]）。"""
    n = len(seq)
    if n == 0:
        return 0.0
    recent_n = max(3, round(n * RECENT_FRACTION)) if n >= 4 else n
    recent = seq[-recent_n:]
    return sum(1.0 for v in recent if v > 0) / len(recent)


def classify_status(profile: SurvivalProfile) -> SurvivalLabel:
    """§45.1 规则分类。

    优先级（写死顺序，避免重叠导致多义）：

    1. BROKEN    — 最近 30% 段 RankIC 均值 < 0（段为负贡献）
    2. RECOVERED — 曾在早期为负（早期 30% 段均值 < 0）且近期均值 > 0.6×历史峰值
    3. PERSISTENT_ALPHA — 全序列支持期 ≥ 12 且各段均值全 > 0 且最近段 > 0
    4. HEALTHY   — recent_retention > 0.8 且 slope 未恶化
    5. DEGRADING — recent_retention < 0.4 或 max_deterioration 超阈值
    6. REGIME_DEPENDENT — 信号翻转率高（≥ 0.4）且斜率未明确下降
    7. UNCLASSIFIED — 其余
    """
    seq: list[float] = getattr(profile, "_rankic_sequence", None)
    if not seq:
        return SurvivalLabel.UNCLASSIFIED

    n = len(seq)
    if n == 0:
        return SurvivalLabel.UNCLASSIFIED

    recent_n = max(3, round(n * RECENT_FRACTION)) if n >= 4 else n
    early_n = max(3, round(n * RECENT_FRACTION)) if n >= 4 else n
    recent = seq[-recent_n:]
    early = seq[:early_n]

    recent_mean = float(statistics.fmean(recent)) if recent else 0.0
    early_mean = float(statistics.fmean(early)) if early else 0.0
    hist_max = max(0.0, max(seq))

    retention = profile.recent_retention or 0.0
    slope = profile.rolling_slope
    flip_rate = profile.sign_flip_rate or 0.0
    max_det = profile.max_deterioration or 0.0

    # 1) BROKEN：最近段均值 < 0（段为负贡献）
    if recent_mean < 0:
        return SurvivalLabel.BROKEN

    # 2) RECOVERED：早期段均值 < 0 且最近段显著回升（> 0.6 × 历史峰值）
    if early_mean < 0 and hist_max > 1e-9 and recent_mean > RECOVERY_THRESHOLD * hist_max:
        return SurvivalLabel.RECOVERED

    # 3) PERSISTENT_ALPHA：支持期 ≥ 12 且每段（early/recent）均值 > 0 且近期仍 > 0
    if (
        n >= 12
        and early_mean > 0
        and recent_mean > 0
        and all(v > 0 for v in seq)
    ):
        return SurvivalLabel.PERSISTENT_ALPHA

    # 4) HEALTHY：retention > 0.8 且斜率未恶化
    if retention > RETENTION_HIGH and (slope is None or slope >= 0.35):
        return SurvivalLabel.HEALTHY

    # 4b) 样本不足（<6 期）：不做强结论，先归 UNCLASSIFIED（§45.1 低支撑不下结论）
    if n < 6:
        return SurvivalLabel.UNCLASSIFIED

    # 5) DEGRADING：retention < 0.4 或恶化超过 0.5
    if retention < RETENTION_LOW or max_det > 0.5:
        return SurvivalLabel.DEGRADING

    # 6) REGIME_DEPENDENT：信号翻转频繁但未明确持续下降
    if flip_rate >= 0.4 and (slope is None or slope < 0.65):
        return SurvivalLabel.REGIME_DEPENDENT

    # 7) 兜底
    return SurvivalLabel.UNCLASSIFIED


def classify_status_from_sequence(
    monthly_rankic: Sequence[float] | list[float],
) -> tuple[SurvivalProfile, SurvivalLabel]:
    """便捷入口：直接对 monthly RankIC 序列计算 profile + label。"""
    profile = compute_survival_metrics(monthly_rankic)
    # 把序列挂到 profile 供 classify_status 使用（不落盘字段）
    object.__setattr__(profile, "_rankic_sequence", [float(x) for x in monthly_rankic])
    label = classify_status(profile)
    return profile, label
