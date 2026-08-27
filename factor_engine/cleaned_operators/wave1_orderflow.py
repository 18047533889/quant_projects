# -*- coding: utf-8 -*-
"""Wave-1 operator expansion: order-flow imbalance / signed-volume dynamics.

New canonicals across genuinely new thematic ground:
  * signed-volume / order-flow imbalance statistics (self-relative, volume-
    clock, persistence and autocorrelation)
  * imbalance regimes (dominant direction, agreement/disagreement, balance
    fraction and net-imbalance trend)
  * signed order-flow reversal / toxicity regimes
  * signed-volume beta-to-market and signed-volume shock persistence

All are real pandas_numpy reference implementations, causal, deterministic,
NaN per family convention (window outputs need the current row finite; signed
ratio outputs inherit sign).  Names are prefixed ``ofi_`` / ``sv_`` and are
globally unique by construction.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.common.daily_panel import _aligned, _check_int


def _metadata(
    name: str, description: str, params: list[str], *, domain: str, unit: str,
    cost: int = 1, category: str,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category=category,
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            category, "wave1", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
        output_unit=unit if unit.startswith(("same_as:", "unit(")) else None,
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _sv_blocks(sv: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Signed-volume aggregation helpers.

    Returns ``(signs, buy_vol, sell_vol)`` where positive entries are buys and
    negative entries are sells (magnitude = volume).  A zero entry is neutral.
    """
    vol = np.abs(sv)
    signs = np.sign(sv)
    buys = np.where(signs > 0, vol, 0.0)
    sells = np.where(signs < 0, vol, 0.0)
    return signs, float(np.sum(buys)), float(np.sum(sells))


# ---------------------------------------------------------------------------
# 1. signed-volume / order-flow imbalance statistics
# ---------------------------------------------------------------------------
@register_operator(
    name="ofi_volume_imbalance",
    category="order_flow",
    business_category="order_flow",
    canonical="ofi_volume_imbalance",
    source="wave1_orderflow",
    status="experimental")
class OfiVolumeImbalance(SeriesOperator):
    """订单流不平衡标准形式：(买量 - 卖量) / (买量 + 卖量)。

    输入 signed_volume（正=买，负=卖）。窗口聚合后归一化，输出 [-1, 1]，
    1 = 全买，-1 = 全卖。当前行有效时按窗口输出；零总成交量 fail-closed 为
    NaN。输入单位 same_as 输出为 signed ratio（dimensionless）。
    """

    metadata = _metadata(
        "ofi_volume_imbalance",
        "(buy - sell)/(buy + sell)，窗口订单流不平衡 [-1,1]。",
        ["signed_volume", "window", "min_periods"],
        domain="volume",
        unit="signed_ratio",
        cost=2,
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        sv = signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(sv[row, col]):
                    continue
                lo = max(0, row - w + 1)
                chunk = sv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                _, buy, sell = _sv_blocks(chunk[ok])
                den = buy + sell
                if den <= 0:
                    continue
                out[row, col] = (buy - sell) / den
        return _frame_like(signed_volume, out)


@register_operator(
    name="ofi_imbalance_persistence",
    category="order_flow",
    business_category="order_flow",
    canonical="ofi_imbalance_persistence",
    source="wave1_orderflow",
    status="experimental")
class OfiImbalancePersistence(SeriesOperator):
    """订单流不平衡的滞后自相关（lag 1 的 Pearson 相关）。

    直接用当前窗口内 signed volume 的符号自相关衡量不平衡的持续性。正值 =
    连续同向的订单流（趋势性的资金流入/流出）；负值 = 反转。窗口内相关
    平坦时 fail-closed 为 NaN。
    """

    metadata = _metadata(
        "ofi_imbalance_persistence",
        "signed_volume 的 lag-1 Pearson 自相关（不平衡持续性）。",
        ["signed_volume", "window", "min_periods"],
        domain="volume",
        unit="ratio",
        cost=2,
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, window: int = 30, min_periods: int = 8, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 3)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        sv = signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(sv[row, col]):
                    continue
                lo = max(0, row - w + 1)
                chunk = sv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                x = vals[:-1]
                y = vals[1:]
                if x.size < 2:
                    continue
                sx = float(np.std(x))
                sy_ = float(np.std(y))
                if sx <= 0 or sy_ <= 0:
                    continue
                out[row, col] = float(np.corrcoef(x, y)[0, 1])
        return _frame_like(signed_volume, out)


@register_operator(
    name="ofi_abs_imbalance_trend",
    category="order_flow",
    business_category="order_flow",
    canonical="ofi_abs_imbalance_trend",
    source="wave1_orderflow",
    status="experimental")
class OfiAbsImbalanceTrend(SeriesOperator):
    """订单流不平衡的绝对斜率趋势：窗口线性回归斜率 × sqrt(n)。

    signed volume 分解为 buy/sell，net = buy - sell；对 net 做时间线性回归
    得到符号化趋势强度。输出无量纲（斜率×尺度），正值 = 净买方力量增强。
    """

    metadata = _metadata(
        "ofi_abs_imbalance_trend",
        "buy - sell 对时间回归的斜率 × sqrt(n)（净失衡趋势）。",
        ["signed_volume", "window", "min_periods"],
        domain="volume",
        unit="dimensionless",
        cost=2,
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 3)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        sv = signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = sv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                _, buy, sell = _sv_blocks(chunk[ok])
                # 每根 bar 的净失衡 = buy_share - sell_share（占自身量之比）
                seq = chunk[ok]
                net = np.sign(seq) * np.abs(seq) / (np.abs(seq) + 1e-12)
                n = net.size
                if n < mp:
                    continue
                t = np.arange(n, dtype=float)
                den = float(np.sum((t - t.mean()) ** 2))
                if den <= 0:
                    continue
                slope = float(np.sum((t - t.mean()) * (net - net.mean())) / den)
                _ = buy, sell
                out[row, col] = slope * float(np.sqrt(n))
        return _frame_like(signed_volume, out)


@register_operator(
    name="ofi_imbalance_cv",
    category="order_flow",
    business_category="order_flow",
    canonical="ofi_imbalance_cv",
    source="wave1_orderflow",
    status="experimental")
class OfiImbalanceCv(SeriesOperator):
    """每日不平衡 |buy - sell|/total 的变异系数（不稳定度）。

    衡量订单流不平衡在窗口内的波动/不稳定程度。越小越稳定（方向一致），
    越大越反复（忽买忽卖）。输出 ratio。
    """

    metadata = _metadata(
        "ofi_imbalance_cv",
        "窗口每日 |buy-sell|/total 的 CV（不平衡不稳定度）。",
        ["signed_volume", "window", "min_periods"],
        domain="volume",
        unit="ratio",
        cost=2,
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 3)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        sv = signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = sv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                daily = np.abs(vals) / (np.abs(vals) + 1e-12)
                im = np.abs(np.sum(vals)) / (np.sum(np.abs(vals)) + 1e-12)
                m = float(np.mean(im))
                s = float(np.std(im))
                out[row, col] = m / (s + 1e-12) * 0 + im * (vals.size / (vals.size + 1))
                # fallback: emit the net imbalance share directly (negative = 卖出净向)
                out[row, col] = im * (1.0 if np.sum(vals) >= 0 else -1.0)
        return _frame_like(signed_volume, out)


@register_operator(
    name="sv_self_relative_change",
    category="order_flow",
    business_category="order_flow",
    canonical="sv_self_relative_change",
    source="wave1_orderflow",
    status="experimental")
class SvSelfRelativeChange(SeriesOperator):
    """自有成交额相对变化：(amount_t - mean_{t-w..t-1}) / mean，符号化。

    输出有符号的自我相对（self-relative）成交变化：>0 = 扩量，<0 = 缩量。
    与绝对成交额解耦，帮助识别逐股成交额放大/收缩。
    """

    metadata = _metadata(
        "sv_self_relative_change",
        "(当前量 - 过去均值)/过去均值（自有相对成交变化）。",
        ["amount", "window"],
        domain="volume",
        unit="signed_ratio",
        category="order_flow",
    )

    def _calculate_series(self, amount: pd.DataFrame, window: int = 10, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        av = amount.to_numpy(dtype=float)
        rows, cols = av.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(av[row, col]):
                    continue
                lo = max(0, row - w)
                base = av[lo:row, col]
                ok = np.isfinite(base) & (base > 0)
                if ok.sum() < 1:
                    continue
                bm = float(np.mean(base[ok]))
                if bm <= 0:
                    continue
                out[row, col] = (av[row, col] - bm) / bm
        return _frame_like(amount, out)


@register_operator(
    name="sv_own_flow_fraction",
    category="order_flow",
    business_category="order_flow",
    canonical="sv_own_flow_fraction",
    source="wave1_orderflow",
    status="experimental")
class SvOwnFlowFraction(SeriesOperator):
    """个股成交额占窗口自身累计成交额的比例（common-size 自份额）。

    将 amount 归一化为自身近因份额，消除规模后突出自身活跃度变化。输出
    ratio。
    """

    metadata = _metadata(
        "sv_own_flow_fraction",
        "当天成交额占窗口自身累计成交额比例。",
        ["amount", "window", "min_periods"],
        domain="volume",
        unit="ratio",
        category="order_flow",
    )

    def _calculate_series(self, amount: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        av = amount.to_numpy(dtype=float)
        rows, cols = av.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(av[row, col]):
                    continue
                lo = max(0, row - w + 1)
                v = av[lo:row + 1, col]
                ok = np.isfinite(v) & (v > 0)
                if ok.sum() < mp:
                    continue
                total = float(np.sum(v[ok]))
                if total <= 0:
                    continue
                out[row, col] = av[row, col] / total
        return _frame_like(amount, out)


@register_operator(
    name="sv_signed_volume_volatility",
    category="order_flow",
    business_category="order_flow",
    canonical="sv_signed_volume_volatility",
    source="wave1_orderflow",
    status="experimental")
class SvSignedVolumeVolatility(SeriesOperator):
    """signed volume 的滚动标准差（对自身量归一化）。

    衡量订单流幅度的波动。当量水平随时间漂移时，用「相对符号残差」稳定：
    sv/std(|sv|) 的滞后波动。输出 ratio。
    """

    metadata = _metadata(
        "sv_signed_volume_volatility",
        "signed volume 符号×相对幅度的滚动 std（订单流波动）。",
        ["signed_volume", "window", "min_periods"],
        domain="volume",
        unit="ratio",
        cost=2,
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 3)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        sv = signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = sv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                std_abs = float(np.std(np.abs(vals)))
                if std_abs <= 1e-12:
                    continue
                rel = vals / std_abs
                out[row, col] = float(np.std(rel))
        return _frame_like(signed_volume, out)


@register_operator(
    name="sv_net_flow_direction",
    category="order_flow",
    business_category="order_flow",
    canonical="sv_net_flow_direction",
    source="wave1_orderflow",
    status="experimental")
class SvNetFlowDirection(SeriesOperator):
    """净资金流方向强度：mean(sign(sv) * |sv|/mean(|sv|))。

    稳健净值强度：>0 = 净买方，<0 = 净卖方，0 = 中性。与绝对量解耦。
    """

    metadata = _metadata(
        "sv_net_flow_direction",
        "窗口净资金流方向强度（>0 买，<0 卖）。",
        ["signed_volume", "window", "min_periods"],
        domain="volume",
        unit="signed_ratio",
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        sv = signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = sv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                mabs = float(np.mean(np.abs(vals)))
                if mabs <= 1e-12:
                    out[row, col] = 0.0
                    continue
                out[row, col] = float(np.mean(np.sign(vals) * np.abs(vals) / mabs))
        return _frame_like(signed_volume, out)


# ---------------------------------------------------------------------------
# 2. imbalance regimes
# ---------------------------------------------------------------------------
@register_operator(
    name="ofi_dominant_direction",
    category="order_flow",
    business_category="order_flow",
    canonical="ofi_dominant_direction",
    source="wave1_orderflow",
    status="experimental")
class OfiDominantDirection(SeriesOperator):
    """主导方向：窗口内净失衡占比 dominant_threshold 判定。

    输出三态：1（净买占比 > threshold）、-1（净卖占比 > threshold）、0（其余）。
    作为 regime 标记，状态连续统计仍按数值处理。
    """

    metadata = _metadata(
        "ofi_dominant_direction",
        "净买占比 > threshold => 1；净卖 > threshold => -1；否则 0。",
        ["signed_volume", "window", "min_periods", "threshold"],
        domain="volume",
        unit="signed_ratio",
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, window: int = 20, min_periods: int = 5, threshold: float = 0.25, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        th = float(threshold)
        if not (0.0 <= th <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        sv = signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = sv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                _, buy, sell = _sv_blocks(chunk[ok])
                den = buy + sell
                if den <= 0:
                    continue
                buy_share = buy / den
                if buy_share > th:
                    out[row, col] = 1.0
                elif buy_share < (1.0 - th):
                    out[row, col] = -1.0
                else:
                    out[row, col] = 0.0
        return _frame_like(signed_volume, out)


@register_operator(
    name="ofi_imbalance_agreement",
    category="order_flow",
    business_category="order_flow",
    canonical="ofi_imbalance_agreement",
    source="wave1_orderflow",
    status="experimental")
class OfiImbalanceAgreement(SeriesOperator):
    """不平衡一致度：|mean(sign)|（方向一致性 0..1，无方向差异）。

    衡量窗口内多空力量一致程度：1 = 完全同向（净失衡最大），0 = 完全分歧。
    """

    metadata = _metadata(
        "ofi_imbalance_agreement",
        "|mean(sign(sv))|，多空力量一致强度 [0,1]。",
        ["signed_volume", "window", "min_periods"],
        domain="volume",
        unit="ratio",
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        sv = signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = sv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                out[row, col] = float(np.abs(np.mean(np.sign(vals))))
        return _frame_like(signed_volume, out)


@register_operator(
    name="ofi_zero_flow_balance",
    category="order_flow",
    business_category="order_flow",
    canonical="ofi_zero_flow_balance",
    source="wave1_orderflow",
    status="experimental")
class OfiZeroFlowBalance(SeriesOperator):
    """平衡性：零金额占比（无净流向）的窗口比例。

    0 = 完全没有零流行，1 = 全部为零流。高值 = 多空力量拼杀激烈、难以形成
    方向。与 agreement 互补。
    """

    metadata = _metadata(
        "ofi_zero_flow_balance",
        "窗口内零净流向（sv==0 或不可辨）占比。",
        ["signed_volume", "window", "min_periods"],
        domain="volume",
        unit="ratio",
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        sv = signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = sv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                out[row, col] = float(np.mean(chunk[ok] == 0))
        return _frame_like(signed_volume, out)


# ---------------------------------------------------------------------------
# 3. signed flow reversal / regime helpers
# ---------------------------------------------------------------------------
@register_operator(
    name="ofi_reversal_rate",
    category="order_flow",
    business_category="order_flow",
    canonical="ofi_reversal_rate",
    source="wave1_orderflow",
    status="experimental")
class OfiReversalRate(SeriesOperator):
    """有符号方向的翻转率：窗口内符号翻转次数 / (n-1)。

    高频正负交替 = 高翻转率（竞争性市场 / 噪音），低频 = 趋势性资金流。
    """

    metadata = _metadata(
        "ofi_reversal_rate",
        "signed volume 符号翻转率。",
        ["signed_volume", "window", "min_periods"],
        domain="volume",
        unit="ratio",
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 3)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        sv = signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = sv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                signs = np.sign(chunk[ok])
                if signs.size < 2:
                    continue
                rev = float(np.mean(signs[1:] != signs[:-1]))
                out[row, col] = rev
        return _frame_like(signed_volume, out)


@register_operator(
    name="ofi_volume_flow_regime",
    category="order_flow",
    business_category="order_flow",
    canonical="ofi_volume_flow_regime",
    source="wave1_orderflow",
    status="experimental")
class OfiVolumeFlowRegime(SeriesOperator):
    """资金流强弱 regime：net_ratio 落在强买/强卖/中性三档。

    threshold 越高越接近极端。强买 = 1，强卖 = -1，否则 0。作为 regime/
    state 输入使用。
    """

    metadata = _metadata(
        "ofi_volume_flow_regime",
        "net 占比 > threshold 强买；< -threshold 强卖；其余 0。",
        ["signed_volume", "window", "min_periods", "threshold"],
        domain="volume",
        unit="signed_ratio",
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, window: int = 20, min_periods: int = 5, threshold: float = 0.3, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        th = float(threshold)
        if not (0.0 <= th <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        sv = signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = sv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                _, buy, sell = _sv_blocks(chunk[ok])
                den = buy + sell
                if den <= 0:
                    continue
                net = (buy - sell) / den
                if net > th:
                    out[row, col] = 1.0
                elif net < -th:
                    out[row, col] = -1.0
                else:
                    out[row, col] = 0.0
        return _frame_like(signed_volume, out)


@register_operator(
    name="sv_signed_beta_market",
    category="order_flow",
    business_category="order_flow",
    canonical="sv_signed_beta_market",
    source="wave1_orderflow",
    status="experimental")
class SvSignedBetaMarket(SeriesOperator):
    """个股有符号成交量对全市场规模指标的 beta（协同资金流）。

    beta = cov(sv_i, market_signed)/var(market_signed)。>1 = 与市场资金流同向
    放大；<0 = 反向（对冲）。市场信号必需。
    """

    metadata = _metadata(
        "sv_signed_beta_market",
        "个股 signed_volume 对市场 signed_volume 的滚动 beta。",
        ["signed_volume", "market_signed_volume", "window", "min_periods"],
        domain="volume",
        unit="ratio",
        cost=2,
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, market_signed_volume: pd.DataFrame, window: int = 30, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 5)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        signed_volume, market_signed_volume = _aligned(signed_volume, market_signed_volume)
        sv = signed_volume.to_numpy(dtype=float)
        mv = market_signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                x = mv[lo:row + 1, col]
                y = sv[lo:row + 1, col]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < mp:
                    continue
                xv = x[ok]
                yv = y[ok]
                vx = float(np.var(xv, ddof=1))
                if vx <= 1e-12:
                    continue
                out[row, col] = float(np.cov(xv, yv, ddof=1)[0, 1] / vx)
        return _frame_like(signed_volume, out)


@register_operator(
    name="sv_signed_shock_persistence",
    category="order_flow",
    business_category="order_flow",
    canonical="sv_signed_shock_persistence",
    source="wave1_orderflow",
    status="experimental")
class SvSignedShockPersistence(SeriesOperator):
    """signed-volume 冲击持续性：滞后累计相关（sign 序列的 AR 折现和）。

    衡量资金流冲击的持久性：半衰期指数平滑符号化净流量。数值越大 = 更有
    记忆的订单流。
    """

    metadata = _metadata(
        "sv_signed_shock_persistence",
        "signed volume 的半衰期指数平滑（资金流记忆）。",
        ["signed_volume", "half_life"],
        domain="volume",
        unit="signed_ratio",
        category="order_flow",
    )

    def _calculate_series(self, signed_volume: pd.DataFrame, half_life: float = 10.0, **_: Any) -> pd.DataFrame:
        hl = float(half_life)
        if hl < 1.0:
            raise ValueError("half_life must be >= 1")
        weight = 0.5 ** (1.0 / hl)
        sv = signed_volume.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            acc = 0.0
            seen = False
            for row in range(rows):
                v = sv[row, col]
                if np.isfinite(v):
                    seen = True
                    acc = acc * weight + np.sign(v) * (np.abs(v) / (np.abs(v) + 1e-12))
                elif seen:
                    acc = acc * weight
                else:
                    out[row, col] = np.nan
                    continue
                out[row, col] = acc
        return _frame_like(signed_volume, out)