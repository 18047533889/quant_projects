"""alphaprobe.legacy（Task 1）：仅 OFFLINE_TEST 的旧评估路径命名空间。

本包承载被 :mod:`alphaprobe.pipeline` 生产主链淘汰的本地手算 evaluate 助手：

- :func:`make_fe_evaluate_fn`：fe_bridge 本地 numpy/scipy bundle 计算（rank_ic /
  ic / icir / coverage / nan_inf_ratio），配 vwap→vwap 远期 label。
- :func:`_bundle_from_plane` / :func:`_all_none`：单因子平面 → metric bundle。

**生产纪律**：PRODUCTION / RESEARCH_DEGRADED 评估只认
``alphaprobe.evaluator_adapter.QuantEvaluatorAdapter``（QE 权威），**绝不**回落
本命名空间的本地手算 RankIC/ICIR（Part G #7）。本包仅 OFFLINE_TEST 的
LegacyCompatEvaluator 注入使用。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)


def make_fe_evaluate_fn(
    stock_data: Any,
    *,
    label_days: int = 20,
    segment: str = "train",
) -> Callable[[Sequence[str], str, Any], list[dict[str, float | None]]]:
    """fe_bridge 真算 evaluate_fn（LegacyCompatEvaluator 注入用，仅 OFFLINE_TEST）。

    公式列表 → FactorEngineStockData.evaluate_many 批算因子平面 → vwap→vwap
    远期收益 label（Ref(vwap,-label_days)/vwap-1）→ 每因子算 rank_ic/ic/icir/
    coverage/nan_inf_ratio，返回 metric bundle dict 列表（与 formula 一一对应）。

    数据无法构造时返回全 None bundle 并记录 degraded（不抛）。
    泄漏纪律：只读传入的 train 段 stock_data，不接收 test 段数据。

    .. note:: **legacy（仅 OFFLINE_TEST）**——生产评估只认 evaluator/QE 权威，
       本函数属本地 numpy/scipy 手算回退路径，PRODUCTION 模式由
       ``SearchPipeline`` 的 fail-closed 守卫禁止调用。
    """
    import numpy as np

    def _evaluate_fn(
        formulas: Sequence[str],
        fidelity: str = "L2_full_train",
        context: Any = None,
    ) -> list[dict[str, float | None]]:
        if stock_data is None:
            return [None] * len(formulas)
        fmls = [str(f) for f in formulas if f]
        try:
            planes = stock_data.evaluate_many(fmls)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - 数据不可用降级为全 None，不抛
            logger.warning("legacy.evaluators evaluate_many failed, degraded: %s", exc)
            return [None] * len(formulas)
        try:
            # vwap → 远期收益 label：Ref(vwap,-label_days)/vwap-1（后复权口径）
            names = list(stock_data._field_names())
            if "vwap" not in names:
                logger.warning("legacy.evaluators: no vwap field, degraded")
                return [None] * len(formulas)
            vwap = stock_data.data[:, :, names.index("vwap")]
            T = vwap.shape[0]
            if T <= label_days:
                logger.warning("legacy.evaluators: series too short for label_days, degraded")
                return [None] * len(formulas)
            label = vwap[label_days:, :] / vwap[:-label_days, :] - 1.0
        except Exception as exc:  # noqa: BLE001
            logger.warning("legacy.evaluators vwap label build failed, degraded: %s", exc)
            return [None] * len(formulas)
        # 因子平面与 label 对齐：factor 也用 [label_days:, :] 段
        bundles: list[dict[str, float | None]] = []
        for plane in planes:
            try:
                f_arr = np.asarray(plane, dtype=float)
            except Exception:  # noqa: BLE001
                f_arr = plane
            if len(f_arr.shape) == 3:
                # evaluate_many 返回 (T,N,F) 时取最后一维（F=1）
                f_arr = f_arr[:, :, -1]
            try:
                f_arr = f_arr[-label.shape[0]:, :]
            except Exception:  # noqa: BLE001
                pass
            bundles.append(_bundle_from_plane(f_arr, label))
        return bundles

    return _evaluate_fn


def _bundle_from_plane(
    plane: Any,
    label: Any,
) -> dict[str, float | None]:
    """单因子平面 × label → 基础 metric bundle（离线 numpy/scipy，不跑 qlib）。

    .. note:: **legacy（仅 OFFLINE_TEST）**——同 :func:`make_fe_evaluate_fn`，
       PRODUCTION 评估不消费本函数（无手算回退）。
    """
    try:
        import numpy as np
        from scipy import stats as _scipy_stats

        f = np.asarray(plane, dtype=float)
        y = np.asarray(label, dtype=float)
        T, N = f.shape
        if T == 0 or N == 0:
            return _all_none()
        # 对齐到有效 label 段
        f = f[-y.shape[0]:, :]
        T = f.shape[0]
        valid = np.isfinite(f) & np.isfinite(y)
        f_clean = np.where(valid, f, np.nan)
        cov = np.mean(np.isfinite(f), axis=1)
        coverage = float(np.mean(cov >= 0.5))
        nan_ratio = float(np.mean(~np.isfinite(f)))
        rankics: list[float] = []
        ics: list[float] = []
        for t in range(T):
            ft = f_clean[t]
            yt = y[t]
            mask = np.isfinite(ft) & np.isfinite(yt)
            if mask.sum() < 30:
                continue
            r = _scipy_stats.spearmanr(ft[mask], yt[mask], nan_policy="omit")
            if (
                r is not None
                and hasattr(r, "correlation")
                and r.correlation is not None
                and np.isfinite(r.correlation)
            ):
                rankics.append(float(r.correlation))
            c = _scipy_stats.pearsonr(ft[mask], yt[mask])
            if (
                c is not None
                and hasattr(c, "statistic")
                and c.statistic is not None
                and np.isfinite(c.statistic)
            ):
                ics.append(float(c.statistic))
        rank_ic = float(np.mean(rankics)) if rankics else None
        ic = float(np.mean(ics)) if ics else None
        icir = float(np.mean(rankics) / (np.std(rankics) + 1e-6)) if len(rankics) > 1 else None
        return {
            "rankic": rank_ic,
            "ic": ic,
            "icir": icir,
            "coverage": coverage,
            "nan_inf_ratio": nan_ratio,
            "untradeable_ratio": None,
        }
    except Exception as exc:  # noqa: BLE001 - scipy/numpy 缺失时降级全 None
        logger.warning("legacy.evaluators metric computation degraded: %s", exc)
        return _all_none()


def _all_none() -> dict[str, float | None]:
    return {
        "rankic": None,
        "ic": None,
        "icir": None,
        "coverage": None,
        "nan_inf_ratio": None,
        "untradeable_ratio": None,
    }


__all__ = [
    "make_fe_evaluate_fn",
    "_bundle_from_plane",
    "_all_none",
]
