"""FactorFitness V2 —— calibration（U 与 MetricCalibrator 衔接）。

所有 metric 先过 MetricCalibrator（现有 calibration/warmup ordinal + freeze z-score
体系）。U() 直接用 calibrator 输出的 utility。

与 fitness/__init__.py 的现有 utility() 衔接：保留公共入口，内部走 V2。

本文件只做「键 ↔ lower_is_better 语义 → calibrator.utility」的映射与批处理；
不实现 calibrator 本身（那是 fitness/__init__.py 的 MetricCalibrator）。
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from alphaprobe.fitness.contracts import QE_LOWER_IS_BETTER, EvaluationBundle

#: 键 → lower_is_better 的补充覆盖（可注入，优先于 QE_LOWER_IS_BETTER 默认表）
LowerIsBetterMap = Mapping[str, bool]
#: U 函数签名：U(metric_key, value, lower_is_better) -> float ∈ [0, 1]
UtilityFn = Callable[[str, Any, bool], float]


def utility_for_bundle(
    bundle: EvaluationBundle,
    calibrator: Any,
    *,
    keys: Iterable[str],
    lower_is_better: LowerIsBetterMap | None = None,
    utility: UtilityFn | None = None,
) -> dict[str, float]:
    """对 bundle 里的一批 metric 键跑 calibrator.utility，返回 {key: u}。

    - ``calibrator``：fitness.MetricCalibrator 实例（或任何带
      ``utility(metric, value, lower_is_better=...)`` 的对象）。
    - 未观测/缺失/非有限值 → utility 0.0（与 MetricCalibrator 行为一致）。
    - 解析规则：显式 ``lower_is_better`` 覆盖 > QE_LOWER_IS_BETTER 默认表 > False。
    """
    if utility is None:
        utility = _default_utility(calibrator)
    lib = {k: True for k in QE_LOWER_IS_BETTER}
    if lower_is_better:
        lib.update(lower_is_better)
    out: dict[str, float] = {}
    for k in keys:
        v = bundle.get(k)
        if v is None or not _finite(v):
            out[k] = 0.0
        else:
            out[k] = utility(str(k), v, bool(lib.get(str(k), False)))
    return out


def single_utility(
    calibrator: Any,
    key: str,
    value: Any,
    *,
    lower_is_better: bool | None = None,
) -> float:
    """单个 metric → utility。lower_is_better=None → 按 QE_LOWER_IS_BETTER 默认表。"""
    if lower_is_better is None:
        lower_is_better = str(key) in QE_LOWER_IS_BETTER
    if value is None or not _finite(value):
        return 0.0
    return _default_utility(calibrator)(str(key), value, bool(lower_is_better))


def _default_utility(calibrator: Any) -> UtilityFn:
    def _u(key: str, value: Any, lower: bool) -> float:
        try:
            u = calibrator.utility(key, value, lower_is_better=lower)
        except (TypeError, ValueError):
            return 0.0
        if not _finite(u):
            return 0.0
        return max(0.0, min(1.0, float(u)))

    return _u


def _finite(v: Any) -> bool:
    import math

    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False
