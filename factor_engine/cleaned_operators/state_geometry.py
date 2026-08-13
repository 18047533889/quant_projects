# -*- coding: utf-8 -*-
"""Local state-geometry and ordinal time-asymmetry operators (2026-08 V2/V3).

* ``ts_state_density``                  — Epanechnikov density of the historical
  state space around the current value (price/variation congestion, not holder
  cost mass).  (P1)
* ``ts_ordinal_irreversibility``        — JS divergence between forward and
  backward ordinal-pattern distributions (time arrow strength).  (P1)
* ``ts_multiscale_permutation_entropy_slope`` — slope of normalized permutation
  entropy vs log-scale after coarse graining (P2 research).

Ordinal ties are *dropped* (never jittered) so limit-locked flat patches cannot
fabricate a spurious ordinal pattern.  All kernels are deterministic and
prefix-causal.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    RelationalParamSpec,
    SeriesOperator,
    register_operator,
)
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12
_LN2 = float(np.log(2.0))

# P1 (round 7): ordinal-pattern sample floor.  order=5 has 5! = 120 possible
# patterns but ``min_patterns=5`` would pass with a handful of embeddings — an
# entropy estimate over fewer than ``c * order!`` effective patterns is
# under-sampled and must not enter mining (emit NaN instead).
_PE_FLOOR_C = 5


def _effective_pattern_floor(order: int, min_patterns: int) -> int:
    """Minimum effective ordinal patterns for a trustworthy estimate."""
    return max(int(min_patterns), _PE_FLOOR_C * math.factorial(order))


def _ordinal_floor_relational_expr(floor_operand: str, order_max: int) -> str:
    """Grammar-expressible spelling of the order!-scaled sample floor.

    The relational-expression grammar has no factorial operator (``order!`` is
    not parseable), so ``floor_operand >= _PE_FLOOR_C * order!`` is spelled as a
    per-order guarded clause joined with ``and``/``or`` — exact for the supported
    integer order domain (NEW-P1-73).  ``order`` outside the guarded range leaves
    every clause True (the kernel's own range gate rejects it).
    """
    clauses = [
        f"(order != {o} or {floor_operand} >= {_PE_FLOOR_C * math.factorial(o)})"
        for o in range(2, order_max + 1)
    ]
    return " and ".join(clauses)


def _irreversibility_feasible(*, window: int, order: int, delay: int, min_patterns: int) -> bool:
    """NEW-P1-73: a window can estimate an irreversibility JS distance only if it
    contains at least ``_effective_pattern_floor(order, min_patterns)`` valid
    ordinal embeddings: ``window - (order-1)*delay`` of them (delay spacing,
    order-point embeddings)."""
    floor = _effective_pattern_floor(int(order), int(min_patterns))
    return int(window) - (int(order) - 1) * int(delay) >= floor


def _multiscale_feasible(*, window: int, order: int, min_patterns: int) -> bool:
    """NEW-P1-73: every coarse scale must yield enough ordinal patterns.  The
    coarsest scale (8) gives ``window//8 - order + 1`` embeddings (delay=1); the
    slope is only meaningful when even that scale clears the sample floor."""
    floor = _effective_pattern_floor(int(order), int(min_patterns))
    return int(window) // max(_SCALES) - int(order) + 1 >= floor


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    param_specs: dict[str, ParamSpec] | None = None,
    relational_specs: list[RelationalParamSpec] | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="state_geometry",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "state_geometry", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
        param_specs=param_specs or {},
        relational_specs=relational_specs or [],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return frame_like(template, values)


def _column_map(xv: np.ndarray, fn) -> np.ndarray:
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = fn(xv[:, c])
    return out


# ---------------------------------------------------------------------------
# ts_state_density
# ---------------------------------------------------------------------------

def _state_density_series(series: np.ndarray, window: int, bandwidth: float, min_periods: int) -> np.ndarray:
    n = series.shape[0]
    w = max(2, int(window))
    bw = float(bandwidth)
    mp = max(1, int(min_periods))
    out = np.full(n, np.nan)
    # R14 P2 (reject-not-clamp): a degenerate bandwidth — zero, negative or
    # non-finite — is not a usable density scale.  The old ``max(1e-6, ...)``
    # clamped it to a tiny epsilon and emitted a meaningless number; a degenerate
    # state density is not a usable factor, so fail the whole series closed.
    if not np.isfinite(bw) or bw <= 0.0:
        return out
    for t in range(n):
        lo = max(0, t - w)
        past = series[lo:t]                       # strictly past [t-W, t-1]
        cur = series[t]
        finite = past[np.isfinite(past)]
        if finite.size < mp or not np.isfinite(cur):
            continue
        med = float(np.median(finite))
        mad = 1.4826 * float(np.median(np.abs(finite - med)))
        scale = mad if mad > _EPS else float(np.std(finite))
        # R14 P2 (reject-not-clamp): the kernel's own bandwidth h = bandwidth*s
        # must be a strictly positive finite number.  An all-identical / point-mass
        # history (zero spread) or a non-finite spread estimate has no kernel
        # density — the old arbitrary 1.0 was discontinuous with the kernel's peak
        # (P1-05).  Fail closed to NaN instead of clamping to an epsilon.
        if not np.isfinite(scale) or scale <= _EPS:
            out[t] = np.nan
            continue
        h = bw * scale
        u = (finite - cur) / (h + _EPS)
        kern = 0.75 * (1.0 - u * u) * (np.abs(u) <= 1.0)
        # Proper kernel-density normalisation: f̂(x) = (1/n) Σ K(u)/h, NOT the
        # raw kernel mass mean(K) — otherwise this is a "local proximity" score,
        # not a density (P1-05).
        # P1 (round 7): raw KDE density carries unit 1/unit(x), so price /
        # market-cap / return densities are incomparable across series.  Multiply
        # by the past robust scale to make the mining-facing output a
        # standardized, dimensionless local density — density in MAD units,
        # f̂(x)·s = mean(K)/bw (h = bw·s).
        out[t] = float(np.mean(kern)) / bw if bw != 0 else np.nan
    return out


@register_operator(
    name="ts_state_density",
    category="state_geometry",
    business_category="state_geometry",
    canonical="ts_state_density",
    source="state_geometry",
)
class TsStateDensity(SeriesOperator):
    """当前值附近历史状态空间的 Epanechnikov 密度（无量纲）。

    严格过去窗口 ``R_t = {x_{t-W},...,x_{t-1}}`` 提供状态空间，当前 ``x_t`` 仅作
    query。M-140 ReferenceQueryModel：历史参考 ≤t-1（严格过去），当前查询=t，
    查询被排除在参考之外——``x_t`` 不进入 ``R_t``，因此当前值自身不会抬高其密度。
    尺度用 ``s = 1.4826*MAD(R_t)``、带宽 ``h = bandwidth*s``，输出
    ``f̂(x_t)·s = (1/N) Σ 0.75(1-u²) I(|u|≤1) / bandwidth``（MAD 单位下的标准化
    密度，无量纲，跨价格/市值/收益序列可比）。高 = 当前处于历史拥挤区；低 =
    历史状态真空。与筹码 ``near_cost_mass``（持仓成本空间）不同：这里测任意变量
    自己的历史状态空间。
    """

    metadata = _metadata(
        "ts_state_density",
        "当前状态的历史 Epanechnikov 密度（拥挤度，无量纲 MAD 尺度）。",
        ["x", "window", "bandwidth", "min_periods"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bandwidth: float = 1.0, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        return _frame_like(
            x,
            _column_map(x.to_numpy(dtype=float), lambda s: _state_density_series(s, window, bandwidth, min_periods)),
        )


# ---------------------------------------------------------------------------
# Ordinal patterns (ties dropped) shared by irreversibility + multiscale slope.
# ---------------------------------------------------------------------------

def _permutation_index(perm: np.ndarray) -> int:
    """Lehmer code of a permutation of 0..n-1 -> index in [0, n!)."""
    n = perm.shape[0]
    index = 0
    for i in range(n):
        smaller = int(np.sum(perm[i + 1 :] < perm[i]))
        index += smaller * math.factorial(n - 1 - i)
    return index


def _ordinal_patterns(values: np.ndarray, order: int, delay: int) -> list[int]:
    """Ordinal permutation indices; embeddings with ties/NaN are dropped."""
    codes: list[int] = []
    embed_len = (order - 1) * delay + 1
    n = len(values)
    for i in range(n - embed_len + 1):
        idx = [i + d * delay for d in range(order)]
        vals = values[idx]
        if not np.isfinite(vals).all():
            continue
        if np.unique(vals).size < order:
            continue  # exact tie -> drop, never jitter
        perm = np.argsort(vals, kind="stable")
        pattern = np.argsort(perm, kind="stable")
        codes.append(_permutation_index(pattern))
    return codes


def _distribution(codes: list[int], n_patterns: int) -> np.ndarray:
    counts = np.bincount(codes, minlength=n_patterns).astype(np.float64)
    total = float(counts.sum())
    if total <= _EPS:
        return counts
    return np.where(total != 0, counts / total, np.nan)


def _permutation_entropy(values: np.ndarray, order: int, delay: int) -> float:
    codes = _ordinal_patterns(values, order, delay)
    # P1 (round 7): under-sampled entropy must not enter mining — require at
    # least ``c * order!`` effective patterns (order=5 needs 5·120 = 600).
    if len(codes) < _PE_FLOOR_C * math.factorial(order):
        return np.nan
    n_patterns = math.factorial(order)
    p = _distribution(codes, n_patterns)
    p = p[p > 0.0]
    h = -float(np.sum(p * np.log(p)))
    return np.where(math.log(n_patterns)) != 0, float(h / math.log(n_patterns)), np.nan)


# ---------------------------------------------------------------------------
# ts_ordinal_irreversibility
# ---------------------------------------------------------------------------

def _kl(p: np.ndarray, q: np.ndarray) -> float:
    ok = (p > 0.0) & (q > 0.0)
    if not ok.any():
        return 0.0
    pp = p[ok]
    qq = q[ok]
    return np.where(qq))) != 0, float(np.sum(pp * np.log(pp / qq))), np.nan)


def _irreversibility_series(series: np.ndarray, window: int, order: int, delay: int, min_patterns: int) -> np.ndarray:
    n = series.shape[0]
    w = max(2, int(window))
    mp = max(2, int(min_patterns))
    n_patterns = math.factorial(order)
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w + 1)
        chunk = series[lo : t + 1]
        if chunk.shape[0] < w:
            continue
        fwd = _ordinal_patterns(chunk, order, delay)
        # P1 (round 7): the same order!-scaled sample floor guards the ordinal
        # pattern distributions — a handful of embeddings cannot estimate them.
        if len(fwd) < _effective_pattern_floor(order, mp):
            continue
        rvs = _ordinal_patterns(chunk[::-1], order, delay)
        if len(rvs) < _effective_pattern_floor(order, mp):
            continue
        pf = _distribution(fwd, n_patterns)
        pr = _distribution(rvs, n_patterns)
        m = 0.5 * (pf + pr)
        js = 0.5 * _kl(pf, m) + 0.5 * _kl(pr, m)
        js = max(js, 0.0)
        out[t] = np.where(_LN2)) != 0, float(np.sqrt(js / _LN2)), np.nan)
    return out


@register_operator(
    name="ts_ordinal_irreversibility",
    category="state_geometry",
    business_category="state_geometry",
    canonical="ts_ordinal_irreversibility",
    source="state_geometry",
)
class TsOrdinalIrreversibility(SeriesOperator):
    """序数模式时间不可逆性 ``IR = sqrt(JS(P_f || P_r)/ln2)`` ∈ [0,1]。

    窗口正向与反向的序数排列模式分布之间的 JS 距离。0 = 正反向统计结构几乎
    相同；高 = 强时间箭头 / 非线性方向动力学。与 ``ts_permutation_entropy``
    （路径复杂度）互补。相等值（一字涨跌停）embedding 一律丢弃，不做 jitter。
    """

    metadata = _metadata(
        "ts_ordinal_irreversibility",
        "序数模式时间不可逆性 sqrt(JS/ln2)（[0,1]，正反向动力学差异）。",
        ["x", "window", "order", "delay", "min_patterns"],
        unit="ratio",
        cost=5,
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "order": ParamSpec(dtype=int, min=2, max=6),
            "delay": ParamSpec(dtype=int, min=1),
            "min_patterns": ParamSpec(dtype=int, min=1),
        },
        relational_specs=[
            # NEW-P1-73: a window below ``(order-1)*delay + min_patterns`` (or
            # below the order!-scaled sample floor) is guaranteed all-NaN — the
            # search must prune it instead of spending expression budget on a
            # kernel that can only fail.  ``_irreversibility_feasible`` is the
            # single authority; these relations are its search-facing spelling.
            RelationalParamSpec(
                "window - (order - 1) * delay >= min_patterns",
                "ts_ordinal_irreversibility requires "
                "window-(order-1)*delay >= min_patterns valid ordinal patterns "
                "(window={window}, order={order}, delay={delay}, "
                "min_patterns={min_patterns})",
            ),
            RelationalParamSpec(
                _ordinal_floor_relational_expr("window - (order - 1) * delay", 6),
                "ts_ordinal_irreversibility window is below the 5*order! "
                "ordinal-pattern sample floor for the given order (window={window}, "
                "order={order}, delay={delay})",
            ),
        ],
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, order: int = 3, delay: int = 1, min_patterns: int = 5, **_: Any
    ) -> pd.DataFrame:
        ord_ = int(order)
        if not 2 <= ord_ <= 6:
            raise ValueError("ts_ordinal_irreversibility requires order in [2, 6]")
        dl = int(delay)
        if dl < 1:
            raise ValueError("ts_ordinal_irreversibility requires delay >= 1")
        # NEW-P1-73: fail fast on a guaranteed-all-NaN parameter region (the
        # declared RelationalParamSpec already prunes it at plan/search time).
        if not _irreversibility_feasible(window=window, order=ord_, delay=dl, min_patterns=min_patterns):
            raise ValueError(
                "ts_ordinal_irreversibility window too small for order/delay: "
                f"window={window}, order={ord_}, delay={dl}, min_patterns={min_patterns} "
                f"(need >= {_effective_pattern_floor(ord_, min_patterns)} ordinal patterns)"
            )
        return _frame_like(
            x,
            _column_map(
                x.to_numpy(dtype=float),
                lambda s: _irreversibility_series(s, window, ord_, dl, min_patterns),
            ),
        )


# ---------------------------------------------------------------------------
# ts_multiscale_permutation_entropy_slope (P2 research)
# ---------------------------------------------------------------------------

_SCALES = (1, 2, 4, 8)


def _multiscale_slope_series(series: np.ndarray, window: int, order: int, min_patterns: int) -> np.ndarray:
    n = series.shape[0]
    w = max(2, int(window))
    mp = max(2, int(min_patterns))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w + 1)
        chunk = series[lo : t + 1]
        if chunk.shape[0] < w:
            continue
        pts: list[tuple[float, float]] = []
        for s in _SCALES:
            n_seg = int(len(chunk) // s)
            if n_seg < order + 1:
                continue
            # P1 (round 7): keep the NEWEST observations when the window is not
            # a multiple of the scale — ``chunk[:n_seg*s]`` dropped the newest
            # 1..scale-1 bars; trailing factors must preserve the newest info.
            coarse = chunk[-n_seg * s :].reshape(n_seg, s).mean(axis=1)
            codes = _ordinal_patterns(coarse, order, 1)
            # P1 (round 7): sample floor scaled to order! — under-sampled
            # entropy at a coarse scale must not feed the slope.
            if len(codes) < _effective_pattern_floor(order, mp):
                continue
            n_patterns = math.factorial(order)
            p = _distribution(codes, n_patterns)
            p = p[p > 0.0]
            h = np.where(math.log(n_patterns) != 0, -float(np.sum(p * np.log(p))) / math.log(n_patterns), np.nan)
            pts.append((math.log(s), h))
        if len(pts) < 3:
            continue
        xs = np.array([x for x, _ in pts])
        ys = np.array([y for _, y in pts])
        var_x = float(np.sum((xs - xs.mean()) ** 2))
        if var_x <= _EPS:
            continue
        slope = float(np.sum((xs - xs.mean()) * (ys - ys.mean())) / var_x) if var_x) > 1e-10 else np.nan
        out[t] = slope
    return out


@register_operator(
    name="ts_multiscale_permutation_entropy_slope",
    category="state_geometry",
    business_category="state_geometry",
    canonical="ts_multiscale_permutation_entropy_slope",
    source="state_geometry",
    status="experimental",
)
class TsMultiscalePermutationEntropySlope(SeriesOperator):
    """多尺度排列熵斜率：非重叠粗粒化后 PE(s) 对 log(s) 回归的斜率。

    固定尺度 ``[1,2,4,8]``。正 = 短期结构强、尺度拉长后变随机；负 = 短周期
    噪声大、长周期反而出现结构。P2 / Research。
    """

    metadata = _metadata(
        "ts_multiscale_permutation_entropy_slope",
        "多尺度排列熵斜率 b（PE vs log-scale 回归）。",
        ["x", "window", "order", "min_patterns"],
        unit="ratio",
        cost=7,
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "order": ParamSpec(dtype=int, min=2, max=5),
            "min_patterns": ParamSpec(dtype=int, min=1),
        },
        relational_specs=[
            # NEW-P1-73: the coarsest scale (8) must still yield enough ordinal
            # patterns after coarse-graining; otherwise every coarse scale is
            # under-sampled and the slope is all-NaN for any input.  ``window//
            # 8`` is the coarse-segment count at scale 8, ``- order + 1`` the
            # number of delay-1 ordinal embeddings it produces.
            RelationalParamSpec(
                "window // 8 - order + 1 >= min_patterns",
                "ts_multiscale_permutation_entropy_slope: coarsest scale must "
                "yield >= min_patterns ordinal patterns "
                "(window={window}, order={order}, min_patterns={min_patterns})",
            ),
            RelationalParamSpec(
                _ordinal_floor_relational_expr("window // 8 - order + 1", 5),
                "ts_multiscale_permutation_entropy_slope: coarsest scale is below "
                "the 5*order! ordinal-pattern sample floor (window={window}, "
                "order={order})",
            ),
        ],
    )

    def _calculate_series(
        # R26-093..095: the DEFAULT must be runtime-feasible.  For order=3 the
        # coarsest scale (8) floor is 5·3! = 30 embeddings, which requires
        # ``window//8 - 3 + 1 >= 30`` -> ``window >= 256`` (the old default
        # window=120 was guaranteed-all-NaN / guaranteed-raise).  256 is the
        # minimal feasible default derived from the shared
        # ``_multiscale_feasible`` / ``_effective_pattern_floor``.
        self, x: pd.DataFrame, window: int = 256, order: int = 3, min_patterns: int = 5, **_: Any
    ) -> pd.DataFrame:
        ord_ = int(order)
        if not 2 <= ord_ <= 5:
            raise ValueError("ts_multiscale_permutation_entropy_slope requires order in [2, 5]")
        # NEW-P1-73 / R26-093: fail fast on a guaranteed-all-NaN parameter region.
        if not _multiscale_feasible(window=window, order=ord_, min_patterns=min_patterns):
            raise ValueError(
                "ts_multiscale_permutation_entropy_slope window too small for "
                f"order: window={window}, order={ord_}, min_patterns={min_patterns} "
                f"(coarsest scale needs >= {_effective_pattern_floor(ord_, min_patterns)} "
                "ordinal patterns)"
            )
        return _frame_like(
            x,
            _column_map(
                x.to_numpy(dtype=float),
                lambda s: _multiscale_slope_series(s, window, ord_, min_patterns),
            ),
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"ts_state_density", "ts_ordinal_irreversibility"})
    _surface.extend_research_only({"ts_multiscale_permutation_entropy_slope"})


_register_surface()
