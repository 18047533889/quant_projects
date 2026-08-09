# -*- coding: utf-8 -*-
"""String / ordinal complexity operators (2026-08 geometry/math expansion).

Two complementary views of *how much structure* a trailing price window holds:

* ``ts_lempel_ziv_complexity`` — the series is symbolized by its *trailing PIT
  quantile* (each value mapped through the window's own empirical CDF into
  ``bins`` symbols), then the Lempel-Ziv 76 greedy-parse complexity ``c(N)`` is
  normalized by the random-string baseline ``N/log(N)`` (natural log).
* the forbidden-ordinal-pattern family (``ts_forbidden_ordinal_pattern_ratio`` /
  ``ts_forbidden_ordinal_pattern_excess`` /
  ``ts_forbidden_ordinal_pattern_signed_excess``) — the fraction of all ordinal
  patterns of a given length that *never* appear in the window
  (1 - distinct/factorial).

Both are trailing-window, prefix-causal and deterministic.  Ordinal embeddings
with ties are dropped (never jittered) so flat patches cannot fabricate a
pattern.  Windows too short to symbolise / embed emit ``NaN``; invalid
parameters raise ``ValueError``.

Honesty notes:
* ``ts_lempel_ziv_complexity`` is a *retrospective* window complexity (audit
  #84): each rolling window re-symbolizes its whole history with the window's
  own empirical CDF — it is NOT a persistent sequential symbolic process.  Its
  random normalization is an asymptotic baseline (audit #86); a validity floor
  (``min_effective_n``) and a suffix guard (``min_contiguous_fraction``) fail
  closed where the baseline or the comparison window is unreliable.
* The forbidden-ordinal family uses the finite-sample null
  ``(1 - 1/m!)^N`` which treats overlapping ordinal embeddings as independent
  — an approximation (audit #87) — and counts ONLY no-tie embeddings in the
  null baseline N (audit #88).

R11 round-2 honest naming (review items #20/#21): the historical canonical
``ts_forbidden_ordinal_pattern_ratio`` actually computed the finite-sample
*excess* ``max(0, F_obs - F_null)``, not the raw ratio.  The implementation was
migrated to the honest canonical ``ts_forbidden_ordinal_pattern_excess`` (the
old spelling is handled via ``rename_canonical``); the name
``ts_forbidden_ordinal_pattern_ratio`` is re-claimed by the true RAW ratio
canonical, and the unclipped signed-excess variant (``F_obs - F_null``, which
keeps the information in "fewer forbidden patterns than the null") lives at
``ts_forbidden_ordinal_pattern_signed_excess``.  All three share one
feasibility gate (review item #22): ``window >= (order-1)*delay + min_embeddings``
must hold or the grammar rejects the combination at binding.
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
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12

# R11 round-2 (review item #22): feasibility floor for ordinal-pattern
# embeddings.  A (window, order, delay) combination is only a *feasible*
# embedding when the window holds at least ``_MIN_EMBEDDINGS`` ordinal
# embeddings: ``window - (order-1)*delay >= _MIN_EMBEDDINGS``.  Below that the
# kernel is guaranteed to emit all-NaN (too few embeddings to estimate the
# forbidden-pattern fraction), so the grammar rejects the combination at
# binding instead of running then failing.
_MIN_EMBEDDINGS = 8


def _trailing_contiguous_finite(chunk: np.ndarray) -> np.ndarray:
    """Trailing contiguous finite suffix ending at the window's last row.

    Complexity operators must never re-connect values across a gap: after a
    recent missing value only the trailing contiguous finite block is a valid
    sample for the *current* complexity (review R4-62 / R4-64).  R5 P0-03: a
    NaN at the current row returns the empty block (fail-closed) so a missing
    value today never emits a stale-history complexity factor.
    """
    n = len(chunk)
    if n == 0 or not np.isfinite(chunk[-1]):
        return np.empty(0, dtype=float)
    i = n
    while i > 0 and np.isfinite(chunk[i - 1]):
        i -= 1
    return chunk[i:].astype(float)


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
        category="complexity_ext",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "complexity_ext", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:complexity",
            f"unit:{unit}", f"cost:{cost}",
        ],
        param_specs=param_specs or {},
        relational_specs=relational_specs or [],
    )


# ---------------------------------------------------------------------------
# Lempel-Ziv 76 greedy parsing
# ---------------------------------------------------------------------------
def _lz76_complexity(symbols: np.ndarray) -> float:
    """LZ76 complexity ``c(N)`` (greedy distinct-substring parsing).

    Implements the Kaspar-Schuster scanning procedure.  The sequence is parsed
    left-to-right; a new component is counted whenever the current extension can
    no longer be matched inside the already-parsed prefix.
    """
    n = len(symbols)
    if n == 0:
        return 0.0
    if n < 2:
        return 1.0
    c = 1          # phrase count
    l = 1          # length of parsed prefix
    i = 0          # scan pointer inside the prefix
    k = 1          # current extension length
    k_max = 1
    while True:
        if symbols[i + k - 1] == symbols[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
        if i + k > n:
            break
    return float(c)


def _symbolize_pit_trailing_quantile(chunk: np.ndarray, bins: int) -> np.ndarray | None:
    """Symbolize a window by its own empirical CDF (trailing PIT quantile).

    Each finite value ``v`` is mapped to ``floor( PIT(v) * bins )`` where
    ``PIT(v) = #{w in window : w <= v} / N``; the last bin is the closed one.
    Only the trailing contiguous finite suffix is symbolized (review R4-62):
    a gap must not splice [1,2,NaN,4] into [1,2,4].  Returns a length-N symbol
    array or ``None`` when fewer than 2 finite values.
    """
    v = _trailing_contiguous_finite(chunk)
    n = v.size
    if n < 2:
        return None
    order = np.sort(v)
    frac_le = np.searchsorted(order, v, side="right") / float(n)  # in (0, 1]
    symbols = np.minimum((frac_le * bins).astype(np.int64), bins - 1)
    return symbols.astype(np.int64)


def _lz_complexity_series(
    x2d: np.ndarray,
    window: int,
    bins: int,
    min_contiguous_fraction: float,
    min_effective_n: int,
) -> np.ndarray:
    """Retrospective-window LZ complexity over the trailing contiguous suffix.

    Audit #85: the trailing finite suffix can drop from a full window down to a
    handful of symbols after a recent gap.  A window whose ``effective_n`` falls
    below ``max(min_effective_n, ceil(min_contiguous_fraction * window))`` must
    NOT be compared with full windows — below that the parse length is a
    different regime and the normalized value is not comparable -> NaN.

    Audit #86: the random baseline ``N·ln(bins)/ln(N)`` is an ASYMPTOTIC null.
    For small ``N`` a random string's normalized complexity can sit far from 1.
    ``min_effective_n`` is the validity floor below which the asymptotic
    baseline is unreliable and the output fails closed to NaN; see the operator
    docstring for the honest documentation of this limitation.
    """
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, b = int(window), int(bins)
    frac = float(min_contiguous_fraction)
    if not np.isfinite(frac) or frac < 0.0:
        raise ValueError("min_contiguous_fraction must be in [0, 1]")
    min_n = max(2, int(min_effective_n))
    min_req = max(min_n, int(np.ceil(frac * w)))
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            symbols = _symbolize_pit_trailing_quantile(col[i0 : r + 1], b)
            if symbols is None:
                continue
            n = symbols.size
            if n < min_req:
                continue  # effective_n below the suffix / validity floor -> NaN
            c_lz = _lz76_complexity(symbols)
            # Normalize by the alphabet size (review R4-17): a random string over
            # ``bins`` symbols has c(N) ~ N·ln(bins)/ln(N), so C_norm =
            # c(N)·ln(N)/(N·ln(bins)) → 1 for random data.  The old c(N)·ln(N)/N
            # made different ``bins`` incomparable and the binary baseline ≠ 1.
            out[r, c] = c_lz * math.log(float(n)) / (float(n) * math.log(float(b)))
    return out


# ---------------------------------------------------------------------------
# Forbidden ordinal patterns
# ---------------------------------------------------------------------------
def _ordinal_pattern_code(vals: np.ndarray) -> tuple[int, ...] | None:
    """Stable ordinal pattern of ``vals`` or ``None`` when a tie is present."""
    u = np.unique(vals)
    if u.size != vals.size:
        return None
    order = np.argsort(vals, kind="stable")
    pattern = np.argsort(order, kind="stable")
    return tuple(int(x) for x in pattern)


def _forbidden_ordinal_ratio_series(
    x2d: np.ndarray,
    window: int,
    order: int,
    delay: int,
    min_embeddings: int = _MIN_EMBEDDINGS,
    mode: str = "excess",
) -> np.ndarray:
    """Rolling forbidden-ordinal-pattern statistic.

    ``mode`` selects the reported quantity (R11 round-2 review #20/#21):
      * ``"excess"``        — ``max(0, F_obs - F_null)``  (historical behavior);
      * ``"ratio"``         — the RAW ``F_obs = 1 - n_obs/fact`` (the true
                              "fraction of forbidden patterns");
      * ``"signed_excess"`` — the UNCLIPPED ``F_obs - F_null`` (keeps the
                              information in "fewer forbidden patterns than the
                              null", which the clip-to-0 erases).
    ``min_embeddings`` is the binding-time feasibility floor; the kernel itself
    additionally fails closed when fewer than 2 no-tie embeddings survive.
    """
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, o, d = int(window), int(order), int(delay)
    fact = float(math.factorial(o))
    embed_len = (o - 1) * d + 1
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            chunk = col[i0 : r + 1]
            run = _trailing_contiguous_finite(chunk)  # review R4-64
            n = run.size
            if n < embed_len:
                continue
            patterns: set[tuple[int, ...]] = set()
            n_emb = 0      # ALL embeddings (valid + tied)
            n_valid = 0    # embeddings that produced a no-tie pattern
            for start in range(n - embed_len + 1):
                vals = run[start + np.arange(o) * d]
                n_emb += 1
                code = _ordinal_pattern_code(vals)
                if code is not None:
                    n_valid += 1
                    patterns.add(code)
            if not patterns or n_valid < 2:
                continue
            n_obs = len(patterns)
            # Finite-sample baseline (review R4-63): with N valid embeddings of a
            # random series, E[F] = (1 - 1/fact)^N — so order=6 / window=120
            # (only ~115 embeddings vs 720 patterns) mechanically reports ~0.84
            # even for white noise.  Report the *excess* forbiddenness above the
            # random null: F_excess = F_obs - E[F_random | N, order].
            #
            # Audit #88: the null baseline N must count ONLY usable no-tie
            # embeddings (``n_valid``).  Ties are dropped from the observed
            # pattern set, so their count must NOT enter the null baseline — the
            # old code used ``n_emb`` (all embeddings), under-stating the null
            # when ties are frequent and over-reporting forbiddenness.
            #
            # Audit #87: ``(1 - 1/m!)^N`` treats the N ordinal embeddings as
            # independent, but overlapping embeddings are correlated, so it is an
            # APPROXIMATION.  For a production (strictly-validated) null use a
            # permutation/surrogate null; this extended factor documents the
            # approximation honestly instead of pretending exactness.
            f_obs = 1.0 - n_obs / fact
            f_null = float((1.0 - 1.0 / fact) ** n_valid)
            if mode == "ratio":
                out[r, c] = f_obs
            elif mode == "signed_excess":
                out[r, c] = f_obs - f_null
            else:  # excess (clipped, historical behavior)
                out[r, c] = max(0.0, f_obs - f_null)
    return out


def _ordinal_param_specs() -> dict[str, ParamSpec]:
    """Shared cross-parameter contract for the forbidden-ordinal family.

    ``min_embeddings`` carries a floor of ``_MIN_EMBEDDINGS`` (review item #22):
    below that many embeddings the kernel is guaranteed to emit all-NaN, so the
    binding gate rejects the combination instead.
    """
    return {
        "window": ParamSpec(dtype=int, min=2),
        "order": ParamSpec(dtype=int, min=2),
        "delay": ParamSpec(dtype=int, min=1),
        "min_embeddings": ParamSpec(dtype=int, min=_MIN_EMBEDDINGS),
    }


def _ordinal_relational_specs(name: str) -> list[RelationalParamSpec]:
    """Grammar-expressible feasibility gate for the ordinal family.

    ``window - (order-1)*delay`` is the number of ordinal embeddings a window of
    size ``window`` yields (delay spacing, order-point embeddings).  Requiring it
    to be >= ``min_embeddings`` is exactly the binding-time rejection of
    ``window >= (order-1)*delay + 1 + min_embeddings - 1`` (review item #22).
    """
    return [
        RelationalParamSpec(
            "window - (order - 1) * delay >= min_embeddings",
            f"{name} requires window-(order-1)*delay >= min_embeddings valid "
            f"ordinal embeddings (window={{window}}, order={{order}}, delay={{delay}}, "
            f"min_embeddings={{min_embeddings}})",
        ),
    ]


def _check_ordinal_feasibility(
    *,
    window: int,
    order: int,
    delay: int,
    min_embeddings: int,
) -> None:
    """Binding-time feasibility gate (review item #22).

    A (window, order, delay) combination with ``window < (order-1)*delay +
    min_embeddings`` forms a valid embedding yet yields all-NaN (too few
    embeddings).  Reject it at binding with a ValueError instead of running and
    emitting all-NaN.
    """
    o, d, me = int(order), int(delay), int(min_embeddings)
    need = (o - 1) * d + me
    if int(window) < need:
        raise ValueError(
            f"window too small for order/delay/min_embeddings: window={window}, "
            f"order={o}, delay={d}, min_embeddings={me} "
            f"(need window >= (order-1)*delay + min_embeddings = {need})"
        )


def _check_complexity_params(
    window: int,
    bins: int | None = None,
    order: int | None = None,
    delay: int | None = None,
    min_embeddings: int | None = None,
) -> None:
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")
    if bins is not None and int(bins) < 2:
        raise ValueError("bins must be >= 2")
    if order is not None:
        o = int(order)
        if o < 2:
            raise ValueError("order must be >= 2")
        if math.factorial(o) > 720:
            raise ValueError("order must satisfy factorial(order) <= 720")
    if delay is not None and int(delay) < 1:
        raise ValueError("delay must be >= 1")
    if min_embeddings is not None:
        me = int(min_embeddings)
        if me < _MIN_EMBEDDINGS:
            raise ValueError(f"min_embeddings must be >= {_MIN_EMBEDDINGS}")
        if order is not None and delay is not None:
            _check_ordinal_feasibility(window=w, order=order, delay=delay, min_embeddings=me)


@register_operator(
    name="ts_lempel_ziv_complexity",
    category="complexity_ext",
    business_category="complexity_ext",
    canonical="ts_lempel_ziv_complexity",
    source="complexity_ext",
)
class TsLempelZivComplexity(SeriesOperator):
    """Lempel-Ziv 76 复杂度（按 N·ln(bins)/ln(N) 归一）。

    先把窗口内每个值经"窗口自身经验 CDF"（trailing PIT 分位）符号化成 ``bins`` 个
    符号，再做 LZ76 贪心切分计数 ``c(N)``，输出 ``c(N)·ln(N)/(N·ln(bins))``。
    随机串 → 接近 1；结构化/周期串 → 明显更低；不同 ``bins`` 可比（R4-17）。P2。

    **回顾式窗口复杂度（audit #84）**：每个滚动窗口都用 *该窗口自身的经验 CDF*
    对整段历史重新符号化——这是一个"回顾式（retrospective）"复杂度，不是持续
    的逐点顺序符号化过程（每个历史点的符号并非用当时可见的严格过去 CDF 生成，
    而是用整窗的经验 CDF 事后重编码）。因此该因子描述"该窗口内的结构复杂度"，
    不应被解读为一种在线/增量的事件状态复杂度。

    **渐近归一化（audit #86）**：随机基线 ``N·ln(bins)/ln(N)`` 是渐近量；小 N
    时随机串的归一化复杂度可能明显偏离 1。``min_effective_n`` 是有效性下限——
    低于它基线不可靠，输出 NaN。``min_contiguous_fraction`` 防止尾随有限后缀
    从满窗掉到几个符号时被拿去做跨窗比较（audit #85）。
    """

    metadata = _metadata(
        "ts_lempel_ziv_complexity",
        "LZ76 回顾式窗口复杂度，按渐近随机基线 N·ln(bins)/ln(N) 归一（bins 可比）。",
        ["x", "window", "bins", "min_contiguous_fraction", "min_effective_n"],
        unit="ratio",
        cost=4,
    )
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2),
        "bins": ParamSpec(dtype=int, min=2, max=8),
        "min_contiguous_fraction": ParamSpec(dtype=float, min=0.0, max=1.0),
        "min_effective_n": ParamSpec(dtype=int, min=2),
    }

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        bins: int = 2,
        min_contiguous_fraction: float = 0.5,
        min_effective_n: int = 16,
        **_: Any,
    ) -> pd.DataFrame:
        _check_complexity_params(window, bins=bins)
        return frame_like(
            x,
            _lz_complexity_series(
                x.to_numpy(dtype=float),
                window,
                bins,
                min_contiguous_fraction,
                min_effective_n,
            ),
        )


# ---------------------------------------------------------------------------
# Forbidden-ordinal-pattern family (R11 round-2 honest naming, review #20/#21)
# ---------------------------------------------------------------------------
# The class below is the HISTORICAL ``ts_forbidden_ordinal_pattern_ratio``
# implementation.  Its actual output is the finite-sample EXCESS
# ``max(0, F_obs - F_null)`` (review item #20) — the "ratio" spelling was
# dishonest.  ``_register_ordinal_family`` migrates it to the honest canonical
# ``ts_forbidden_ordinal_pattern_excess`` via ``rename_canonical`` (the old
# spelling is left as a deprecated alias), then re-claims the name for the RAW
# ratio canonical and registers the signed-excess canonical alongside it.
@register_operator(
    name="ts_forbidden_ordinal_pattern_ratio",
    category="complexity_ext",
    business_category="complexity_ext",
    canonical="ts_forbidden_ordinal_pattern_ratio",
    source="complexity_ext",
)
class TsForbiddenOrdinalPatternRatio(SeriesOperator):
    """（迁移源）禁序模式**超额**：``max(0, F_excess)``。

    R11 round-2（review #20）：本算子的真实输出是有限样本超额
    ``max(0, F_obs - F_null)``，不是裸比例。``_register_ordinal_family`` 会把它
    rename 到诚实的 ``ts_forbidden_ordinal_pattern_excess``；旧拼写
    ``ts_forbidden_ordinal_pattern_ratio`` 在改名时作为 deprecated alias 保留，
    然后被新的裸比例 canonical 重新占用。

    ``F_obs = 1 - N_obs/factorial(order)``，N_obs 为窗口内出现过的（无并列）长度
    ``order``、延迟 ``delay`` 的序数模式种数，N_valid 为无并列的可用嵌入数（带
    并列的嵌入被丢弃且不进入零假设 N，audit #88）。减去的 ``(1-1/order!)^N_valid``
    是随机序列在 N_valid 个嵌入下"某模式从未出现"的期望（有限样本下限修正，
    R4-63）——order=6 / window=120 时随机基准 ≈0.84，因此裸 F_obs 会高估结构
    受限。F_excess>0 → 相对随机显著缺失序数模式。

    **近似性（audit #87）**：``(1-1/m!)^N`` 假定 N 个序数嵌入相互独立，但重叠
    嵌入是相关的，因此该零假设是近似值。生产级严格零假设应改用 permutation /
    surrogate null。

    **可行性门（review #22）**：``window >= (order-1)*delay + min_embeddings``
    不满足时在绑定期拒绝（raise），而不是运行后全 NaN。
    """

    metadata = _metadata(
        "ts_forbidden_ordinal_pattern_excess",
        "禁序模式比例超出随机零假设期望的部分（结构受限程度，裁剪到 >= 0）。",
        ["x", "window", "order", "delay", "min_embeddings"],
        unit="ratio",
        cost=4,
        param_specs=_ordinal_param_specs(),
        relational_specs=_ordinal_relational_specs("ts_forbidden_ordinal_pattern_excess"),
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        order: int = 3,
        delay: int = 1,
        min_embeddings: int = _MIN_EMBEDDINGS,
        **_: Any,
    ) -> pd.DataFrame:
        _check_complexity_params(window, order=order, delay=delay, min_embeddings=min_embeddings)
        return frame_like(
            x,
            _forbidden_ordinal_ratio_series(
                x.to_numpy(dtype=float), window, order, delay, min_embeddings, mode="excess"
            ),
        )


class _ForbiddenOrdinalPatternRatio(SeriesOperator):
    """（新 canonical）禁序模式**裸比例**：``F_obs = 1 - N_obs/factorial(order)``。

    窗口内从未出现的序数模式占全部 ``order!`` 种的比例（"禁序模式比例"的诚实
    语义，review #20）。不做有限样本零假设修正；要剔除随机基线的影响请使用
    ``ts_forbidden_ordinal_pattern_excess`` / ``_signed_excess``。

    **可行性门（review #22）**：``window >= (order-1)*delay + min_embeddings``
    不满足时在绑定期拒绝（raise）。
    """

    metadata = _metadata(
        "ts_forbidden_ordinal_pattern_ratio",
        "禁序模式裸比例 F_obs = 1 - N_obs/order!（窗口内从未出现的序数模式占比）。",
        ["x", "window", "order", "delay", "min_embeddings"],
        unit="ratio",
        cost=4,
        param_specs=_ordinal_param_specs(),
        relational_specs=_ordinal_relational_specs("ts_forbidden_ordinal_pattern_ratio"),
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        order: int = 3,
        delay: int = 1,
        min_embeddings: int = _MIN_EMBEDDINGS,
        **_: Any,
    ) -> pd.DataFrame:
        _check_complexity_params(window, order=order, delay=delay, min_embeddings=min_embeddings)
        return frame_like(
            x,
            _forbidden_ordinal_ratio_series(
                x.to_numpy(dtype=float), window, order, delay, min_embeddings, mode="ratio"
            ),
        )


class _ForbiddenOrdinalPatternSignedExcess(SeriesOperator):
    """（新 canonical）禁序模式**符号超额**：``F_obs - F_null``（不裁剪）。

    review #21：裁剪到 0 抹掉了"禁序模式少于随机零假设"这一信息——负超额携带
    结构信息（该窗口的模式多样性 *超过* 随机期望，即模式"缺失"少于零假设）。
    本算子是 mining 偏好变体；正/负方向都保留。

    **可行性门（review #22）**：``window >= (order-1)*delay + min_embeddings``
    不满足时在绑定期拒绝（raise）。
    """

    metadata = _metadata(
        "ts_forbidden_ordinal_pattern_signed_excess",
        "禁序模式符号超额 F_obs - F_null（不裁剪，负值=模式缺失少于随机零假设）。",
        ["x", "window", "order", "delay", "min_embeddings"],
        unit="ratio",
        cost=4,
        param_specs=_ordinal_param_specs(),
        relational_specs=_ordinal_relational_specs("ts_forbidden_ordinal_pattern_signed_excess"),
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        order: int = 3,
        delay: int = 1,
        min_embeddings: int = _MIN_EMBEDDINGS,
        **_: Any,
    ) -> pd.DataFrame:
        _check_complexity_params(window, order=order, delay=delay, min_embeddings=min_embeddings)
        return frame_like(
            x,
            _forbidden_ordinal_ratio_series(
                x.to_numpy(dtype=float), window, order, delay, min_embeddings, mode="signed_excess"
            ),
        )


_NEW_CANONICALS = (
    "ts_lempel_ziv_complexity",
    "ts_forbidden_ordinal_pattern_excess",
    "ts_forbidden_ordinal_pattern_ratio",
    "ts_forbidden_ordinal_pattern_signed_excess",
)


def _register_ordinal_family() -> None:
    """R11 round-2 honest-naming migration for the forbidden-ordinal family.

    Review #20: the historical ``ts_forbidden_ordinal_pattern_ratio`` actually
    computed the finite-sample *excess* ``max(0, F_obs - F_null)``.  The
    implementation is migrated to the honest canonical
    ``ts_forbidden_ordinal_pattern_excess`` via ``rename_canonical`` (the old
    spelling is kept as a deprecated alias by that call).  The name is then
    re-claimed by the new RAW-ratio canonical, and the signed-excess canonical
    (review #21) is registered alongside.  The raw/signed classes are defined
    WITHOUT the ``@register_operator`` decorator because the historical name is
    still occupied at class-definition time; they are registered here, after the
    rename has freed the name.
    """
    from cleaned_operators.registry import OperatorRegistry

    if "ts_forbidden_ordinal_pattern_ratio" in OperatorRegistry._operators:
        OperatorRegistry.rename_canonical(
            "ts_forbidden_ordinal_pattern_ratio",
            "ts_forbidden_ordinal_pattern_excess",
        )
        # Re-claim the honest name for the raw-ratio canonical: the deprecated
        # alias edge ``rename_canonical`` left behind must not block a canonical
        # registration under the same name (registry invariant — a name is
        # either an alias or a canonical, never both).
        OperatorRegistry._aliases.pop("ts_forbidden_ordinal_pattern_ratio", None)
        _cat = OperatorRegistry._catalog.get("ts_forbidden_ordinal_pattern_excess")
        if _cat and "aliases" in _cat:
            _cat["aliases"] = [
                a for a in _cat["aliases"] if a != "ts_forbidden_ordinal_pattern_ratio"
            ]
    if "ts_forbidden_ordinal_pattern_ratio" not in OperatorRegistry._operators:
        OperatorRegistry.register(
            _ForbiddenOrdinalPatternRatio(),
            canonical="ts_forbidden_ordinal_pattern_ratio",
            backend="pandas_numpy",
            source="complexity_ext",
            status="implemented",
            backend_explicit=True,
        )
    if "ts_forbidden_ordinal_pattern_signed_excess" not in OperatorRegistry._operators:
        OperatorRegistry.register(
            _ForbiddenOrdinalPatternSignedExcess(),
            canonical="ts_forbidden_ordinal_pattern_signed_excess",
            backend="pandas_numpy",
            source="complexity_ext",
            status="implemented",
            backend_explicit=True,
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _register_ordinal_family()
    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
