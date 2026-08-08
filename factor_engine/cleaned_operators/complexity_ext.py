# -*- coding: utf-8 -*-
"""String / ordinal complexity operators (2026-08 geometry/math expansion).

Two complementary views of *how much structure* a trailing price window holds:

* ``ts_lempel_ziv_complexity`` — the series is symbolized by its *trailing PIT
  quantile* (each value mapped through the window's own empirical CDF into
  ``bins`` symbols), then the Lempel-Ziv 76 greedy-parse complexity ``c(N)`` is
  normalized by the random-string baseline ``N/log(N)`` (natural log).
* ``ts_forbidden_ordinal_pattern_ratio`` — the fraction of all ordinal patterns
  of a given length that *never* appear in the window (1 - distinct/factorial).

Both are trailing-window, prefix-causal and deterministic.  Ordinal embeddings
with ties are dropped (never jittered) so flat patches cannot fabricate a
pattern.  Windows too short to symbolise / embed emit ``NaN``; invalid
parameters raise ``ValueError``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12


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


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
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


def _lz_complexity_series(x2d: np.ndarray, window: int, bins: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, b = int(window), int(bins)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            symbols = _symbolize_pit_trailing_quantile(col[i0 : r + 1], b)
            if symbols is None:
                continue
            n = symbols.size
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


def _forbidden_ordinal_ratio_series(x2d: np.ndarray, window: int, order: int, delay: int) -> np.ndarray:
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
            n_emb = 0
            for start in range(n - embed_len + 1):
                vals = run[start + np.arange(o) * d]
                n_emb += 1
                code = _ordinal_pattern_code(vals)
                if code is not None:
                    patterns.add(code)
            if not patterns or n_emb < 1:
                continue
            n_obs = len(patterns)
            # Finite-sample baseline (review R4-63): with n_emb embeddings of a
            # random series, E[F] = (1 - 1/fact)^n_emb — so order=6 / window=120
            # (only ~115 embeddings vs 720 patterns) mechanically reports ~0.84
            # even for white noise.  Report the *excess* forbiddenness above the
            # random null: F_excess = F_obs - E[F_random | n_emb, order].
            f_obs = 1.0 - n_obs / fact
            f_null = float((1.0 - 1.0 / fact) ** n_emb)
            out[r, c] = max(0.0, f_obs - f_null)
    return out


def _check_complexity_params(window: int, bins: int | None = None, order: int | None = None, delay: int | None = None) -> None:
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
    """

    metadata = _metadata(
        "ts_lempel_ziv_complexity",
        "LZ76 贪心解析复杂度，按 N·ln(bins)/ln(N) 归一（随机性程度，bins 可比）。",
        ["x", "window", "bins"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, bins: int = 2, **_: Any) -> pd.DataFrame:
        _check_complexity_params(window, bins=bins)
        return frame_like(x, _lz_complexity_series(x.to_numpy(dtype=float), window, bins))


@register_operator(
    name="ts_forbidden_ordinal_pattern_ratio",
    category="complexity_ext",
    business_category="complexity_ext",
    canonical="ts_forbidden_ordinal_pattern_ratio",
    source="complexity_ext",
)
class TsForbiddenOrdinalPatternRatio(SeriesOperator):
    """禁序模式比例（随机零假设超额）：``F_excess = F_obs - (1-1/order!)^N_emb``。

    ``F_obs = 1 - N_obs/factorial(order)``，N_obs 为窗口内出现过的（无并列）长度
    ``order``、延迟 ``delay`` 的序数模式种数，N_emb 为有效嵌入数。减去的
    ``(1-1/order!)^N_emb`` 是随机序列在 N_emb 个嵌入下"某模式从未出现"的期望
    （有限样本下限修正，R4-63）——order=6 / window=120 时随机基准 ≈0.84，因此裸
    F_obs 会高估结构受限。F_excess>0 → 相对随机显著缺失序数模式。P2。
    """

    metadata = _metadata(
        "ts_forbidden_ordinal_pattern_ratio",
        "禁序模式比例超出随机零假设期望的部分（结构受限程度）。",
        ["x", "window", "order", "delay"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, order: int = 3, delay: int = 1, **_: Any) -> pd.DataFrame:
        _check_complexity_params(window, order=order, delay=delay)
        return frame_like(x, _forbidden_ordinal_ratio_series(x.to_numpy(dtype=float), window, order, delay))


_NEW_CANONICALS = (
    "ts_lempel_ziv_complexity",
    "ts_forbidden_ordinal_pattern_ratio",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
