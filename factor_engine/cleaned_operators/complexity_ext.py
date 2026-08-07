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
    Returns a length-N symbol array or ``None`` when fewer than 2 finite values.
    """
    v = chunk[np.isfinite(chunk)]
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
            out[r, c] = c_lz * math.log(float(n)) / float(n)
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
            n = chunk.size
            if n < embed_len:
                continue
            patterns: set[tuple[int, ...]] = set()
            for start in range(n - embed_len + 1):
                vals = chunk[start + np.arange(o) * d]
                if not np.isfinite(vals).all():
                    continue
                code = _ordinal_pattern_code(vals)
                if code is not None:
                    patterns.add(code)
            if not patterns:
                continue
            n_obs = len(patterns)
            out[r, c] = 1.0 - n_obs / fact
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
    """Lempel-Ziv 76 复杂度（按 N/log(N) 归一）。

    先把窗口内每个值经"窗口自身经验 CDF"（trailing PIT 分位）符号化成 ``bins`` 个
    符号，再做 LZ76 贪心切分计数 ``c(N)``，输出 ``c(N)·log(N)/N``。随机串 → 接近 1；
    结构化/周期串 → 明显更低。P2。
    """

    metadata = _metadata(
        "ts_lempel_ziv_complexity",
        "LZ76 贪心解析复杂度，按 N/log(N) 归一（随机性程度）。",
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
    """禁序模式比例：``F = 1 - N_obs/factorial(order)``。

    N_obs 为窗口内出现过的（无并列）长度 ``order``、延迟 ``delay`` 的序数模式种数。
    F 高 → 大量序数模式缺失（结构受限 / 强有序）；F≈0 → 接近随机。P2。
    """

    metadata = _metadata(
        "ts_forbidden_ordinal_pattern_ratio",
        "1 - 观测序数模式种数 / order!（结构受限程度）。",
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
