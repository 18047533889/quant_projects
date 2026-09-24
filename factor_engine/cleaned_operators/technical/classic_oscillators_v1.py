# -*- coding: utf-8 -*-
"""Classic stochastic oscillators (R67): KDJ / Stochastic %K-%D / Williams %R.

Daily wide panels (columns = instruments, index = dates) in, daily wide panels
out.  Six causal, deterministic, PIT-safe operators are added here because the
DSL surface had **no** KDJ (three-line stochastic) and no usable Williams %R
(``StochasticK`` / ``StochasticD`` / ``WilliamsR`` in ``technical/signal.py``
are unregistered by the dedupe/overhaul passes and never reach the catalog).

Operators (canonical -> DSL name)
---------------------------------
  1. ``kdj_k``     %K of the KDJ family
  2. ``kdj_d``     %D of the KDJ family
  3. ``kdj_j``     %J of the KDJ family
  4. ``stoch_k``   fast stochastic %K (== RSV)
  5. ``stoch_d``   slow stochastic %D (SMA of %K)
  6. ``williams_r`` Williams %R

Formulas (``close`` / ``high`` / ``low`` are the argument order, see below)
--------------------------------------------------------------------------
    RSV = 100 * (close - LLV(low, N)) / (HHV(high, N) - LLV(low, N))
    K   = 2/3 * K_prev + 1/3 * RSV          (seed: K = RSV)
    D   = 2/3 * D_prev + 1/3 * K            (seed: D = K)
    J   = 3K - 2D
    %K  = RSV
    %D  = SMA(%K, m)
    %R  = -100 * (HHV(high, N) - close) / (HHV(high, N) - LLV(low, N))

NaN / degenerate-window policy (matched to the repo's LLV/HHV analogues
``ts_min`` / ``ts_max``, which use ``rolling(..., min_periods=1)``):
  * the rolling extrema and the %D SMA use ``min_periods=1`` — partial warmup
    windows emit a value (textbook TDX ``LLV``/``HHV`` behaviour), and NaN only
    ever comes from NaN in the inputs;
  * a zero denominator (``HHV == LLV``, a run of one-price limit bars) yields
    ``RSV = 100``; hence ``%R = RSV - 100 = 0`` on the same bars;
  * a row with a NaN ``close`` cannot be scored and stays NaN (NaN ``high`` /
    ``low`` are skipped inside the window like the rolling extrema do);
  * the recursive K/D smoothing carries the previous state through a NaN gap
    (``ewm`` ``ignore_na=False``), so K/D/J are only NaN before the first seed.

The recursive K/D smoothing is exactly ``pandas.ewm(alpha=1/3, adjust=False,
min_periods=1)`` (NaN rows carry the previous state — the pandas default
``ignore_na=False``), which the DuckDB SQL emitter's ``_ewm_adjust_false_sql``
reproduces bit-for-bit.

Registration follows the ``technical_signal`` family convention (see
``technical/signal.py`` / ``technical/new_indicators.py``): ``register_operator``
on a ``SeriesOperator`` subclass, canonical = operator name, source =
``technical.classic_oscillators_v1``.  Each canonical is put on the extended
partition, promoted onto the reviewed daily surface (``register_daily_migration``)
and given an exact-parity polars bridge.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)

_CANONICALS: list[str] = []

# KDJ / Stochastic smoothing weights: K = 2/3 K_prev + 1/3 RSV.
_KDJ_ALPHA = 1.0 / 3.0


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _meta(
    name: str,
    description: str,
    params: list[str],
    *,
    param_specs: dict[str, ParamSpec],
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="technical_signal",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "technical_signal", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:technical_signal",
            "unit:level", "cost:1",
        ],
        param_specs=param_specs,
    )


def _window_specs(name: str) -> dict[str, ParamSpec]:
    return {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
    }


def _rsv(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, window: int) -> pd.DataFrame:
    """Raw stochastic value in [0, 100]; 100 on a zero-range (one-price) window.

    ``min_periods=1`` mirrors the repo's ``ts_min`` / ``ts_max`` rolling extrema
    (textbook ``LLV``/``HHV``): partial windows emit a value, so no artificial
    NaN warmup is introduced.  A row with a NaN ``close`` cannot be scored and
    stays NaN; NaN ``high``/``low`` are simply skipped inside the window, exactly
    like the rolling extrema / the SQL ``_inst_window`` aggregate.
    """
    w = max(2, int(window))
    llv = low.rolling(window=w, min_periods=1).min()
    hhv = high.rolling(window=w, min_periods=1).max()
    denom = hhv - llv
    with np.errstate(divide="ignore", invalid="ignore"):
        rsv = 100.0 * (close - llv) / denom.where(denom != 0)
    # a zero-range window is a one-price run: no intraday range, RSV pinned to
    # 100 (mainstream convention), never a fabricated NaN/0.
    rsv = rsv.mask(denom == 0, 100.0)
    return rsv.where(close.notna())


def _ewm_kdj(series: pd.DataFrame) -> pd.DataFrame:
    """``2/3 * prev + 1/3 * cur`` with the first value seeded from ``cur``.

    Equivalent to ``ewm(alpha=1/3, adjust=False, min_periods=1)``.  A NaN input
    row carries the previous state forward (pandas' default ``ignore_na=False``
    semantics), which the DuckDB emitter's ``_ewm_adjust_false_sql`` mirrors.
    """
    return series.ewm(alpha=_KDJ_ALPHA, adjust=False, min_periods=1).mean()


def _kdj_triple(
    close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, window: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rsv = _rsv(close, high, low, window)
    k = _ewm_kdj(rsv)
    d = _ewm_kdj(k)
    j = 3.0 * k - 2.0 * d
    return k, d, j


# ---------------------------------------------------------------------------
# 1-3. KDJ (K / D / J)
# ---------------------------------------------------------------------------
class _KDJBase(SeriesOperator):
    _output = "k"

    def _calculate_series(
        self,
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        window: int = 9,
        **_: Any,
    ) -> pd.DataFrame:
        if not isinstance(close, pd.DataFrame):  # pragma: no cover - polars bridge
            close = close.to_pandas()
        k, d, j = _kdj_triple(close, high, low, int(window))
        if self._output == "k":
            return k
        if self._output == "d":
            return d
        return j


@register_operator(
    name="kdj_k",
    category="technical_signal",
    business_category="technical_signal",
    canonical="kdj_k",
    source="technical.classic_oscillators_v1",
)
class KDJK(_KDJBase):
    """KDJ %K: Wilder-style smoothed raw stochastic value.

    ``RSV = 100*(close - LLV(low, N)) / (HHV(high, N) - LLV(low, N))`` with
    ``min_periods=1`` rolling extrema and ``RSV = 100`` on a zero-range window;
    ``K = ewm(alpha=1/3, adjust=False)`` of ``RSV`` (first value = RSV).
    Argument order is ``(close, high, low, window)``; N defaults to 9.
    """

    _output = "k"
    metadata = _meta(
        "kdj_k",
        "KDJ 随机指标 %K：RSV 的 1/3-权重递推平滑（首值取 RSV）。",
        ["close", "high", "low", "window"],
        param_specs=_window_specs("window"),
    )


@register_operator(
    name="kdj_d",
    category="technical_signal",
    business_category="technical_signal",
    canonical="kdj_d",
    source="technical.classic_oscillators_v1",
)
class KDJD(_KDJBase):
    """KDJ %D: 1/3-weight smoothing of %K (first value = %K).

    Argument order is ``(close, high, low, window)``; N defaults to 9.
    """

    _output = "d"
    metadata = _meta(
        "kdj_d",
        "KDJ 随机指标 %D：%K 的同权重递推平滑（首值取 %K）。",
        ["close", "high", "low", "window"],
        param_specs=_window_specs("window"),
    )


@register_operator(
    name="kdj_j",
    category="technical_signal",
    business_category="technical_signal",
    canonical="kdj_j",
    source="technical.classic_oscillators_v1",
)
class KDJJ(_KDJBase):
    """KDJ %J = 3*%K - 2*%D.

    Argument order is ``(close, high, low, window)``; N defaults to 9.
    """

    _output = "j"
    metadata = _meta(
        "kdj_j",
        "KDJ 随机指标 %J：3*%K - 2*%D。",
        ["close", "high", "low", "window"],
        param_specs=_window_specs("window"),
    )


# ---------------------------------------------------------------------------
# 4-5. Stochastic %K / %D
# ---------------------------------------------------------------------------
@register_operator(
    name="stoch_k",
    category="technical_signal",
    business_category="technical_signal",
    canonical="stoch_k",
    source="technical.classic_oscillators_v1",
)
class StochK(SeriesOperator):
    """Fast stochastic %K == RSV (no smoothing).

    ``%K = 100*(close - LLV(low, N)) / (HHV(high, N) - LLV(low, N))`` with
    ``min_periods=1`` rolling extrema and ``%K = 100`` on a zero-range window.
    Argument order is ``(close, high, low, window)``; N defaults to 9.
    """

    metadata = _meta(
        "stoch_k",
        "随机指标快线 %K（即 RSV）。",
        ["close", "high", "low", "window"],
        param_specs=_window_specs("window"),
    )

    def _calculate_series(
        self,
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        window: int = 9,
        **_: Any,
    ) -> pd.DataFrame:
        if not isinstance(close, pd.DataFrame):  # pragma: no cover - polars bridge
            close = close.to_pandas()
        return _rsv(close, high, low, int(window))


@register_operator(
    name="stoch_d",
    category="technical_signal",
    business_category="technical_signal",
    canonical="stoch_d",
    source="technical.classic_oscillators_v1",
)
class StochD(SeriesOperator):
    """Slow stochastic %D = SMA(%K, m) with ``min_periods=1``.

    Argument order is ``(close, high, low, window, smooth)``; N defaults to 9 and
    the smoothing length m defaults to 3.
    """

    metadata = _meta(
        "stoch_d",
        "随机指标慢线 %D：%K 的 m 周期简单均线。",
        ["close", "high", "low", "window", "smooth"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
            "smooth": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self,
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        window: int = 9,
        smooth: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        if not isinstance(close, pd.DataFrame):  # pragma: no cover - polars bridge
            close = close.to_pandas()
        k = _rsv(close, high, low, int(window))
        m = max(1, int(smooth))
        return k.rolling(window=m, min_periods=1).mean()


# ---------------------------------------------------------------------------
# 6. Williams %R
# ---------------------------------------------------------------------------
@register_operator(
    name="williams_r",
    category="technical_signal",
    business_category="technical_signal",
    canonical="williams_r",
    source="technical.classic_oscillators_v1",
)
class WilliamsR(SeriesOperator):
    """Williams %R in [-100, 0] == RSV - 100.

    ``%R = -100*(HHV(high, N) - close) / (HHV(high, N) - LLV(low, N))``; a
    zero-range window yields 0 (the consistent ``RSV = 100`` limit).
    Argument order is ``(close, high, low, window)``; N defaults to 14.
    """

    metadata = _meta(
        "williams_r",
        "威廉指标 %R：-100*(HHV(high,N)-close)/(HHV-LLV)，值域 [-100, 0]。",
        ["close", "high", "low", "window"],
        param_specs=_window_specs("window"),
    )

    def _calculate_series(
        self,
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        window: int = 14,
        **_: Any,
    ) -> pd.DataFrame:
        if not isinstance(close, pd.DataFrame):  # pragma: no cover - polars bridge
            close = close.to_pandas()
        return _rsv(close, high, low, int(window)) - 100.0


_CANONICALS.extend(
    ["kdj_k", "kdj_d", "kdj_j", "stoch_k", "stoch_d", "williams_r"]
)


def _register_surface() -> None:
    """Extended partition + reviewed daily migration + polars bridge."""
    import factor_engine.cleaned_operators.operator_surface as _surface
    from factor_engine.cleaned_operators.rolling_pack import register_polars_bridge

    # layer_governance's static partition check requires every daily-migrated
    # canonical to also live in a partition set (DAILY_FACTOR_MIGRATED is a
    # classification refinement, not a partition).
    _surface.extend_extended_only(set(_CANONICALS))

    # Forward path onto the reviewed daily authoring surface.
    for _canon in _CANONICALS:
        _surface.register_daily_migration(
            _canon,
            review_id=f"R67-{_canon}",
            semantic_hash=f"r67-{_canon.lower()}-v1",
        )

    # polars backend: exact-parity delegation to the pandas_numpy reference.
    for _canon in _CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
