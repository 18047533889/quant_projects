# -*- coding: utf-8 -*-
"""R67 relation-cross primitives: ``cross_above`` / ``cross_under``.

The relational family already shipped the elementwise comparisons
(``gt``/``ge``/``lt``/``le``/``eq``/``ne``) and the stateful directional
crossing event ``cross_event``, but it had **no basic boolean crossing
primitive** — the everyday "A 上穿 B" event.  Quant factor expressions use it
constantly (``cross_above(close, ts_mean(close, 20))``), so it is added here.

Semantics are copied verbatim from ``stateful.events.CrossEvent`` — the review
authority for this repository's first-bar / missing-value convention:

* ``cross_above(a, b)`` = ``(a_t > b_t) & (a_{t-1} <= b_{t-1})``
* ``cross_under(a, b)`` = ``(a_t < b_t) & (a_{t-1} >= b_{t-1})``
* row 0 has no predecessor, so it emits NaN;
* a non-finite value (NaN **or** ±Inf) at the current *or* previous row of either
  input emits NaN — a missing observation is never read as "no cross";
* every other row emits the float event indicator ``1.0`` / ``0.0`` — the same
  float dtype as the elementwise comparison family (``gt`` → ``0.0`` / ``1.0`` /
  NaN).  It is never a Python ``bool``, so the output composes arithmetically
  with the rest of the DSL.

Three backends are registered per canonical:

* ``pandas_numpy`` — the certified reference kernel in this module;
* ``polars`` — a genuine Polars **native expression** kernel
  (``with_columns`` + ``shift``), no ``to_pandas`` / NumPy / ``rolling_map``;
* ``sql`` — DuckDB/ClickHouse emission, wired in
  ``backend/sql_pushdown/emitter.py`` + ``backend/sql_tiers.py``.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.rolling_pack import frame_like

_SOURCE = "relation_cross_v1"

# Axis / identity columns that are never operator inputs.
_SKIP_COLS = frozenset({"date", "stock_code", "ts", "inst"})


def _metadata(name: str, description: str) -> OperatorMetadata:
    """Metadata mirroring ``stateful.events.cross_event`` (minus ``direction``)."""
    md = OperatorMetadata(
        name=name,
        category="time_series_event",
        description=description,
        param_names=["a", "b"],
        return_type="series",
        tags=[
            "time_series_event", "relation", "crossing", "daily",
            "pit_safe", "causal", "typed_v2", "deterministic",
            "signature:a,b->series", "domain:price_volume", "unit:state",
            "cost:1",
        ],
    )
    md.panel_params = ("a", "b")
    md.panel_arity = 2
    md.scalar_params = ()
    md.output_unit = "state"
    return md


def _cross_panel(a: pd.DataFrame, b: pd.DataFrame, *, above: bool) -> pd.DataFrame:
    """Exact ``CrossEvent`` kernel: row 0 and any non-finite adjacent value → NaN."""
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    if av.shape != bv.shape:
        raise ValueError(
            "cross_above/cross_under inputs must share a shape, "
            f"got {av.shape} and {bv.shape}"
        )
    out = np.full(av.shape, np.nan, dtype=float)
    # ``np.isfinite`` excludes NaN *and* ±Inf — the repo convention (a non-finite
    # observation is invalid, never a mathematical comparison operand).
    finite = np.isfinite(av) & np.isfinite(bv)
    valid = finite[1:] & finite[:-1]  # current AND previous row both valid
    if above:
        crossed = (av[1:] > bv[1:]) & (av[:-1] <= bv[:-1])
    else:
        crossed = (av[1:] < bv[1:]) & (av[:-1] >= bv[:-1])
    out[1:] = np.where(valid, crossed.astype(float), np.nan)
    return frame_like(a, out)


@register_operator(
    name="cross_above",
    category="time_series_event",
    business_category="time_series_event",
    canonical="cross_above",
    source=_SOURCE,
)
class CrossAbove(SeriesOperator):
    """``(a_t > b_t) & (a_{t-1} <= b_{t-1})`` → 1.0/0.0; first bar & NaN → NaN."""

    metadata = _metadata("cross_above", "A 上穿 B: a 由 <=b 变为 >b。")

    def _calculate_series(self, a: pd.DataFrame, b: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return _cross_panel(a, b, above=True)


@register_operator(
    name="cross_under",
    category="time_series_event",
    business_category="time_series_event",
    canonical="cross_under",
    source=_SOURCE,
)
class CrossUnder(SeriesOperator):
    """``(a_t < b_t) & (a_{t-1} >= b_{t-1})`` → 1.0/0.0; first bar & NaN → NaN."""

    metadata = _metadata("cross_under", "A 下穿 B: a 由 >=b 变为 <b。")

    def _calculate_series(self, a: pd.DataFrame, b: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return _cross_panel(a, b, above=False)


# ---------------------------------------------------------------------------
# Polars native expression backend (no pandas / no NumPy inside the kernel).
# ---------------------------------------------------------------------------


def _register_polars_backends() -> tuple[str, ...]:
    try:  # pragma: no cover - exercised only when polars is installed
        import polars as pl
    except Exception:  # pragma: no cover
        return ()

    from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
    from factor_engine.cleaned_operators.base_polars import (
        SeriesOperator as PolarsSeriesOperator,
    )
    from factor_engine.cleaned_operators.base_polars import (
        register_operator as register_polars_operator,
    )
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    def _cross_expr(av: "pl.Expr", bv: "pl.Expr", *, above: bool) -> "pl.Expr":
        # ``is_finite`` false for NULL/NaN/±Inf → masked to NULL, matching the
        # pandas reference's ``np.isfinite`` gate.
        af = pl.when(av.is_finite()).then(av).otherwise(None)
        bf = pl.when(bv.is_finite()).then(bv).otherwise(None)
        ap = af.shift(1)
        bp = bf.shift(1)
        valid = (
            af.is_not_null() & bf.is_not_null() & ap.is_not_null() & bp.is_not_null()
        )
        if above:
            cond = (af > bf) & (ap <= bp)
        else:
            cond = (af < bf) & (ap >= bp)
        return pl.when(~valid).then(None).when(cond).then(1.0).otherwise(0.0)

    def _make(canonical: str, *, above: bool) -> type:
        reference = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
        if reference is None:  # pragma: no cover - registration order guard
            raise RuntimeError(f"missing pandas_numpy reference for {canonical}")
        spec = PhysicalImplementationSpec(
            canonical=canonical,
            backend="polars",
            execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
            supports_lazy=False,
            supports_streaming=False,
            materializes_full_panel=True,
            requires_sorted=True,
            supports_nulls=True,
            supports_nan=True,
            supports_inf=True,
            implementation_source_hash=f"relation_cross_v1:{canonical}:v1",
            kernel_identity=f"relation_cross_v1:{canonical}",
            parameter_domain_hash=f"{canonical}:declared:v1",
            semantic_contract_hash=f"{canonical}:polars_native_expr:v1",
            notes="Pure Polars expressions (with_columns + shift); no pandas/NumPy/rolling_map.",
        )

        class _CrossPolars(PolarsSeriesOperator):
            metadata = copy.deepcopy(reference.metadata)
            _physical_spec = spec

            def _calculate_series(self, a: "pl.DataFrame", b: "pl.DataFrame", **_):
                cols = [
                    c for c in a.columns if c not in _SKIP_COLS and c in b.columns
                ]
                return a.with_columns(
                    [
                        _cross_expr(
                            pl.col(c).cast(pl.Float64), b[c].cast(pl.Float64), above=above
                        ).alias(c)
                        for c in cols
                    ]
                )

        _CrossPolars.__name__ = f"{canonical.title().replace('_', '')}Polars"
        _CrossPolars.__qualname__ = _CrossPolars.__name__
        return _CrossPolars

    registered: list[str] = []
    for canonical, above in (("cross_above", True), ("cross_under", False)):
        register_polars_operator(
            name=canonical,
            canonical=canonical,
            category="time_series_event",
            business_category="time_series_event",
            source=_SOURCE,
            backend="polars",
        )(_make(canonical, above=above))
        registered.append(canonical)
    return tuple(registered)


_POLARS_REGISTERED = _register_polars_backends()

__all__ = ["CrossAbove", "CrossUnder"]
