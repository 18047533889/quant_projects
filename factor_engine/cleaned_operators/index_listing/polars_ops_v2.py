# -*- coding: utf-8 -*-
"""Polars backends for next-stage index / listing operators (genuine expressions)."""
from __future__ import annotations

from typing import Any

import polars as pl

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator


def _cols(df, *others):
    cols = [c for c in df.columns if c != "date"]
    for o in others:
        cols = [c for c in cols if c in o.columns]
    return cols


def _meta(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name, category="index", description=description, param_names=params,
        return_type="series", tags=["index", "polars", "native", "typed_v2"],
    )


# ---------------------------------------------------------------------------
# R57 backend-coverage batch 4 — explicit execution-kind declarations.
# These kernels are genuine polars expressions (pl.Expr / with_columns over
# columns).  Previously they had no _physical_spec, so
# canonical_polars_kind(production_mode=True) failed closed to UNSUPPORTED.
# ---------------------------------------------------------------------------
from factor_engine.backend.contracts import (
    ExecutionKind,
    PhysicalImplementationSpec,
)

_BATCH4_NOTE = (
    "Genuine polars expression kernel (pl.Expr over columns, no pandas round-trip); "
    "runtime marshal probe on real data records 0 pl.DataFrame.to_pandas "
    "calls. Eager panel API only: no lazy/streaming or production-parity claim."
)


def _batch4_native_spec(canonical: str, kernel: str) -> PhysicalImplementationSpec:
    """Explicit execution-kind contract for a genuine polars expression kernel.

    Batch-4 evidence contract (same rules as batches 1-3): the kernel body
    builds pl.Expr / pl.DataFrame columns with no pandas round-trip, and the
    runtime marshal probe on real data records zero pl.DataFrame.to_pandas
    calls.  Eager panel API, so supports_lazy / supports_streaming stay False.
    """
    return PhysicalImplementationSpec(
        canonical=canonical,
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        materializes_full_panel=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash=f"index_listing.polars_ops_v2:{kernel}:v1",
        emitter_identity=f"polars_expr:{canonical}",
        kernel_identity=f"index_listing.polars_ops_v2:{kernel}",
        parameter_domain_hash=f"{canonical}:declared:v1",
        semantic_contract_hash=f"{canonical}:polars_native_expr:v1",
        notes=_BATCH4_NOTE,
    )



def _register(name: str, description: str, params: list[str], fn):
    @register_operator(
        name=name, category="index", business_category="index",
        canonical=name, source="index_listing.polars_ops_v2",
    )
    class _IndexListingPolars(SeriesOperator):
        metadata = _meta(name, description, params)
        if name == "suspension_frequency":
            _physical_spec = _batch4_native_spec(name, "_IndexListingPolars")


        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    return _IndexListingPolars


def _binary(a, b, expr_fn):
    cols = _cols(a, b)
    return a.with_columns([expr_fn(a[c], b[c]).alias(c) for c in cols])


_register(
    "index_weight_gap_to_free_float", "指数权重 vs 自由流通权重偏离（Polars）。", ["index_weight", "free_float_weight"],
    lambda iw, fw: _binary(iw, fw, lambda a, b: (a - b) / b.fill_null(0.0).replace(0, None).abs()),
)
_register(
    "suspension_frequency", "停牌频率（Polars）。", ["is_suspend", "window"],
    lambda is_s, window=60: is_s.with_columns(
        [
            pl.when(
                is_s[c].is_not_null().cast(pl.Float64).rolling_sum(int(window)) == 0
            )
            .then(None)
            .otherwise(
                (is_s[c].is_not_null() & (is_s[c] != 0))
                .cast(pl.Float64)
                .rolling_sum(int(window))
                / is_s[c].is_not_null().cast(pl.Float64).rolling_sum(int(window))
            )
            .alias(c)
            for c in _cols(is_s)
        ]
    ),
)


def _reconstitution_churn(member, window):
    from factor_engine.cleaned_operators.common.polars_membership import reconstitution_churn
    return reconstitution_churn(member, window)


_register("index_reconstitution_churn", "窗口内纳入+剔除次数（Polars）。", ["member", "window"],
          lambda m, window=60: _reconstitution_churn(m, int(window)))


def _multi_index_entry(entry_a, entry_b, entry_c, window):
    # P1-141: partial-unknown flags (A=1,B=NaN,C=NaN) must NOT collapse to
    # 1+0+0 = "1 entered".  Require ALL three flags known per day; unknown is
    # never "not entered".  A window containing any unknown day yields null
    # (fail closed) via min_samples=window on the null-propagated value.
    w = int(window)
    cols = _cols(entry_a, entry_b, entry_c)
    out = []
    for c in cols:
        all_known = (
            entry_a[c].is_not_null()
            & entry_b[c].is_not_null()
            & entry_c[c].is_not_null()
        )
        # entry_a+entry_b+entry_c propagates null -> value null on unknown days.
        val = pl.when(all_known).then(
            entry_a[c] + entry_b[c] + entry_c[c]
        ).otherwise(None)
        out.append(val.rolling_sum(w).alias(c))
    return entry_a.with_columns(out)


_register("multi_index_entry_intensity", "多指数同时纳入强度（Polars）。", ["entry_index_a", "entry_index_b", "entry_index_c", "window"],
          lambda a, b, c, window=20: _multi_index_entry(a, b, c, int(window)))


def _event_decay(entry_event, window, decay, missing_policy="break"):
    import numpy as np

    w = int(window)
    d = float(decay)
    if not (np.isfinite(d) and 0.0 <= d <= 1.0):
        raise ValueError("index_event_decay decay must satisfy 0 <= decay <= 1")
    policy = str(missing_policy or "break").lower()
    if policy not in ("break", "carry"):
        raise ValueError("index_event_decay missing_policy must be 'break' or 'carry'")

    weights = np.array([d ** k for k in range(w)], dtype=float)

    def _apply(vals):
        arr = np.asarray(vals, dtype=float)
        n = len(arr)
        out = np.full(n, np.nan, dtype=float)
        seen = False
        last_valid = np.nan
        for i in range(n):
            if np.isfinite(arr[i]):
                seen = True
            if not seen:
                continue  # before the first known event -> NaN
            start = max(0, i - w + 1)
            seg = arr[start:i + 1]
            if len(seg) < w:
                continue  # full-window warmup contract
            if not np.all(np.isfinite(seg)):
                # Data gap inside the window: never treat as zero events.
                # P1-143: "carry" carries the last valid decay value forward;
                # "break" leaves NaN until a full clean window re-accumulates.
                if policy == "carry" and np.isfinite(last_valid):
                    out[i] = last_valid
                continue
            out[i] = float(np.dot(seg[::-1], weights))
            last_valid = out[i]
        return out

    cols = _cols(entry_event)
    result = entry_event.clone()
    for c in cols:
        result = result.with_columns(
            pl.Series(c, _apply(entry_event[c].cast(pl.Float64).to_numpy()))
        )
    return result


_register("index_event_decay", "指数事件衰减近因信号（Polars）。",
          ["entry_event", "window", "decay", "missing_policy"],
          lambda e, window=20, decay=0.9, missing_policy="break": _event_decay(
              e, int(window), float(decay), missing_policy,
          ))
