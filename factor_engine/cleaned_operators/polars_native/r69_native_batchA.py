# -*- coding: utf-8 -*-
"""R69 batch A (strict pure-Polars): two state machines become expr-native.

Group-A audit (R.catalog() x execution_kind, 2026-10-01) left exactly twelve
in-use operators on fake-Polars slots: two ``polars_udf_pandas_delegate``
bridges and ten ``polars_numpy_kernel`` state machines.  This module converts
the canonicals whose float64 semantics are exactly expressible with pure
Polars expression chains (no pandas, no NumPy, no ``map_elements``, no
``to_pandas``):

* ``state_quantile_hysteresis``     NumPy row loop -> last-event-wins fill
* ``state_l2_partial_adjustment``   NumPy row loop -> closed-form segment
                                    restarted EWM (adjust=False + decayed
                                    history correction, exact to float64)

NOTE (same-day): the ``group_ex_self_mean`` / ``group_ex_self_weighted_mean``
polars SLOTS are owned by the concurrent R69 agent's ``group_expr_extensions``
registration (which imports the kernels below and registers after this
module); this module therefore does not re-register them.

The remaining eight group-A state machines are *strictly sequential* (greedy
event chains and bounded-slew recursions).  Their exact float64 evaluation has
neither a closed form nor an associative scan, so a pure expression kernel
cannot meet the R69 parity gate; they intentionally keep the certified bounded
kernel path (verdicts in ``/tmp/r69/PROGRESS_A.md``).

Wide panels enter, wide panels leave (``__fe_time__`` and metadata columns are
preserved).  Long intermediates are row-batched so no full-panel copy is made
(same 250k-cell cap as ``group_ops_native_expr_20261001``).
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import Operator

_SOURCE = "factor_engine.cleaned_operators.polars_native.r69_native_batchA"
_EPS = 1e-12
_META = {
    "__fe_time__", "date", "timestamp", "trade_date", "datetime",
    "stock_code", "instrument", "symbol", "session", "inst", "ts",
}
_MAX_CELLS = 250_000


# ---------------------------------------------------------------------------
# shared helpers (pure polars; no numpy / pandas anywhere)
# ---------------------------------------------------------------------------
def _as_frame(value: Any) -> pl.DataFrame:
    if isinstance(value, pl.Series):
        return value.to_frame()
    if not isinstance(value, pl.DataFrame):
        raise TypeError(f"expected a polars panel, got {type(value)!r}")
    return value


def _cols(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in _META]


def _assert_same_axes(panels: list[pl.DataFrame], cols: list[str], names: str) -> None:
    for panel, name in zip(panels[1:], names.split(",")):
        if _cols(panel) != cols:
            raise ValueError(f"{name} panel instrument columns must exactly match x")
        if panel.height != panels[0].height:
            raise ValueError(f"{name} panel rows must exactly match x")


def _label_expr(column: str, dtype: pl.DataType) -> tuple[pl.Expr, pl.Expr]:
    """(group_key, valid) expressions mirroring ``_valid_membership_label``."""
    if dtype == pl.String or dtype == pl.Utf8:
        raw = pl.col(column)
        key = pl.lit("text:", dtype=pl.String) + raw.cast(pl.String, strict=False)
        valid = raw.is_not_null() & (raw != "")
        return key, valid.fill_null(False)
    if dtype.is_numeric() or dtype == pl.Boolean:
        f = pl.col(column).cast(pl.Float64, strict=False)
        key = pl.lit("number:", dtype=pl.String) + f.cast(pl.String)
        valid = f.is_finite().fill_null(False)
        return key, valid
    raw = pl.col(column)
    key = pl.lit(str(dtype) + ":", dtype=pl.String) + raw.cast(pl.String, strict=False)
    return key, raw.is_not_null().fill_null(False)


def _label_frame(group: pl.DataFrame, cols: list[str], start: int, stop: int) -> pl.DataFrame:
    gb = group.slice(start, stop - start)
    pieces = []
    for c in cols:
        key, valid = _label_expr(c, gb.schema[c])
        pieces.append(gb.select(
            pl.int_range(start, stop, dtype=pl.Int64).alias("_row"),
            pl.lit(c, dtype=pl.String).alias("_col"),
            key.alias("_g"),
            valid.alias("_gv"),
        ))
    schema = {"_row": pl.Int64, "_col": pl.String, "_g": pl.String, "_gv": pl.Boolean}
    if not pieces:
        return pl.DataFrame(schema=schema)
    return pl.concat(pieces, how="vertical")


def _pivot_wide(long: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    wide = long.select("_row", "_col", "_out").pivot(
        on="_col", index="_row", values="_out", aggregate_function="first"
    ).sort("_row")
    return wide.select(
        [pl.col(c).cast(pl.Float64, strict=False).alias(c) if c in wide.columns
         else pl.lit(None, dtype=pl.Float64).alias(c) for c in cols]
    )


def _wide_result(base: pl.DataFrame, chunks: dict[str, list[pl.Series]]) -> pl.DataFrame:
    return base.with_columns(
        [pl.concat(chunks[c], how="vertical").alias(c) for c in _cols(base)]
    )


# ---------------------------------------------------------------------------
# group_ex_self_mean / group_ex_self_weighted_mean (leave-one-out peer means)
#
# NOTE: these two kernels are the shared implementation behind the concurrent
# agent's ``group_expr_extensions`` registration (which imports them from this
# module).  They are deliberately NOT registered again here, so exactly one
# polars slot exists per canonical.
# ---------------------------------------------------------------------------
def group_ex_self_mean_kernel(bound: dict) -> pl.DataFrame:
    x = _as_frame(bound["x"])
    group = _as_frame(bound["group"])
    cols = _cols(x)
    if not cols:
        raise ValueError("panel has no instrument columns")
    _assert_same_axes([x, group], cols, "group")
    n = x.height
    step = max(1, _MAX_CELLS // len(cols))
    chunks: dict[str, list[pl.Series]] = {c: [] for c in cols}
    for start in range(0, n, step):
        stop = min(n, start + step)
        xb = x.slice(start, stop - start).select(cols).with_columns(
            pl.int_range(start, stop, dtype=pl.Int64).alias("_row")
        )
        vals = xb.unpivot(index=["_row"], on=cols, variable_name="_col", value_name="_x")
        vals = vals.join(_label_frame(group, cols, start, stop), on=["_row", "_col"], how="left")
        xf = pl.col("_x").cast(pl.Float64, strict=False)
        valid = xf.is_finite().fill_null(False) & pl.col("_gv").fill_null(False)
        stats = (
            vals.with_columns(valid.alias("_valid"))
            .group_by("_row", "_g")
            .agg(
                pl.col("_x").filter(pl.col("_valid")).cast(pl.Float64).sum().alias("_gsum"),
                pl.col("_valid").sum().alias("_gcnt"),
            )
        )
        joined = vals.join(stats, on=["_row", "_g"], how="left")
        peers = pl.col("_gcnt") - 1
        joined = joined.with_columns(
            pl.when(valid & (peers >= 1))
            .then((pl.col("_gsum") - xf) / peers)
            .otherwise(None)
            .alias("_out")
        )
        wide = _pivot_wide(joined, cols)
        for c in cols:
            chunks[c].append(wide[c])
    return _wide_result(x, chunks)


def group_ex_self_weighted_mean_kernel(bound: dict) -> pl.DataFrame:
    x = _as_frame(bound["x"])
    weight = _as_frame(bound["weight"])
    group = _as_frame(bound["group"])
    cols = _cols(x)
    if not cols:
        raise ValueError("panel has no instrument columns")
    _assert_same_axes([x, weight, group], cols, "weight,group")
    n = x.height
    step = max(1, _MAX_CELLS // len(cols))
    chunks: dict[str, list[pl.Series]] = {c: [] for c in cols}
    for start in range(0, n, step):
        stop = min(n, start + step)
        xb = x.slice(start, stop - start).select(cols).with_columns(
            pl.int_range(start, stop, dtype=pl.Int64).alias("_row")
        )
        vals = xb.unpivot(index=["_row"], on=cols, variable_name="_col", value_name="_x")
        wb = weight.slice(start, stop - start).select(cols).with_columns(
            pl.int_range(start, stop, dtype=pl.Int64).alias("_row")
        )
        wvals = wb.unpivot(index=["_row"], on=cols, variable_name="_col", value_name="_w")
        vals = vals.join(wvals, on=["_row", "_col"], how="left")
        vals = vals.join(_label_frame(group, cols, start, stop), on=["_row", "_col"], how="left")
        xf = pl.col("_x").cast(pl.Float64, strict=False)
        wf = pl.col("_w").cast(pl.Float64, strict=False)
        # R5 P1-37(a): numerator and denominator share one mask:
        # member(x) & finite(w) & finite(x) & (w >= 0).
        mask = (
            xf.is_finite().fill_null(False)
            & wf.is_finite().fill_null(False)
            & (wf >= 0.0).fill_null(False)
            & pl.col("_gv").fill_null(False)
        )
        stats = (
            vals.with_columns(mask.alias("_mask"))
            .group_by("_row", "_g")
            .agg(
                (pl.col("_w") * pl.col("_x")).filter(pl.col("_mask")).cast(pl.Float64).sum().alias("_swx"),
                pl.col("_w").filter(pl.col("_mask")).cast(pl.Float64).sum().alias("_wsum"),
                pl.col("_mask").sum().alias("_gcnt"),
            )
        )
        joined = vals.join(stats, on=["_row", "_g"], how="left")
        peers = pl.col("_gcnt") - 1
        num = pl.col("_swx") - wf * xf
        den = pl.col("_wsum") - wf
        joined = joined.with_columns(
            pl.when(mask & (peers >= 1) & den.is_finite() & (den > 0.0))
            .then(num / den)
            .otherwise(None)
            .alias("_out")
        ).with_columns(
            pl.when(pl.col("_out").is_finite()).then(pl.col("_out")).otherwise(None).alias("_out")
        )
        wide = _pivot_wide(joined, cols)
        for c in cols:
            chunks[c].append(wide[c])
    return _wide_result(x, chunks)


# ---------------------------------------------------------------------------
# state_quantile_hysteresis (last-event-wins fill; no sequential loop)
# ---------------------------------------------------------------------------
def state_quantile_hysteresis_kernel(bound: dict) -> pl.DataFrame:
    x = _as_frame(bound["x"])
    group = bound.get("group")
    enter_q = float(bound.get("enter_quantile", 0.9))
    exit_q = float(bound.get("exit_quantile", 0.8))
    if not (0.0 <= enter_q <= 1.0) or not (0.0 <= exit_q <= 1.0):
        raise ValueError("state_quantile_hysteresis requires quantiles in [0, 1]")
    if exit_q > enter_q:
        raise ValueError("state_quantile_hysteresis requires exit_quantile <= enter_quantile")
    cols = _cols(x)
    if not cols:
        raise ValueError("panel has no instrument columns")
    grouped = group is not None
    if grouped:
        group = _as_frame(group)
        _assert_same_axes([x, group], cols, "group")
    n = x.height
    step = max(1, _MAX_CELLS // len(cols))
    chunks: dict[str, list[pl.Series]] = {c: [] for c in cols}
    # The hysteresis state is an unbounded-memory scan, so row batches must
    # hand the per-column internal state (0.0/1.0 at the last row) to the next
    # batch instead of restarting at 0.  A NaN-rank last row maps to state 0
    # in both the authority and the carry below.
    carry: dict[str, float] = {}
    for start in range(0, n, step):
        stop = min(n, start + step)
        xb = x.slice(start, stop - start).select(cols).with_columns(
            pl.int_range(start, stop, dtype=pl.Int64).alias("_row")
        )
        vals = xb.unpivot(index=["_row"], on=cols, variable_name="_col", value_name="_x")
        if grouped:
            vals = vals.join(_label_frame(group, cols, start, stop), on=["_row", "_col"], how="left")
            valid_label = pl.col("_gv").fill_null(False)
        else:
            valid_label = pl.lit(True)
        xf = pl.col("_x").cast(pl.Float64, strict=False)
        finite = xf.is_finite().fill_null(False) & valid_label
        part = ["_row", "_g"] if grouped else ["_row"]
        work = vals.with_columns(
            pl.when(finite).then(xf).otherwise(None).alias("_xv")
        ).with_columns(
            pl.col("_xv").rank("average").over(part).alias("_rk"),
            pl.col("_xv").is_not_null().sum().over(part).alias("_cnt"),
        )
        q = pl.when(finite).then(
            pl.col("_rk").cast(pl.Float64) / pl.col("_cnt").cast(pl.Float64)
        ).otherwise(None)
        # Two-threshold hysteresis == "last decision event wins":
        #   enter: q >= enter_q - eps ; exit: q < exit_q - eps ; NaN q resets to 0.
        # exit_quantile <= enter_quantile keeps the two events disjoint.
        ev = (
            pl.when(q.is_null()).then(pl.lit(0.0))
            .when(q >= enter_q - _EPS).then(pl.lit(1.0))
            .when(q < exit_q - _EPS).then(pl.lit(0.0))
            .otherwise(None)
        )
        work = work.sort("_col", "_row")
        if carry:
            carry_df = pl.DataFrame(
                {"_col": list(carry.keys()), "_carry": list(carry.values())}
            )
            work = work.join(carry_df, on="_col", how="left")
            base = pl.when(pl.col("_row") == start).then(
                pl.coalesce(ev, pl.col("_carry"))
            ).otherwise(ev)
        else:
            base = ev
        work = work.with_columns(
            base.alias("_base")
        ).with_columns(
            pl.col("_base").forward_fill().over("_col").fill_null(0.0).alias("_state")
        ).with_columns(
            pl.when(finite).then(pl.col("_state")).otherwise(None).alias("_out")
        )
        wide = _pivot_wide(work, cols)
        for c in cols:
            chunks[c].append(wide[c])
            last = wide[c][-1]
            carry[c] = float(last) if last is not None else 0.0
    return _wide_result(x, chunks)


# ---------------------------------------------------------------------------
# state_l2_partial_adjustment (closed-form segment-restarted EWM)
# ---------------------------------------------------------------------------
def state_l2_partial_adjustment_kernel(bound: dict) -> pl.DataFrame:
    x = _as_frame(bound["x"])
    lam = float(bound.get("lambda_smooth", 1.0))
    if lam < 0.0:
        raise ValueError("state_l2_partial_adjustment requires lambda_smooth >= 0")
    cols = _cols(x)
    if not cols:
        raise ValueError("panel has no instrument columns")
    alpha = 1.0 / (1.0 + lam)      # update weight
    decay = lam / (1.0 + lam)      # carry weight (1 - alpha), strictly < 1
    rr = pl.int_range(0, pl.len(), dtype=pl.Int64)
    exprs: list[pl.Expr] = []
    for c in cols:
        v = pl.col(c).cast(pl.Float64, strict=False)
        finite = v.is_finite().fill_null(False)
        prev_finite = finite.shift(1).fill_null(False)
        boundary = finite & ~prev_finite
        csf = finite.cast(pl.Int64).cum_sum()
        seed_ord = pl.when(boundary).then(csf).otherwise(None).forward_fill()
        seg0 = (
            pl.when(boundary).then(finite & (rr == 0)).otherwise(None)
            .forward_fill().fill_null(False)
        )
        k = (csf - seed_ord).cast(pl.Float64)
        seed_x = pl.when(boundary).then(v).otherwise(None).forward_fill()
        # EWM over a zero-filled input decays the pre-segment history through
        # NaN gaps; the authority resets instead, so the carried history is
        # removed again by the exact decayed correction term below:
        #   segment from row 0:  y_t = E_t + decay^k * x_seed
        #   segment from row s>0: y_t = E_t - decay^(k+1) * (E_{s-1} - x_seed)
        # (k = rows since the segment seed; E = adjust=False EWM over the
        # zero-filled input whose row-0 term is forced to zero).
        z = pl.when(finite).then(v).otherwise(0.0)
        z = pl.when(rr == 0).then(pl.lit(0.0)).otherwise(z)
        ewm = z.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True, min_samples=1)
        # E_{s-1}: the EWM state on the row BEFORE the segment seed (sampled at
        # the boundary row and carried through the segment), not E_{t-1}.
        ewm_before_seed = (
            pl.when(boundary).then(ewm.shift(1)).otherwise(None).forward_fill()
        )
        y = (
            pl.when(seg0)
            .then(ewm + pl.lit(decay).pow(k) * seed_x)
            .otherwise(ewm - pl.lit(decay).pow(k + 1.0) * (ewm_before_seed - seed_x))
        )
        exprs.append(pl.when(finite).then(y).otherwise(None).alias(c))
    return x.with_columns(exprs)


_KERNELS = {
    "state_quantile_hysteresis": state_quantile_hysteresis_kernel,
    "state_l2_partial_adjustment": state_l2_partial_adjustment_kernel,
}


class _R69BatchANative(Operator):
    _HANDLES_CALL_CONTRACT = True

    def __init__(self, canonical: str, metadata: Any, kernel):
        self._canonical, self.metadata, self._kernel = canonical, metadata, kernel

    def calculate(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
        args, kwargs = self._prepare_call(args, kwargs)
        bound = dict(zip(self.metadata.param_names, args))
        bound.update(kwargs)
        return self._kernel(bound)

    _calculate_series = calculate


def register_r69_native_batch_a() -> tuple[str, ...]:
    from types import MappingProxyType

    from factor_engine.cleaned_operators import (
        record_backend_replacement_after,
        replace_backend,
    )
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if isinstance(OperatorRegistry._operators, MappingProxyType):
        return ()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    registered: list[str] = []
    for canonical, kernel in _KERNELS.items():
        reference = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
        if reference is None:
            continue
        current_meta = (
            ((OperatorRegistry._catalog.get(canonical, {}) or {}).get("backend_meta") or {})
            .get("polars") or {}
        )
        if current_meta.get("source") == _SOURCE:
            continue
        metadata = copy.deepcopy(reference.metadata)
        metadata.tags = list(metadata.tags or []) + ["preserve_panel_time_coordinate"]
        operator = _R69BatchANative(canonical, metadata, kernel)
        operator._physical_spec = PhysicalImplementationSpec(
            canonical=canonical,
            backend="polars",
            execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
            supports_lazy=False,
            supports_streaming=False,
            materializes_full_panel=True,
            supports_nulls=True,
            supports_nan=True,
            supports_inf=False,
            implementation_source_hash=source_hash,
            emitter_identity=f"{_SOURCE}:expr-native:v1",
            kernel_identity=f"{_SOURCE}._KERNELS:{canonical}",
            parameter_domain_hash=hashlib.sha256(
                repr((metadata.param_names, getattr(metadata, "param_specs", None))).encode()
            ).hexdigest(),
            semantic_contract_hash=hashlib.sha256(
                f"{canonical}:pandas-authority-parity:r69-batch-a:v1".encode()
            ).hexdigest(),
            notes=(
                "Pure Polars expression kernel (long-batch / ewm closed form); "
                "no pandas conversion, no NumPy, no UDF."
            ),
        )
        migration = replace_backend(
            canonical, "polars",
            reason="R69 batch A strict pure-polars conversion replaces numpy-kernel slot",
            source=_SOURCE,
        )
        OperatorRegistry.register(operator, canonical=canonical, backend="polars", source=_SOURCE)
        spec = operator._physical_spec
        OperatorRegistry._catalog[canonical]["backend_meta"]["polars"].update({
            "execution_kind": spec.execution_kind.value,
            "supports_lazy": spec.supports_lazy,
            "supports_streaming": spec.supports_streaming,
            "materializes_full_panel": spec.materializes_full_panel,
            "supports_nulls": spec.supports_nulls,
            "supports_nan": spec.supports_nan,
            "supports_inf": spec.supports_inf,
            "supports_scalar_broadcast": False,
            "supports_group": canonical == "state_quantile_hysteresis",
            "supports_window": False,
            "supports_min_periods": False,
        })
        record_backend_replacement_after(migration, canonical, "polars", source=_SOURCE)
        registered.append(canonical)
    return tuple(registered)


__all__ = [
    "register_r69_native_batch_a",
    "group_ex_self_mean_kernel",
    "group_ex_self_weighted_mean_kernel",
    "state_quantile_hysteresis_kernel",
    "state_l2_partial_adjustment_kernel",
]
