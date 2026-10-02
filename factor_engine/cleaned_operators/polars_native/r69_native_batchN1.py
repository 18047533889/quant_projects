# -*- coding: utf-8 -*-
"""R69 batch N1 (strict pure-Polars): wave 1 — 18 in-use numpy kernels.

Canonicals (11.4 万因子目录 EXECUTED 使用频次降序, 全部原 ``polars_numpy_kernel``
槽位; 目录频次见 /tmp/r69/n1_selection.json):

    open_to_vwap_return                        vwap/open - 1 (PositivePrice gate)
    ts_path_efficiency                         net/path over trailing finite run
    ts_vol_of_vol / ts_vol_term_structure /
    ts_vol_acceleration / ts_vol_clustering    rolling population-std compositions
    ts_jump_bipower_proxy                      RV/BV over trailing finite run
    ts_days_since                              NaN-censored last-event distance
    ts_corr_if / ts_min_if / ts_max_if /
    ts_quantile_if                             condition-masked rolling stats
    ts_beta                                    paired rolling cov/var
    event_frequency                            rolling hit ratio

Every kernel is a pure Polars expression chain (``pl.col`` / ``when`` /
``rolling_*`` / ``shift`` / ``cum_*`` / ``int_range``): no pandas, no NumPy,
no ``map_elements`` / ``rolling_map`` / ``to_pandas``.  NaN/±Inf inputs are
masked to null so the Polars rolling aggregates reproduce the numpy
``np.isfinite`` semantics of the pandas authorities.  ``min_samples=1``
rolling windows reproduce the authorities' shrinking-window
``lo = max(0, r - w + 1)`` behaviour; the explicit ``n >= min_periods`` gates
reproduce their finite-count support policy.

Registration replaces the ``polars`` slot (``replace_backend``) and stamps
``ExecutionKind.POLARS_NATIVE_EXPR``; the pandas authority slots stay (they
are the parity oracle).
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import Operator

_SOURCE = "factor_engine.cleaned_operators.polars_native.r69_native_batchN1"
_EPS = 1e-12
_META = {
    "__fe_time__", "date", "timestamp", "trade_date", "datetime",
    "stock_code", "instrument", "symbol", "session", "inst", "ts",
}


# ---------------------------------------------------------------------------
# shared helpers (pure polars)
# ---------------------------------------------------------------------------
def _as_frame(value: Any) -> pl.DataFrame:
    if isinstance(value, pl.DataFrame):
        return value
    if isinstance(value, pl.Series):
        return value.to_frame()
    raise TypeError(f"expected a polars panel, got {type(value)!r}")


def _cols(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in _META]


def _int_rows() -> pl.Expr:
    return pl.int_range(0, pl.len(), dtype=pl.Int64)


def _fx(col: str) -> pl.Expr:
    """Finite-only float view of a panel column (NaN / ±Inf / null -> null)."""
    v = pl.col(col).cast(pl.Float64, strict=False)
    return pl.when(v.is_finite()).then(v).otherwise(None)




def _ms(mp: int, w: int) -> int:
    return max(1, min(int(mp), int(w)))




def _attach(base: pl.DataFrame, cols: list[str], out: pl.DataFrame) -> pl.DataFrame:
    return base.with_columns([pl.Series(c, out[c]) for c in cols])


def _hstack(frames: list[pl.DataFrame], cols: list[str], prefixes: list[str]) -> pl.DataFrame:
    """Horizontally stack same-axis panels with prefixed column names."""
    pieces = []
    for frame, prefix in zip(frames, prefixes):
        pieces.append(frame.select(cols).rename({c: f"{prefix}{c}" for c in cols}))
    return pl.concat(pieces, how="horizontal_extend")




# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def open_to_vwap_return_kernel(bound: dict) -> pl.DataFrame:
    """vwap/open_px - 1 with the P0-61 PositivePrice gate (finite price <= 0 -> NaN)."""
    open_px = _as_frame(bound["open_px"])
    vwap = _as_frame(bound["vwap"])
    cols = _cols(open_px)
    vw_cols = _cols(vwap)
    if vw_cols != cols:
        raise ValueError("open_to_vwap_return requires vwap columns to match open_px")
    work = _hstack([open_px, vwap], cols, ["o_", "v_"])
    exprs = []
    for c in cols:
        o = _fx(f"o_{c}")
        vw = _fx(f"v_{c}")
        exprs.append(
            pl.when((o > 0.0) & (vw > 0.0))
            .then(vw / o - 1.0)
            .otherwise(None)
            .alias(c)
        )
    return _attach(open_px, cols, work.select(exprs))


def _run_anchor(idx: pl.Expr, v: pl.Expr, w: int) -> dict[str, pl.Expr]:
    """Trailing-contiguous finite run plumbing shared by path/BV kernels.

    The authority keeps the window's trailing finite run ``[s, t]`` with
    ``s = max(t - w + 1, last_nonfinite + 1, 0)``.  Prefix sums over the full
    column recover the window sum as ``cs_t - cs_anchor`` with
    ``anchor = s - 1 = max(t - w, last_bad)``; because prefix sums are
    monotone, ``min(xA, xB)`` over the two anchor candidates reproduces the
    max-anchor choice without variable-length indexing.
    """
    bad = v.is_null()
    last_bad = pl.when(bad).then(idx).otherwise(None).forward_fill()
    anchorB = last_bad.fill_null(-1)
    return {
        "last_bad": last_bad,
        "anchorA": idx - w,
        "anchorB": anchorB,
        "n_run": pl.min_horizontal(
            pl.min_horizontal(idx, pl.lit(w - 1, dtype=pl.Int64)) + 1,
            (idx - anchorB).clip(0, w),
        ),
    }


def ts_path_efficiency_kernel(bound: dict) -> pl.DataFrame:
    """net/path over the trailing finite run (constant_path_policy=ZERO)."""
    x = _as_frame(bound["x"])
    w = int(bound.get("window", 20) or 20)
    mp = max(2, int(bound.get("min_periods", 2) or 2))
    if w < 2:
        raise ValueError("window must be >= 2")
    idx = _int_rows()
    exprs = []
    for c in _cols(x):
        v = _fx(c)
        anc = _run_anchor(idx, v, w)
        d = v - v.shift(1)
        cs = pl.when(d.is_not_null()).then(d.abs()).otherwise(0.0).cum_sum()
        # Window-full authority run starts at s = t-w+1; path pairs END at
        # [t-w+2, t] -> subtract cs_{t-w+1} (= cs.shift(w-1)).  The run-truncated
        # branch subtracts cs at last_bad, which equals cs_{last_bad+1} because
        # d at last_bad+1 bridges a non-finite value and contributes zero.
        pathA = cs - cs.shift(w - 1).fill_null(0.0)
        cs_at_b = pl.when(anc["last_bad"] == idx).then(cs).otherwise(None).forward_fill().fill_null(0.0)
        pathB = cs - cs_at_b
        path = pl.min_horizontal(pathA, pathB)
        # net = |x_t - x_s| with s = anchor + 1: value at the winning anchor's
        # successor (window start x_{t-w+1}, or x_{last_bad+1}, or x_0).
        x_sA = v.shift(w - 1)
        x_sB = pl.when(anc["last_bad"] + 1 == idx).then(v).otherwise(None).forward_fill()
        x_s0 = pl.when(idx == 0.0).then(v).otherwise(None).forward_fill()
        use_b = anc["last_bad"].is_not_null() & (anc["anchorB"] >= anc["anchorA"])
        x_s = (
            pl.when(use_b).then(x_sB)
            .when(anc["anchorA"] >= 0.0).then(x_sA)
            .otherwise(x_s0)
        )
        net = (v - x_s).abs()
        out = pl.when(v.is_not_null() & (anc["n_run"] >= mp)).then(
            pl.when(path == 0.0).then(0.0).otherwise(net / path)
        ).otherwise(None)
        exprs.append(out.alias(c))
    return x.with_columns(exprs)








def ts_days_since_kernel(bound: dict) -> pl.DataFrame:
    """Rows since the last confirmed event; NaN censors output AND resets.

    Authority (overhaul.daily.pd_days_since): a NaN condition row resets the
    last-true state to "unknown" until the next explicit true rebuilds it.
    Expression: global forward-fill of the last true index, valid only when it
    is NEWER than the last NaN position (a NaN after the last true censors).
    """
    x = _as_frame(bound["condition"])
    limit_raw = bound.get("max_lookback")
    limit = None if limit_raw is None else int(limit_raw)
    idx = _int_rows()
    exprs = []
    for c in _cols(x):
        raw = pl.col(c).cast(pl.Float64, strict=False)
        known = raw.is_finite().fill_null(False)
        last_true = (
            pl.when(known & (raw == 1.0)).then(idx).otherwise(None).forward_fill()
        )
        last_nan = (
            pl.when(~known & raw.is_not_null()).then(idx).otherwise(None).forward_fill()
        )
        ok = last_true.is_not_null() & (
            last_nan.is_null() | (last_true > last_nan)
        )
        dist = idx - last_true
        if limit is not None:
            ok = ok & (dist < limit)
        exprs.append(pl.when(ok).then(dist.cast(pl.Float64)).otherwise(None).alias(c))
    return x.with_columns(exprs)




def _masked_minmax_kernel(bound: dict, op: str) -> pl.DataFrame:
    x = _as_frame(bound["x"])
    cond = _as_frame(bound["condition"])
    w = int(bound.get("window", 20) or 20)
    mp = max(1, int(bound.get("min_periods", 1) or 1))
    cols = _cols(x)
    work = _hstack([x, cond], cols, ["x_", "c_"])
    exprs = []
    for c in cols:
        xv = _fx(f"x_{c}")
        cf = pl.col(f"c_{c}").cast(pl.Float64, strict=False)
        mask = (cf.is_finite() & (cf == 1.0) & xv.is_not_null()).fill_null(False)
        sel = pl.when(mask).then(xv).otherwise(None)
        agg = (
            sel.rolling_min(w, min_samples=_ms(mp, w))
            if op == "min"
            else sel.rolling_max(w, min_samples=_ms(mp, w))
        )
        exprs.append(agg.alias(c))
    return _attach(x, cols, work.select(exprs))


def ts_min_if_kernel(bound: dict) -> pl.DataFrame:
    return _masked_minmax_kernel(bound, "min")


def ts_max_if_kernel(bound: dict) -> pl.DataFrame:
    return _masked_minmax_kernel(bound, "max")


def ts_quantile_if_kernel(bound: dict) -> pl.DataFrame:
    x = _as_frame(bound["x"])
    cond = _as_frame(bound["condition"])
    w = int(bound.get("window", 20) or 20)
    q = float(bound.get("q", 0.5) or 0.5)
    if not 0.0 <= q <= 1.0:
        raise ValueError("ts_quantile_if requires 0 <= q <= 1")
    mp = max(1, int(bound.get("min_periods", 1) or 1))
    cols = _cols(x)
    work = _hstack([x, cond], cols, ["x_", "c_"])
    exprs = []
    for c in cols:
        xv = _fx(f"x_{c}")
        cf = pl.col(f"c_{c}").cast(pl.Float64, strict=False)
        mask = (cf.is_finite() & (cf == 1.0) & xv.is_not_null()).fill_null(False)
        sel = pl.when(mask).then(xv).otherwise(None)
        exprs.append(
            sel.rolling_quantile(q, interpolation="linear", window_size=w,
                                 min_samples=_ms(mp, w)).alias(c)
        )
    return _attach(x, cols, work.select(exprs))


_KERNELS = {
    "open_to_vwap_return": open_to_vwap_return_kernel,
    "ts_path_efficiency": ts_path_efficiency_kernel,
    "ts_days_since": ts_days_since_kernel,
    "ts_min_if": ts_min_if_kernel,
    "ts_max_if": ts_max_if_kernel,
    "ts_quantile_if": ts_quantile_if_kernel,
}


class _R69BatchN1Native(Operator):
    _HANDLES_CALL_CONTRACT = True

    def __init__(self, canonical: str, metadata: Any, kernel):
        self._canonical, self.metadata, self._kernel = canonical, metadata, kernel

    def calculate(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
        args, kwargs = self._prepare_call(args, kwargs)
        bound = dict(zip(self.metadata.param_names, args))
        bound.update(kwargs)
        return self._kernel(bound)

    _calculate_series = calculate


def register_r69_native_batch_n1() -> tuple[str, ...]:
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
        operator = _R69BatchN1Native(canonical, metadata, kernel)
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
                f"{canonical}:pandas-authority-parity:r69-batch-n1:v1".encode()
            ).hexdigest(),
            notes=(
                "Pure Polars expression kernel (rolling / when / cum chains); "
                "no pandas conversion, no NumPy, no UDF."
            ),
        )
        migration = replace_backend(
            canonical, "polars",
            reason="R69 batch N1 strict pure-polars conversion replaces numpy-kernel slot",
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
            "supports_group": False,
            "supports_window": False,
            "supports_min_periods": False,
        })
        record_backend_replacement_after(migration, canonical, "polars", source=_SOURCE)
        registered.append(canonical)
    return tuple(registered)


__all__ = [
    "register_r69_native_batch_n1",
    *_KERNELS.keys(),
]
