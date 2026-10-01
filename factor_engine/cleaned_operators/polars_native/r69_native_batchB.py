# -*- coding: utf-8 -*-
"""R69 batch B: pure native polars kernels for 60 canonicals (rank 61-120).

R69 strict rules (see /tmp/r69/PROTOCOL.md):

* kernels are pure ``pl`` expression chains only -- NO numpy kernels, no
  ``to_pandas`` / ``from_pandas`` / ``map_elements`` / ``iterrows``;
* every canonical registers ``execution_kind=ExecutionKind.POLARS_NATIVE_EXPR``
  on the production ``polars`` slot (auto never falls back to pandas);
* the old numpy kernel code is deleted from the tree once the native slot is
  live (per-canonical deletion edits, batch report tracks them).
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Callable

import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import Operator, PANEL_SKIP_COLUMNS

_SOURCE = "factor_engine.cleaned_operators.polars_native.r69_native_batchB"

_SKIP = frozenset(PANEL_SKIP_COLUMNS) if PANEL_SKIP_COLUMNS else frozenset({
    "__fe_time__", "date", "timestamp", "trade_date", "datetime",
    "stock_code", "instrument", "symbol", "session", "__fe_instrument__",
})

_CCI_SCALE = 0.015
_EPS = 1e-12
_DISTINCT_CAP = 512


# ---------------------------------------------------------------------------
# shared helpers (pure polars; no pandas / numpy anywhere)
# ---------------------------------------------------------------------------
def _frame(v: Any, name: str) -> pl.DataFrame:
    if isinstance(v, pl.Series):
        return v.to_frame()
    if not isinstance(v, pl.DataFrame):
        raise TypeError(f"{name} must be a polars DataFrame or Series")
    return v


def _cols(f: pl.DataFrame) -> list[str]:
    return [c for c in f.columns if c not in _SKIP]


def _pi(v: Any, name: str, minimum: int = 1) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return v


def _finite(c: str) -> pl.Expr:
    """Cast to Float64; non-finite (NaN / ±Inf) -> null."""
    x = pl.col(c).cast(pl.Float64, strict=False)
    return pl.when(x.is_finite()).then(x).otherwise(None)


def _check_no_infinite(f: pl.DataFrame, cols: list[str], canonical: str) -> None:
    bad = f.select(
        [pl.col(c).cast(pl.Float64, strict=False).is_infinite().any().alias(c) for c in cols]
    )
    if any(bool(b) for b in bad.row(0)):
        raise ValueError(
            f"{canonical}: panel must contain finite state codes or NaN; ±Inf is out-of-domain"
        )


def _zip_frames(frames: list[pl.DataFrame], cols: list[str]) -> pl.DataFrame:
    """Horizontally bind same-named value columns with a @k role suffix."""
    renamed = []
    for k, f in enumerate(frames):
        ren = {c: f"{c}@{k}" for c in f.columns if c not in _SKIP}
        renamed.append(f.select(list(ren.keys())).rename(ren))
    return pl.concat(renamed, how="horizontal")


def _native_op(canonical: str, metadata, kernel: Callable[[dict], pl.DataFrame], spec_note: str) -> Operator:
    class _Native(Operator):
        _HANDLES_CALL_CONTRACT = True

        def __init__(self, c, m, k):
            self._canonical, self.metadata, self._kernel = c, m, k

        def calculate(self, *a, **kw):
            a, kw = self._prepare_call(a, kw)
            b = dict(zip(self.metadata.param_names, a))
            b.update(kw)
            return self._kernel(b)

        _calculate_series = calculate

    op = _Native(canonical, metadata, kernel)
    source_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    domain_hash = hashlib.sha256(
        repr((metadata.param_names, metadata.param_specs)).encode("utf-8")
    ).hexdigest()
    semantic_hash = hashlib.sha256(
        f"{canonical}:pandas-authority-parity:r69-batchB:v1".encode("utf-8")
    ).hexdigest()
    op._physical_spec = PhysicalImplementationSpec(
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
        implementation_source_hash=source_hash,
        emitter_identity=f"{_SOURCE}:pl.Expr:v1",
        kernel_identity=f"{_SOURCE}._KERNELS:{canonical}",
        kernel_signature=repr((metadata.param_names, metadata.param_specs)),
        parameter_domain_hash=domain_hash,
        semantic_contract_hash=semantic_hash,
        notes=spec_note,
    )
    return op


def register_r69_batch_b() -> tuple[str, ...]:
    """Swap the polars slots of this batch onto pure native kernels."""
    from factor_engine.cleaned_operators import (
        record_backend_replacement_after,
        replace_backend,
    )
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() != "building":
        return ()
    done: list[str] = []
    for canonical, kernel in _KERNELS.items():
        ref = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
        cur = OperatorRegistry.get(canonical, "polars", mode="any")
        if ref is None or cur is None:
            continue
        spec = getattr(cur, "_physical_spec", None)
        if spec is not None and spec.execution_kind == ExecutionKind.POLARS_NATIVE_EXPR:
            continue  # already converted (group A / newer batch)
        spec_note = (
            "R69 batch B pure native polars kernel (pl expression chains only; "
            "no pandas/numpy conversion, no per-column numpy UDF). "
            "Eager panel API only: no lazy/streaming claim."
        )
        op = _native_op(canonical, copy.deepcopy(ref.metadata), kernel, spec_note)
        mig = replace_backend(
            canonical, "polars",
            reason="R69 batch B: replace numpy kernel with pure native polars expressions",
            source=_SOURCE,
        )
        OperatorRegistry.register(op, canonical=canonical, backend="polars", source=_SOURCE)
        record_backend_replacement_after(mig, canonical, "polars", source=_SOURCE)
        done.append(canonical)
    return tuple(done)


# ---------------------------------------------------------------------------
# 61. bias -- 100 * (close - SMA) / SMA over a full-finite rolling window
# ---------------------------------------------------------------------------
def _k_bias(b: dict) -> pl.DataFrame:
    f = _frame(b["close"], "close")
    w = _pi(b.get("window", 6), "window", 1)
    exprs = []
    for c in _cols(f):
        x = _finite(c)
        ma = x.rolling_mean(w, min_samples=w)
        exprs.append(
            pl.when(x.is_not_null() & ma.is_not_null() & (ma != 0.0))
            .then(100.0 * (x - ma) / ma)
            .otherwise(None)
            .alias(c)
        )
    return f.with_columns(exprs)


# ---------------------------------------------------------------------------
# 69. cci -- (TP - MA) / (0.015 * MAD), flat/zero-MAD windows -> NaN
# ---------------------------------------------------------------------------
def _k_cci(b: dict) -> pl.DataFrame:
    f = _frame(b["close"], "close")
    h_f = _frame(b["high"], "high")
    l_f = _frame(b["low"], "low")
    w = _pi(b.get("window", 14), "window", 2)
    cols = _cols(f)
    if _cols(h_f) != cols or _cols(l_f) != cols:
        raise ValueError("high/low/close panel columns must match")
    work = _zip_frames([h_f, l_f, f], cols).with_row_index("_row")
    tp_cols = {}
    for c in cols:
        tp = (_finite(f"{c}@0") + _finite(f"{c}@1") + _finite(f"{c}@2")) / 3.0
        work = work.with_columns(tp.alias(f"{c}__tp"))
        tp_cols[c] = f"{c}__tp"

    def _gate(t: pl.Expr) -> pl.Expr:
        return (pl.len() < w) | t.is_null().any()

    aggs = []
    for c in cols:
        t = pl.col(tp_cols[c])
        aggs.append(
            pl.when(_gate(t)).then(None).otherwise(t.mean()).alias(f"{c}__ma")
        )
        aggs.append(
            pl.when(_gate(t)).then(None).otherwise((t - t.mean()).abs().mean()).alias(f"{c}__mad")
        )
        aggs.append(
            pl.when(_gate(t)).then(None).otherwise((t.max() - t.min()) == 0.0).alias(f"{c}__flat")
        )
    rolled = work.rolling(index_column="_row", period=f"{w}i").agg(aggs)
    joined = work.select(["_row", *[tp_cols[c] for c in cols]]).join(rolled, on="_row")
    series = []
    for c in cols:
        t = pl.col(tp_cols[c])
        ma = pl.col(f"{c}__ma")
        mad = pl.col(f"{c}__mad")
        flat = pl.col(f"{c}__flat")
        val = (
            pl.when(t.is_not_null() & ma.is_not_null() & mad.is_not_null()
                    & (mad != 0.0) & (~flat).fill_null(False))
            .then((t - ma) / (_CCI_SCALE * mad))
            .otherwise(None)
            .alias(c)
        )
        series.append(joined.select(val)[c])
    return f.with_columns([pl.Series(c, s) for c, s in zip(cols, series)])


# ---------------------------------------------------------------------------
# 62. bvc_imbalance_ma -- (EMA(flow, fast) - EMA(flow, slow)) / EMA(vol, slow)
# ---------------------------------------------------------------------------
def _bvc_parts(x: pl.Expr, vol: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    prev = x.shift(1)
    valid = (
        x.is_not_null() & (x > 0.0)
        & prev.is_not_null() & (prev > 0.0)
        & vol.is_not_null() & (vol > 0.0)
    )
    sign = (
        pl.when(x > prev).then(1.0)
        .when(x < prev).then(-1.0)
        .otherwise(0.0)
    )
    flow = pl.when(valid).then(sign * vol).otherwise(None)
    vol_m = pl.when(valid).then(vol).otherwise(None)
    return flow, vol_m


def _pandas_ewm(x: pl.Expr, alpha: float, min_periods: int) -> pl.Expr:
    """pandas ``ewm(alpha, adjust=False, min_periods, ignore_na=False)`` port.

    At null rows pandas emits the carried running mean, so the polars ewm value
    is forward-filled and gated on the cumulative count of valid observations.
    """
    carried = x.ewm_mean(
        alpha=alpha, adjust=False, ignore_nulls=False, min_samples=min_periods,
    ).forward_fill()
    cnt = x.is_not_null().cast(pl.Int64).cum_sum()
    return pl.when(cnt >= min_periods).then(carried).otherwise(None)


def _k_bvc_imbalance_ma(b: dict) -> pl.DataFrame:
    f = _frame(b["close"], "close")
    fv = _frame(b["volume"], "volume")
    fast = _pi(b.get("fast_window", 5), "fast_window", 1)
    slow = _pi(b.get("slow_window", 40), "slow_window", 1)
    if fast >= slow:
        raise ValueError("fast_window must be < slow_window")
    cols = _cols(f)
    if _cols(fv) != cols:
        raise ValueError("close/volume panel columns must match")
    work = _zip_frames([f, fv], cols)
    af = 2.0 / (fast + 1.0)
    as_ = 2.0 / (slow + 1.0)
    out = []
    for c in cols:
        flow, vol_m = _bvc_parts(_finite(f"{c}@0"), _finite(f"{c}@1"))
        ef = _pandas_ewm(flow, af, fast)
        es = _pandas_ewm(flow, as_, slow)
        den = _pandas_ewm(vol_m, as_, slow)
        out.append(
            work.select(
                pl.when(den.is_not_null() & (den > 0.0))
                .then((ef - es) / den)
                .otherwise(None)
                .alias(c)
            )[c]
        )
    return f.with_columns([pl.Series(c, s) for c, s in zip(cols, out)])


# ---------------------------------------------------------------------------
# 63. bvc_sign_pct -- sum(sign(dclose)*vol) / sum(vol) over full valid window
# ---------------------------------------------------------------------------
def _k_bvc_sign_pct(b: dict) -> pl.DataFrame:
    f = _frame(b["close"], "close")
    fv = _frame(b["volume"], "volume")
    w = _pi(b.get("window", 20), "window", 2)
    cols = _cols(f)
    if _cols(fv) != cols:
        raise ValueError("close/volume panel columns must match")
    work = _zip_frames([f, fv], cols)
    out = []
    for c in cols:
        flow, vol_m = _bvc_parts(_finite(f"{c}@0"), _finite(f"{c}@1"))
        num = flow.rolling_sum(w, min_samples=w)
        den = vol_m.rolling_sum(w, min_samples=w)
        out.append(
            work.select(
                pl.when(den.is_not_null() & (den > 0.0)).then(num / den).otherwise(None).alias(c)
            )[c]
        )
    return f.with_columns([pl.Series(c, s) for c, s in zip(cols, out)])


# ---------------------------------------------------------------------------
# 64. cash_flow_lifecycle_stage -- signed-flow stage classifier 1..5
# ---------------------------------------------------------------------------
def _k_cash_flow_lifecycle_stage(b: dict) -> pl.DataFrame:
    f = _frame(b["operating"], "operating")
    iv_f = _frame(b["investing"], "investing")
    fn_f = _frame(b["financing"], "financing")
    cols = _cols(f)
    if _cols(iv_f) != cols or _cols(fn_f) != cols:
        raise ValueError("operating/investing/financing panel columns must match")
    work = _zip_frames([f, iv_f, fn_f], cols)
    out = []
    for c in cols:
        o = _finite(f"{c}@0")
        iv = _finite(f"{c}@1")
        fn = _finite(f"{c}@2")
        stage = (
            pl.when((o > 0.0) & (iv < 0.0) & (fn < 0.0)).then(1.0)
            .when((o > 0.0) & (iv < 0.0) & (fn >= 0.0)).then(2.0)
            .when((o <= 0.0) & (iv < 0.0)).then(3.0)
            .when((o <= 0.0) & (iv >= 0.0)).then(4.0)
            .when((o > 0.0) & (iv >= 0.0) & (fn < 0.0)).then(5.0)
            .otherwise(None)
        )
        out.append(work.select(stage.alias(c))[c])
    return f.with_columns([pl.Series(c, s) for c, s in zip(cols, out)])


# ---------------------------------------------------------------------------
# 65. category_frequency -- share of the full-finite window equal to the
#     current state code (any NaN in the window -> NaN)
# ---------------------------------------------------------------------------
def _k_category_frequency(b: dict) -> pl.DataFrame:
    f = _frame(b["state"], "state")
    w = _pi(b.get("window", 2), "window", 2)
    cols = _cols(f)
    _check_no_infinite(f, cols, "category_frequency")
    work = f.with_row_index("_row")
    out = []
    for c in cols:
        res = work.rolling(index_column="_row", period=f"{w}i").agg(
            (pl.when(pl.len() < w).then(None)
             .when(_finite(c).is_null().any()).then(None)
             .otherwise((_finite(c) == _finite(c).last()).sum().cast(pl.Float64) / float(w)))
            .alias(c),
        )
        out.append(res[c])
    return f.with_columns([pl.Series(c, s) for c, s in zip(cols, out)])


# ---------------------------------------------------------------------------
# 66. category_transition_surprise -- standardized transition-count z-score
# ---------------------------------------------------------------------------
def _k_category_transition_surprise(b: dict) -> pl.DataFrame:
    f = _frame(b["state"], "state")
    w = _pi(b.get("window", 20), "window", 2)
    cols = _cols(f)
    _check_no_infinite(f, cols, "category_transition_surprise")
    work = f.with_row_index("_row")
    out = []
    for c in cols:
        x = _finite(c)
        ind = (x != x.shift(1)).cast(pl.Int64)
        trans = ind.rolling_sum(w - 1, min_samples=w - 1)
        distinct = (
            work.rolling(index_column="_row", period=f"{w}i")
            .agg(pl.when(_finite(c).is_null().any()).then(None)
                 .otherwise(_finite(c).n_unique().cast(pl.Int64))
                 .alias(c))
        )[c]
        d_expr = distinct.cast(pl.Float64)
        expected = (pl.lit(float(w - 1)) * (pl.lit(1.0) - pl.lit(1.0) / d_expr))
        variance = (pl.lit(float(w - 1)) * (pl.lit(1.0) / d_expr) * (pl.lit(1.0) - pl.lit(1.0) / d_expr))
        full = x.is_not_null().rolling_sum(w, min_samples=1).eq(w)
        z = (trans.cast(pl.Float64) - expected) / variance.pow(0.5)
        out.append(
            work.select(
                pl.when(full & distinct.is_not_null() & (distinct > 1) & (variance > _EPS))
                .then(z)
                .otherwise(None)
                .alias(c)
            )[c]
        )
    return f.with_columns([pl.Series(c, s) for c, s in zip(cols, out)])


# ---------------------------------------------------------------------------
# 67. capital_change_age -- rows since the change_date-implied index position
# ---------------------------------------------------------------------------
def _capital_change_age_col(frame: pl.DataFrame, work: pl.DataFrame, tcol: str, c: str) -> pl.Series:
    idx_dtype = frame.schema[tcol]
    dt = frame.schema[c]
    raw = pl.col(c)
    if dt == idx_dtype:
        cd = raw.cast(idx_dtype, strict=False)
    elif dt == pl.String:
        unit = getattr(idx_dtype, "time_unit", None) or "us"
        parsed = raw.str.to_datetime(strict=False, time_unit=unit, exact=False)
        if getattr(idx_dtype, "time_zone", None):
            parsed = parsed.dt.replace_time_zone(str(idx_dtype.time_zone))
        cd = parsed.cast(idx_dtype, strict=False)
    elif dt.is_numeric():
        # pandas authority: Timestamp(<number>) == ns since epoch, then day floor
        cd = (
            raw.cast(pl.Float64, strict=False).floor().cast(pl.Int64)
            .cast(pl.Datetime("ns")).dt.truncate("1d")
            .cast(idx_dtype, strict=False)
        )
    else:
        cd = raw.cast(idx_dtype, strict=False)
    idx = frame.select(pl.col(tcol).alias("_t")).with_row_index("_pos").sort("_t")
    probe = (
        work.select(["_i", cd.alias("_cd")])
        .drop_nulls(["_cd"])
        .sort("_cd")
        .join_asof(idx, left_on="_cd", right_on="_t", strategy="forward")
        .select(["_i", "_pos"])
    )
    j = (
        work.select(["_i", cd.alias("_cd")])
        .join(probe, on="_i", how="left")
    )
    has = j["_cd"].is_not_null()
    vflag = has & j["_pos"].is_not_null() & (j["_pos"] <= j["_i"])
    rflag = has & (j["_pos"].is_null() | (j["_pos"] > j["_i"]))
    last_v = pl.when(vflag).then(j["_i"]).otherwise(None).forward_fill()
    last_r = pl.when(rflag).then(j["_i"]).otherwise(None).forward_fill()
    pos_v = pl.when(vflag).then(j["_pos"]).otherwise(None).forward_fill()
    age = (
        pl.when(last_v.is_not_null() & (last_r.is_null() | (last_v > last_r)))
        .then((j["_i"] - pos_v).cast(pl.Float64))
        .otherwise(None)
    )
    return j.select(age.alias(c))[c]


def _k_capital_change_age(b: dict) -> pl.DataFrame:
    frame = _frame(b["change_date"], "change_date")
    for tcol in ("__fe_time__", "date", "timestamp", "trade_date", "datetime"):
        if tcol in frame.columns:
            break
    else:
        raise ValueError("capital_change_age: panel has no time axis")
    cols = _cols(frame)
    work = frame.with_row_index("_i")
    series = [_capital_change_age_col(frame, work, tcol, c) for c in cols]
    return frame.with_columns([pl.Series(c, s) for c, s in zip(cols, series)])


# ---------------------------------------------------------------------------
# 68. composition_aitchison_distance -- ||clr(A) - clr(B)||_2
# ---------------------------------------------------------------------------
def _comp_parts(b: dict, prefix: str) -> list[pl.DataFrame]:
    return [
        _frame(b[f"{prefix}{i}"], f"{prefix}{i}")
        for i in range(1, 7) if b.get(f"{prefix}{i}") is not None
    ]


def _k_composition_aitchison_distance(b: dict) -> pl.DataFrame:
    a = _comp_parts(b, "x")
    bb = _comp_parts(b, "y")
    if len(a) < 2 or len(bb) < 2 or len(a) != len(bb):
        raise ValueError(
            "composition_aitchison_distance requires equal-size compositions (2..6 parts each)"
        )
    base = a[0]
    cols = _cols(base)
    for p in a + bb:
        if _cols(p) != cols:
            raise ValueError("composition parts must share panel columns")
    all_parts = a + bb
    na = float(len(a))
    nb = float(len(bb))
    work = _zip_frames(all_parts, cols)
    k_max = len(all_parts)
    out = []
    for c in cols:
        logs = [_finite(f"{c}@{k}").log() for k in range(k_max)]
        finite_all = pl.lit(True)
        for e in logs:
            finite_all = finite_all & e.is_finite()
        mean_a = None
        for e in logs[:len(a)]:
            mean_a = e if mean_a is None else (mean_a + e)
        mean_a = mean_a / na
        mean_b = None
        for e in logs[len(a):]:
            mean_b = e if mean_b is None else (mean_b + e)
        mean_b = mean_b / nb
        sq = None
        for ea, eb in zip(logs[:len(a)], logs[len(a):]):
            d = (ea - mean_a) - (eb - mean_b)
            sq = d * d if sq is None else (sq + d * d)
        out.append(
            work.select(pl.when(finite_all).then(sq.sqrt()).otherwise(None).alias(c))[c]
        )
    return base.with_columns([pl.Series(c, s) for c, s in zip(cols, out)])


# ---------------------------------------------------------------------------
# 69. candlestick_pattern -- TA-Lib style single-pattern detector
# ---------------------------------------------------------------------------
def _emit(mask: pl.Expr, direction: pl.Expr | float, validmask: pl.Expr) -> pl.Expr:
    return (
        pl.when(validmask & mask).then(direction)
        .when(validmask).then(0.0)
        .otherwise(None)
    )


def _emit_detect_only(mask: pl.Expr, direction: pl.Expr | float, validmask: pl.Expr) -> pl.Expr:
    """Old kernels for 2_crows / evening_doji_star emit NaN (not 0.0) on
    valid-but-not-detected rows (detect-only masking, no zero base)."""
    return pl.when(validmask & mask).then(direction).otherwise(None)


def _candle_col(c: str, p: str, pen: float, bw: int, sw: int) -> pl.Expr:
    o = _finite(f"{c}@0")
    h = _finite(f"{c}@1")
    low = _finite(f"{c}@2")
    cl = _finite(f"{c}@3")

    def mx(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return pl.max_horizontal(a, b)

    def mn(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return pl.min_horizontal(a, b)

    valid = (
        (o > 0.0) & (h > 0.0) & (low > 0.0) & (cl > 0.0)
        & (h >= low) & (h >= mx(o, cl)) & (low <= mn(o, cl))
    )
    cur = (
        o.is_not_null() & h.is_not_null() & low.is_not_null() & cl.is_not_null()
        & valid & ((h - low) > (_EPS_BASE * cl.abs()))
    )
    body = (cl - o).abs()
    rng = h - low
    upper = h - mx(o, cl)
    lower = mn(o, cl) - low
    bavg = body.shift(1).rolling_mean(bw, min_samples=bw)
    ravg = rng.shift(1).rolling_mean(sw, min_samples=sw)
    tol = 0.1 * ravg
    bull = cl > o
    bear = cl < o
    doji = body <= 0.1 * bavg
    long_body = body >= 1.3 * bavg
    short_body = body <= 0.6 * bavg
    long_upper = upper >= 0.8 * ravg
    long_lower = lower >= 0.8 * ravg
    po, pc, ph, pl_ = o.shift(1), cl.shift(1), h.shift(1), low.shift(1)
    pbull, pbear = pc > po, pc < po
    pbody = (pc - po).abs()
    prev1 = (
        po.is_not_null() & pc.is_not_null() & ph.is_not_null() & pl_.is_not_null()
        & valid.shift(1).fill_null(False)
    )
    o2, c2, h2, l2 = o.shift(2), cl.shift(2), h.shift(2), low.shift(2)
    o3, c3, h3, l3 = o.shift(3), cl.shift(3), h.shift(3), low.shift(3)
    o4, c4, h4, l4 = o.shift(4), cl.shift(4), h.shift(4), low.shift(4)
    prev2 = prev1 & o2.is_not_null() & c2.is_not_null() & h2.is_not_null() & l2.is_not_null() & valid.shift(2).fill_null(False)
    prev3 = prev2 & o3.is_not_null() & c3.is_not_null() & h3.is_not_null() & l3.is_not_null() & valid.shift(3).fill_null(False)
    prev4 = prev3 & o4.is_not_null() & c4.is_not_null() & h4.is_not_null() & l4.is_not_null() & valid.shift(4).fill_null(False)
    has_b = bavg.is_not_null()
    has_r = ravg.is_not_null()
    has_b1 = bavg.shift(1).is_not_null()
    has_b2 = bavg.shift(2).is_not_null()
    sign_dir = pl.when(cl > o).then(1.0).when(cl < o).then(-1.0).otherwise(0.0)

    def near(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return (a - b).abs() <= tol

    def sgn(cond: pl.Expr) -> pl.Expr:
        return pl.when(cond).then(1.0).otherwise(-1.0)

    if p == "2_crows":
        bull2 = c2 > o2
        mask = bull2 & pbear & bear & (pl_ > h2) & (o > po) & (cl < c2) & (cl > o2)
        return _emit_detect_only(mask, -1.0, cur & prev1 & prev2)
    if p == "evening_doji_star":
        doji1r = pbody <= 0.1 * bavg.shift(1)
        mid = o2 + (c2 - o2) * (1.0 - pen)
        mask = (c2 > o2) & doji1r & (pl_ > h2) & bear & (cl < mid)
        return _emit_detect_only(mask, -1.0, cur & prev1 & prev2 & has_b1)
    if p == "hammer":
        safe_body = mx(body, pl.lit(1e-12))
        mask = (body <= 0.35 * rng) & (lower >= 2.0 * body) & (upper <= 0.35 * safe_body)
        return _emit(mask, 1.0, cur)
    if p == "long_line":
        return _emit(long_body, sign_dir, cur & has_b)
    if p == "short_line":
        return _emit(short_body, sign_dir, cur & has_b)
    if p == "high_wave":
        return _emit(short_body & long_upper & long_lower, sign_dir, cur & has_b & has_r)
    if p == "long_legged_doji":
        return _emit(doji & long_upper & long_lower, 1.0, cur & has_b & has_r)
    if p == "rickshaw_man":
        m = ((o + cl) / 2.0 - (h + low) / 2.0).abs() <= 0.15 * rng
        return _emit(doji & long_upper & long_lower & m, 1.0, cur & has_b & has_r)
    if p == "takuri":
        return _emit(doji & (lower >= 2.5 * bavg) & (upper <= 0.3 * bavg), 1.0, cur & has_b)
    if p == "belt_hold":
        mask = long_body & ((bull & ((o - low) <= 0.1 * rng)) | (bear & ((h - o) <= 0.1 * rng)))
        return _emit(mask, sign_dir, cur & has_b)
    if p == "closing_marubozu":
        mask = long_body & ((bull & ((h - cl) <= 0.05 * rng)) | (bear & ((cl - low) <= 0.05 * rng)))
        return _emit(mask, sign_dir, cur & has_b)
    if p == "homing_pigeon":
        return _emit(pbear & bear & (o < po) & (cl > pc), 1.0, cur & prev1)
    if p == "matching_low":
        return _emit(pbear & bear & near(cl, pc), 1.0, cur & prev1 & has_r)
    if p == "counterattack":
        return _emit(((pbear & bull) | (pbull & bear)) & near(cl, pc), sign_dir, cur & prev1 & has_r)
    if p == "separating_lines":
        return _emit(
            ((pbear & bull) | (pbull & bear)) & near(o, po) & long_body,
            sign_dir, cur & prev1 & has_r & has_b,
        )
    if p in {"on_neck", "in_neck", "thrusting"}:
        prev_mid = (po + pc) / 2.0
        bull2n = pbear & bull & (o < pc)
        if p == "on_neck":
            mask = bull2n & near(cl, pc)
        elif p == "in_neck":
            mask = bull2n & (cl > pc) & (cl < pc + 0.25 * pbody)
        else:
            mask = bull2n & (cl > pc + 0.25 * pbody) & (cl < prev_mid)
        return _emit(mask, -1.0, cur & prev1)
    if p == "doji_star":
        mask = doji & ((pbear & (h < pc)) | (pbull & (low > pc)))
        return _emit(mask, sgn(pc > po), cur & prev1 & has_b)
    bull2 = c2 > o2
    bear2 = c2 < o2
    body2 = (c2 - o2).abs()
    doji1 = doji.shift(1).fill_null(False)
    doji2 = doji.shift(2).fill_null(False)
    b1_hi, b1_lo = mx(o2, c2), mn(o2, c2)
    b2_hi, b2_lo = mx(po, pc), mn(po, pc)
    harami1 = (bull2 | bear2) & (b2_hi <= b1_hi) & (b2_lo >= b1_lo)
    engulf1 = (
        (bear2 & pbull & (po <= c2) & (pc >= o2))
        | (bull2 & pbear & (po >= c2) & (pc <= o2))
    )
    if p == "3_inside":
        mask = harami1 & ((bear2 & (cl > o2)) | (bull2 & (cl < o2)))
        return _emit(mask, sgn(cl > o2), cur & prev1 & prev2)
    if p == "3_outside":
        mask = engulf1 & ((bear2 & (cl > pc)) | (bull2 & (cl < pc)))
        return _emit(mask, sgn(cl > pc), cur & prev1 & prev2)
    if p == "tristar":
        return _emit(doji & doji1 & doji2, sgn(cl > cl.shift(2)), cur & prev2 & has_b & has_b1 & has_b2)
    if p == "abandoned_baby":
        bullmask = bear2 & doji1 & (ph < l2) & bull & (low > ph) & (cl > o2 - body2 * pen)
        bearmask = bull2 & doji1 & (pl_ > h2) & bear & (h < pl_) & (cl < o2 + body2 * pen)
        return _emit(bullmask | bearmask, sgn(bullmask), cur & prev1 & prev2 & has_b1)
    if p == "stick_sandwich":
        return _emit(bear2 & pbull & bear & near(cl, c2), 1.0, cur & prev2 & has_r)
    if p == "identical_3_crows":
        return _emit(bear & pbear & bear2 & near(o, pc) & near(po, c2), -1.0, cur & prev1 & prev2 & has_r)
    if p == "advance_block":
        mask = bull & pbull & bull2 & (body < body.shift(1)) & (body.shift(1) < body.shift(2))
        return _emit(mask, -1.0, cur & prev2 & has_b & has_b1 & has_b2)
    if p == "stalled_pattern":
        mask = bull & pbull & bull2 & short_body & (body.shift(1) >= body.shift(2))
        return _emit(mask, -1.0, cur & prev2 & has_b & has_b1 & has_b2)
    if p == "3_stars_south":
        mask = (bear & pbear & bear2 & (low > pl_) & (pl_ > l2)
                & (body < body.shift(1)) & (body.shift(1) < body.shift(2)))
        return _emit(mask, 1.0, cur & prev1 & prev2 & has_b & has_b1 & has_b2)
    if p == "unique_3_river":
        mask = bear2 & pbear & bull & (l2 > pl_) & (low > pl_) & short_body
        return _emit(mask, 1.0, cur & prev1 & prev2 & has_b)
    if p == "upside_gap_2_crows":
        mask = bull2 & pbear & bear & (pl_ > h2) & (o > po) & (cl < c2)
        return _emit(mask, -1.0, cur & prev1 & prev2)
    if p == "tasuki_gap":
        up = bull2 & pbull & bear & (low.shift(1) > h2) & (o > pc) & (cl < o.shift(1)) & (cl > h2)
        dn = bear2 & pbear & bull & (h.shift(1) < l2) & (o < pc) & (cl > o.shift(1)) & (cl < l2)
        return _emit(up | dn, sgn(up), cur & prev1 & prev2)
    if p == "xside_gap_3_methods":
        up = bull2 & pbull & bear & (low.shift(1) > h2) & (o > c2) & (cl < c2)
        dn = bear2 & pbear & bull & (h.shift(1) < l2) & (o < c2) & (cl > c2)
        return _emit(up | dn, sgn(up), cur & prev1 & prev2)
    bull3 = c3 > o3
    if p == "3_line_strike":
        up = bull3 & bull.shift(2).fill_null(False) & pbull & bear & (cl < o3) & (o > pc)
        dn = bear3 & bear.shift(2).fill_null(False) & pbear & bull & (cl > o3) & (o < pc)
        return _emit(up | dn, sgn(~up), cur & prev1 & prev2 & prev3)
    if p == "rise_fall_3_methods":
        bull4, bear4 = c4 > o4, c4 < o4
        inside3 = (
            (h.shift(3) < h.shift(4)) & (low.shift(3) > low.shift(4))
            & (h.shift(2) < h.shift(4)) & (low.shift(2) > low.shift(4))
            & (h.shift(1) < h.shift(4)) & (low.shift(1) > low.shift(4))
        )
        up = bull4 & inside3 & bull & (cl > c4)
        dn = bear4 & inside3 & bear & (cl < c4)
        return _emit(up | dn, sgn(up), cur & prev1 & prev2 & prev3 & prev4)
    if p == "ladder_bottom":
        mask = (c4 < o4) & bear3 & bear2 & pbear & bull & (cl > po)
        return _emit(mask, 1.0, cur & prev1 & prev2 & prev3 & prev4)
    if p == "breakaway":
        up = (c4 < o4) & (h.shift(3) < low.shift(4)) & bull & (cl > cl.shift(2))
        dn = (c4 > o4) & (low.shift(3) > h.shift(4)) & bear & (cl < cl.shift(2))
        return _emit(up | dn, sgn(up), cur & prev2 & prev3 & prev4)
    if p == "mat_hold":
        mask = (c4 > o4) & (low.shift(3) > c4) & bull & (cl > c4)
        return _emit(mask, 1.0, cur & prev3 & prev4)
    raise ValueError(f"unsupported candlestick pattern: {p!r}")


_EPS_BASE = 1e-6


def _k_candlestick_pattern(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.price_volume.candle_pattern_engine_v2 import _CANDLE_PARAM_DEPS
    o_f = _frame(b["open"], "open")
    h_f = _frame(b["high"], "high")
    l_f = _frame(b["low"], "low")
    c_f = _frame(b["close"], "close")
    bw = _pi(b.get("body_window", 10), "body_window", 2)
    sw = _pi(b.get("shadow_window", 10), "shadow_window", 2)
    pen = b.get("penetration", 0.3)
    if isinstance(pen, bool) or not isinstance(pen, (int, float)):
        raise ValueError("penetration must be in [0,1]")
    pen = float(pen)
    if not 0 <= pen <= 1:
        raise ValueError("penetration must be in [0,1]")
    p = str(b.get("pattern", "")).strip().lower().removeprefix("cdl_")
    if p not in _CANDLE_PARAM_DEPS:
        raise ValueError(f"unsupported candlestick pattern: {b.get('pattern')!r}")
    cols = _cols(o_f)
    for fr in (h_f, l_f, c_f):
        if _cols(fr) != cols:
            raise ValueError("open/high/low/close panel columns must match")
    work = _zip_frames([o_f, h_f, l_f, c_f], cols)
    out = []
    for c in cols:
        out.append(work.select(_candle_col(c, p, pen, bw, sw).alias(c))[c])
    return o_f.with_columns([pl.Series(c, s) for c, s in zip(cols, out)])


_KERNELS: dict[str, Callable[[dict], pl.DataFrame]] = {
    "bias": _k_bias,
    "cci": _k_cci,
    "bvc_imbalance_ma": _k_bvc_imbalance_ma,
    "bvc_sign_pct": _k_bvc_sign_pct,
    "cash_flow_lifecycle_stage": _k_cash_flow_lifecycle_stage,
    "category_frequency": _k_category_frequency,
    "category_transition_surprise": _k_category_transition_surprise,
    "capital_change_age": _k_capital_change_age,
    "composition_aitchison_distance": _k_composition_aitchison_distance,
    "candlestick_pattern": _k_candlestick_pattern,
}

__all__ = ["register_r69_batch_b"]
