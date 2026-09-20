# -*- coding: utf-8 -*-
"""Native Polars implementations for parameterized technical indicators."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _pi(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _pf(value, name: str, minimum: float | None = None) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _cols(*frames: pl.DataFrame) -> list[str]:
    out = [c for c in frames[0].columns if c not in _SKIP]
    for frame in frames[1:]:
        out = [c for c in out if c in frame.columns]
    return out


def _result(base: pl.DataFrame, values: dict[str, pl.Series]) -> pl.DataFrame:
    return base.with_columns([series.alias(name) for name, series in values.items()])


def _one(frame: pl.DataFrame, column: str, expr: pl.Expr) -> pl.Series:
    return frame.select(expr.alias(column)).to_series()


def _ema(expr: pl.Expr, span: int) -> pl.Expr:
    span = _pi(span, "span")
    # Normalize NaN to null, then preserve pandas ignore_na=False weighting.
    return (
        expr.fill_nan(None)
        .ewm_mean(
            span=span, adjust=False, min_samples=span, ignore_nulls=False
        )
        .fill_null(strategy="forward")
    )


def _ema_numpy(values, span: int) -> np.ndarray:
    """Deterministic pandas ``ewm(adjust=False, ignore_na=False)`` parity."""
    span = _pi(span, "span")
    alpha = 2.0 / (span + 1.0)
    old_wt_factor = 1.0 - alpha
    data = np.asarray(values, dtype=float)
    out = np.full(data.shape, np.nan, dtype=float)
    weighted = np.nan
    old_wt = 1.0
    observations = 0
    for index, value in enumerate(data):
        observed = np.isfinite(value)
        if observed:
            observations += 1
        if np.isfinite(weighted):
            old_wt *= old_wt_factor
            if observed:
                if weighted != value:
                    weighted = (old_wt * weighted + alpha * value) / (old_wt + alpha)
                old_wt = 1.0
        elif observed:
            weighted = value
        if observations >= span:
            out[index] = weighted
    return out


def _wilder(expr: pl.Expr, window: int) -> pl.Expr:
    window = _pi(window, "window")
    return (
        expr.fill_nan(None)
        .ewm_mean(
            alpha=1.0 / window,
            adjust=False,
            min_samples=window,
            ignore_nulls=False,
        )
        .fill_null(strategy="forward")
    )


def _tr_expr() -> pl.Expr:
    previous = pl.col("close").shift(1)
    raw = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - previous).abs(),
        (pl.col("low") - previous).abs(),
    ).fill_nan(None)
    invalid = (
        previous.is_null()
        | previous.is_nan()
        | pl.col("high").is_null()
        | pl.col("high").is_nan()
        | pl.col("low").is_null()
        | pl.col("low").is_nan()
    )
    return pl.when(invalid).then(None).otherwise(raw)


def _ohlc(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.DataFrame:
    return pl.DataFrame({"high": high, "low": low, "close": close})


def _dmi(high, low, close, window):
    window = _pi(window, "window", 2)
    out_plus: dict[str, pl.Series] = {}
    out_minus: dict[str, pl.Series] = {}
    for column in _cols(high, low, close):
        frame = _ohlc(high[column], low[column], close[column])
        up = pl.col("high").diff()
        down = -pl.col("low").diff()
        # 缺口后的首根 bar（up/down 缺失）必须掩为 NaN（"cannot judge"），
        # 不能落成 0.0 混进 Wilder 平滑——与 pandas 参考的
        # ``up.where(cond, 0.0).where(valid)`` 语义一致。
        valid = up.is_not_null() & up.is_not_nan() & down.is_not_null() & down.is_not_nan()
        plus = (
            pl.when(valid)
            .then(pl.when((up > down) & (up > 0)).then(up).otherwise(0.0))
            .otherwise(None)
        )
        minus = (
            pl.when(valid)
            .then(pl.when((down > up) & (down > 0)).then(down).otherwise(0.0))
            .otherwise(None)
        )
        atr = _wilder(_tr_expr(), window)
        p = pl.when(atr != 0).then(100.0 * _wilder(plus, window) / atr).otherwise(None)
        m = pl.when(atr != 0).then(100.0 * _wilder(minus, window) / atr).otherwise(None)
        selected = frame.select(p.alias("plus"), m.alias("minus"))
        out_plus[column], out_minus[column] = selected["plus"], selected["minus"]
    return _result(close, out_plus), _result(close, out_minus)


def DMI_plus(high, low, close, window):
    return _dmi(high, low, close, window)[0]


def DMI_minus(high, low, close, window):
    return _dmi(high, low, close, window)[1]


def DX(high, low, close, window):
    plus, minus = _dmi(high, low, close, window)
    values = {}
    for column in _cols(plus, minus):
        frame = pl.DataFrame({"plus": plus[column], "minus": minus[column]})
        denominator = pl.col("plus") + pl.col("minus")
        values[column] = _one(
            frame,
            column,
            pl.when(denominator != 0)
            .then(100.0 * (pl.col("plus") - pl.col("minus")).abs() / denominator)
            .otherwise(None),
        )
    return _result(close, values)


def NATR(high, low, close, window):
    values = {}
    for column in _cols(high, low, close):
        frame = _ohlc(high[column], low[column], close[column])
        # R5-38: close is PositivePrice — a non-positive close is bad data; the
        # pandas backend masks it to NaN instead of ``abs()``, so the polars
        # backend must not launder it either.
        denominator = pl.when(pl.col("close") > 0.0).then(pl.col("close")).otherwise(None)
        values[column] = _one(
            frame,
            column,
            pl.when(denominator.is_not_null())
            .then(100.0 * _wilder(_tr_expr(), window) / denominator)
            .otherwise(None),
        )
    return _result(close, values)


def _ppo(x, fast_window, slow_window):
    fast = _pi(fast_window, "fast_window")
    slow = _pi(slow_window, "slow_window")
    if fast >= slow:
        raise ValueError("fast_window must be < slow_window")
    values = {}
    for column in _cols(x):
        fast_ema, slow_ema = _ema(pl.col(column), fast), _ema(pl.col(column), slow)
        values[column] = _one(
            x,
            column,
            pl.when(slow_ema != 0).then(100.0 * (fast_ema - slow_ema) / slow_ema).otherwise(None),
        )
    return _result(x, values)


def PPO(close, fast_window, slow_window):
    return _ppo(close, fast_window, slow_window)


def PVO(volume, fast_window, slow_window):
    return _ppo(volume, fast_window, slow_window)


def _signal(oscillator, signal_window):
    signal_window = _pi(signal_window, "signal_window")
    return _result(
        oscillator,
        {c: _one(oscillator, c, _ema(pl.col(c), signal_window)) for c in _cols(oscillator)},
    )


def PPO_signal(close, fast_window, slow_window, signal_window):
    return _signal(PPO(close, fast_window, slow_window), signal_window)


def PPO_hist(close, fast_window, slow_window, signal_window):
    value = PPO(close, fast_window, slow_window)
    signal = _signal(value, signal_window)
    return _result(close, {c: value[c] - signal[c] for c in _cols(value, signal)})


def PVO_signal(volume, fast_window, slow_window, signal_window):
    return _signal(PVO(volume, fast_window, slow_window), signal_window)


def PVO_hist(volume, fast_window, slow_window, signal_window):
    value = PVO(volume, fast_window, slow_window)
    signal = _signal(value, signal_window)
    return _result(volume, {c: value[c] - signal[c] for c in _cols(value, signal)})


def CMO(close, window):
    window = _pi(window, "window", 2)
    values = {}
    for column in _cols(close):
        # pandas: d.clip(lower/upper=0) 保留 NaN，rolling(min_periods=w) 跳过 NaN。
        # polars 中 rolling_sum 忽略 null 但传播 NaN，故先把 diff 的 NaN 归一为 null。
        delta = pl.col(column).diff().fill_nan(None)
        up = delta.clip(lower_bound=0.0).rolling_sum(window, min_samples=window)
        down = (-delta).clip(lower_bound=0.0).rolling_sum(window, min_samples=window)
        denominator = up + down
        values[column] = _one(
            close,
            column,
            pl.when(denominator != 0).then(100.0 * (up - down) / denominator).otherwise(None),
        )
    return _result(close, values)


def _vortex(high, low, close, window, *, positive):
    window = _pi(window, "window", 2)
    values = {}
    for column in _cols(high, low, close):
        frame = _ohlc(high[column], low[column], close[column])
        vm = (
            (pl.col("high") - pl.col("low").shift(1)).abs()
            if positive
            else (pl.col("low") - pl.col("high").shift(1)).abs()
        ).fill_nan(None)
        vm = vm.rolling_sum(window, min_samples=window)
        true_range = _tr_expr().rolling_sum(window, min_samples=window)
        values[column] = _one(
            frame, column, pl.when(true_range != 0).then(vm / true_range).otherwise(None)
        )
    return _result(close, values)


def VortexPlus(high, low, close, window):
    return _vortex(high, low, close, window, positive=True)


def VortexMinus(high, low, close, window):
    return _vortex(high, low, close, window, positive=False)


def TSI(close, long_window, short_window):
    long_window = _pi(long_window, "long_window", 2)
    short_window = _pi(short_window, "short_window", 2)
    values = {}
    for column in _cols(close):
        raw = close[column].to_numpy().astype(float, copy=False)
        momentum = np.empty(raw.shape, dtype=float)
        momentum[0] = np.nan
        momentum[1:] = raw[1:] - raw[:-1]
        numerator = _ema_numpy(_ema_numpy(momentum, long_window), short_window)
        denominator = _ema_numpy(
            _ema_numpy(np.abs(momentum), long_window), short_window
        )
        result = np.full(raw.shape, np.nan, dtype=float)
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
        result[valid] = 100.0 * numerator[valid] / denominator[valid]
        values[column] = pl.Series(column, result)
    return _result(close, values)


def TSI_signal(close, long_window, short_window, signal_window):
    signal_window = _pi(signal_window, "signal_window")
    oscillator = TSI(close, long_window, short_window)
    return _result(
        oscillator,
        {
            column: pl.Series(column, _ema_numpy(oscillator[column], signal_window))
            for column in _cols(oscillator)
        },
    )


def UltimateOscillator(
    high,
    low,
    close,
    short_window,
    medium_window,
    long_window,
    short_weight=4.0,
    medium_weight=2.0,
    long_weight=1.0,
):
    short = _pi(short_window, "short_window", 2)
    medium = _pi(medium_window, "medium_window", 2)
    long = _pi(long_window, "long_window", 2)
    if not short < medium < long:
        raise ValueError("require short_window < medium_window < long_window")
    weights = (
        _pf(short_weight, "short_weight", 0),
        _pf(medium_weight, "medium_weight", 0),
        _pf(long_weight, "long_weight", 0),
    )
    denominator = sum(weights)
    if denominator <= 0:
        raise ValueError("oscillator weights must sum to > 0")
    values = {}
    for column in _cols(high, low, close):
        frame = _ohlc(high[column], low[column], close[column])
        previous = pl.col("close").shift(1).fill_nan(None)
        # pandas np.minimum/np.maximum 会传播 NaN；min_horizontal 会忽略 null，
        # 故用算术 min/max，使 previous 为 null 时整体为 null 并在 rolling 中被跳过。
        low_min = pl.when(2.0 != 0).then(((pl.col("low") + previous - (pl.col("low") - previous).abs())) / (2.0)).otherwise(None)
        high_max = pl.when(2.0 != 0).then(((pl.col("high") + previous + (pl.col("high") - previous).abs())) / (2.0)).otherwise(None)
        buying_pressure = (pl.col("close") - low_min).fill_nan(None)
        true_range = (high_max - low_min).fill_nan(None)

        def average(window):
            range_sum = true_range.rolling_sum(window, min_samples=window)
            return pl.when(range_sum != 0).then(
                buying_pressure.rolling_sum(window, min_samples=window) / range_sum
            ).otherwise(None)

        expression = (
            100.0
            * sum(weight * average(window) for weight, window in zip(weights, (short, medium, long)))
            / denominator
        )
        values[column] = _one(frame, column, expression)
    return _result(close, values)


def DEMA(x, window):
    window = _pi(window, "window")
    values = {}
    for column in _cols(x):
        first = _ema(pl.col(column), window)
        values[column] = _one(x, column, 2.0 * first - _ema(first, window))
    return _result(x, values)


def TEMA(x, window):
    window = _pi(window, "window")
    values = {}
    for column in _cols(x):
        first = _ema(pl.col(column), window)
        second = _ema(first, window)
        values[column] = _one(x, column, 3.0 * first - 3.0 * second + _ema(second, window))
    return _result(x, values)


def CMF(high, low, close, volume, window):
    window = _pi(window, "window")
    values = {}
    for column in _cols(high, low, close, volume):
        frame = pl.DataFrame(
            {"high": high[column], "low": low[column], "close": close[column], "volume": volume[column]}
        )
        spread = pl.col("high") - pl.col("low")
        multiplier = pl.when(spread != 0).then(
            ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / spread
        ).otherwise(None)
        denominator = pl.col("volume").rolling_sum(window, min_samples=window)
        numerator = (multiplier * pl.col("volume")).rolling_sum(window, min_samples=window)
        values[column] = _one(
            frame, column, pl.when(denominator != 0).then(numerator / denominator).otherwise(None)
        )
    return _result(close, values)


def MFI(high, low, close, volume, window):
    window = _pi(window, "window")
    values = {}
    for column in _cols(high, low, close, volume):
        frame = pl.DataFrame(
            {"high": high[column], "low": low[column], "close": close[column], "volume": volume[column]}
        )
        typical = ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0)
        raw = typical * pl.col("volume")
        delta = typical.diff().fill_nan(None)
        valid = raw.is_not_null() & delta.is_not_null()
        positive_flow = pl.when(valid).then(
            pl.when(delta > 0).then(raw).otherwise(0.0)
        ).otherwise(None)
        negative_flow = pl.when(valid).then(
            pl.when(delta < 0).then(raw).otherwise(0.0)
        ).otherwise(None)
        positive = positive_flow.rolling_sum(window, min_samples=window)
        negative = negative_flow.rolling_sum(window, min_samples=window)
        ratio = pl.when(negative != 0).then(positive / negative).otherwise(None)
        expression = pl.when(
            negative.is_null() | positive.is_null()
        ).then(None)
        expression = expression.when(
            (negative == 0) & (positive > 0)
        ).then(100.0)
        expression = expression.when(
            (negative == 0) & (positive == 0)
        ).then(50.0)
        expression = expression.when(negative != 0).then(
            100.0 - 100.0 / (1.0 + ratio)
        ).otherwise(expression)
        values[column] = _one(frame, column, expression)
    return _result(close, values)


_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("DMI_plus", ("high", "low", "close", "window"), DMI_plus, "Wilder positive directional indicator."),
    ("DMI_minus", ("high", "low", "close", "window"), DMI_minus, "Wilder negative directional indicator."),
    ("DX", ("high", "low", "close", "window"), DX, "Directional movement index."),
    ("NATR", ("high", "low", "close", "window"), NATR, "Normalized Wilder ATR."),
    ("PPO", ("close", "fast_window", "slow_window"), PPO, "Percentage price oscillator."),
    ("PPO_signal", ("close", "fast_window", "slow_window", "signal_window"), PPO_signal, "PPO signal line."),
    ("PPO_hist", ("close", "fast_window", "slow_window", "signal_window"), PPO_hist, "PPO histogram."),
    ("PVO", ("volume", "fast_window", "slow_window"), PVO, "Percentage volume oscillator."),
    ("PVO_signal", ("volume", "fast_window", "slow_window", "signal_window"), PVO_signal, "PVO signal line."),
    ("PVO_hist", ("volume", "fast_window", "slow_window", "signal_window"), PVO_hist, "PVO histogram."),
    ("CMO", ("close", "window"), CMO, "Chande momentum oscillator."),
    ("VortexPlus", ("high", "low", "close", "window"), VortexPlus, "Positive Vortex indicator."),
    ("VortexMinus", ("high", "low", "close", "window"), VortexMinus, "Negative Vortex indicator."),
    ("TSI", ("close", "long_window", "short_window"), TSI, "True Strength Index."),
    ("TSI_signal", ("close", "long_window", "short_window", "signal_window"), TSI_signal, "TSI signal line."),
    ("UltimateOscillator", ("high", "low", "close", "short_window", "medium_window", "long_window", "short_weight", "medium_weight", "long_weight"), UltimateOscillator, "Ultimate oscillator."),
    ("DEMA", ("x", "window"), DEMA, "Double exponential moving average."),
    ("TEMA", ("x", "window"), TEMA, "Triple exponential moving average."),
    ("CMF", ("high", "low", "close", "volume", "window"), CMF, "Chaikin Money Flow."),
    ("MFI", ("high", "low", "close", "volume", "window"), MFI, "Money Flow Index."),
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
        implementation_source_hash=f"technical.polars_indicators_v2:{kernel}:v1",
        emitter_identity=f"polars_expr:{canonical}",
        kernel_identity=f"technical.polars_indicators_v2:{kernel}",
        parameter_domain_hash=f"{canonical}:declared:v1",
        semantic_contract_hash=f"{canonical}:polars_native_expr:v1",
        notes=_BATCH4_NOTE,
    )


_BATCH4_SPECS: dict[str, PhysicalImplementationSpec] = {
    _c: _batch4_native_spec(_c, _k)
    for _c, _k in (
        ("NATR", "PolarsIndicatorsV2_NATR"),
        ("PVO", "PolarsIndicatorsV2_PVO"),
        ("CMF", "PolarsIndicatorsV2_CMF"),
        ("VortexMinus", "PolarsIndicatorsV2_VortexMinus"),
        ("VortexPlus", "PolarsIndicatorsV2_VortexPlus"),
        ("PVO_signal", "PolarsIndicatorsV2_PVO_signal"),
        ("PVO_hist", "PolarsIndicatorsV2_PVO_hist"),
        ("TSI_signal", "PolarsIndicatorsV2_TSI_signal"),
        ("PPO_signal", "PolarsIndicatorsV2_PPO_signal"),
        ("TSI", "PolarsIndicatorsV2_TSI"),
        ("DMI_plus", "PolarsIndicatorsV2_DMI_plus"),
        ("MFI", "PolarsIndicatorsV2_MFI"),
        ("DX", "PolarsIndicatorsV2_DX"),
        ("CMO", "PolarsIndicatorsV2_CMO"),
        ("DMI_minus", "PolarsIndicatorsV2_DMI_minus"),
        ("PPO", "PolarsIndicatorsV2_PPO"),
        ("PPO_hist", "PolarsIndicatorsV2_PPO_hist"),
    )
}


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    panel_params = ("close",) if name in {"TSI", "TSI_signal"} else ()
    scalar_params = tuple(params[1:]) if panel_params else ()
    metadata = OperatorMetadata(
        name=name,
        category="technical_signal",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe", "causal", "polars", "native"],
        panel_params=panel_params,
        scalar_params=scalar_params,
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsIndicatorsV2_{name}",
        (SeriesOperator,),
        {
            "metadata": metadata,
            "_calculate_series": _calculate_series,
            "__module__": __name__,
            **({"_physical_spec": _BATCH4_SPECS[name]}
               if name in _BATCH4_SPECS else {}),
        },
    )
    register_operator(
        name=name,
        category="technical_signal",
        business_category="technical",
        canonical=name,
        source="polars_indicators_v2",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
