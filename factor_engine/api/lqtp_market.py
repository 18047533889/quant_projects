# -*- coding: utf-8 -*-
"""LQTP market-risk overloads backed by the registered benchmark Return field."""
from __future__ import annotations

from typing import Any, Callable

DEFAULT_INDEX = "000985.SH"


def _factory(canonical: str) -> Callable[..., Any]:
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    return make_cleaned_call_factory(canonical)


def benchmark_return(index: str = DEFAULT_INDEX):
    from factor_engine.api.source_ref import source_col
    return source_col(
        "BenchmarkIndexDailyBar",
        "Return",
        index=str(index),
        dialect="lqtp",
        dialect_version="2026-07-19",
    )


def market_ret(*args: Any, **kwargs: Any):
    if args or kwargs:
        raise ValueError("market_ret takes no arguments")
    return benchmark_return()


def rolling_beta_to_market(ret: Any, index_or_window: Any, window: Any | None = None):
    if window is None:
        benchmark = benchmark_return()
        window = index_or_window
    else:
        benchmark = benchmark_return(index_or_window) if isinstance(index_or_window, str) else index_or_window
    return _factory("rolling_beta_to_market")(ret, benchmark, window)


def tail_beta(*args: Any):
    if len(args) == 3:
        ret, window, q = args
        benchmark = benchmark_return()
    elif len(args) == 4:
        ret, index_or_benchmark, window, q = args
        benchmark = benchmark_return(index_or_benchmark) if isinstance(index_or_benchmark, str) else index_or_benchmark
    else:
        raise ValueError("tail_beta accepts (ret, window, q) or (ret, index, window, q)")
    return _factory("tail_beta")(ret, benchmark, window, q)


def residual_momentum_capm(*args: Any):
    if len(args) == 2:
        ret, window = args
        benchmark = benchmark_return()
    elif len(args) == 3:
        ret, index_or_benchmark, window = args
        benchmark = benchmark_return(index_or_benchmark) if isinstance(index_or_benchmark, str) else index_or_benchmark
    else:
        raise ValueError("residual_momentum_capm accepts (ret, window) or (ret, index, window)")
    return _factory("residual_momentum_capm")(ret, benchmark, window)


def _three_arg(canonical: str):
    def dispatch(ret: Any, index_or_benchmark: Any, window: Any):
        benchmark = benchmark_return(index_or_benchmark) if isinstance(index_or_benchmark, str) else index_or_benchmark
        return _factory(canonical)(ret, benchmark, window)
    return dispatch


def downside_beta(ret: Any, index_or_benchmark: Any, window: Any):
    benchmark = benchmark_return(index_or_benchmark) if isinstance(index_or_benchmark, str) else index_or_benchmark
    cond = _factory("lt")(benchmark, 0.0)
    zero = _factory("subtract")(ret, ret)
    null_ret = _factory("safe_div_null")(zero, zero)
    zero_b = _factory("subtract")(benchmark, benchmark)
    null_benchmark = _factory("safe_div_null")(zero_b, zero_b)
    return _factory("ts_beta")(
        _factory("where")(cond, ret, null_ret),
        _factory("where")(cond, benchmark, null_benchmark),
        window,
    )


def augment_market(allow: dict[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
    out = dict(allow)
    out.update({
        "market_ret": market_ret,
        "rolling_beta_to_market": rolling_beta_to_market,
        "fp_beta": rolling_beta_to_market,
        "tail_beta": tail_beta,
        "residual_momentum_capm": residual_momentum_capm,
        "coskewness_to_market": _three_arg("coskewness_to_market"),
        "idio_vol": _three_arg("idio_vol"),
        "idio_skew": _three_arg("idio_skew"),
        "downside_beta": downside_beta,
    })
    return out
