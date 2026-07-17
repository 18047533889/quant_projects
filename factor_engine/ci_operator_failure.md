............F..............                                              [100%]
=================================== FAILURES ===================================
____________ test_native_polars_matches_pandas[ts_topk_mean-args3] _____________
factor_engine/tests/operators/test_operator_overhaul.py:146: in test_native_polars_matches_pandas
    polars_result = OperatorRegistry.get(operator, backend="polars").calculate(*polars_args)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
factor_engine/cleaned_operators/overhaul/base.py:143: in calculate
    return self._fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
factor_engine/cleaned_operators/overhaul/daily.py:344: in <lambda>
    lambda x, window, k, min_periods=None, **kw: pl_topbottom(x, window, k, min_periods, top, stat),
                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
factor_engine/cleaned_operators/overhaul/daily.py:302: in pl_topbottom
    return pl_unary_rolling_map(x, w, 1, fn)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
factor_engine/cleaned_operators/overhaul/base.py:88: in pl_unary_rolling_map
    replacements[col] = temp.select(
/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/polars/dataframe/frame.py:10484: in select
    .collect(optimizations=QueryOptFlags._eager())
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/polars/_utils/deprecation.py:97: in wrapper
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/polars/lazyframe/opt_flags.py:344: in wrapper
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/polars/lazyframe/frame.py:2630: in collect
    return wrap_df(ldf.collect(engine, callback))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: No loop matching the specified signature and casting was found for ufunc isfinite
E   
E   This error occurred in the following expression:
E   	col("v").rolling_map()
=========================== short test summary info ============================
FAILED factor_engine/tests/operators/test_operator_overhaul.py::test_native_polars_matches_pandas[ts_topk_mean-args3] - TypeError: No loop matching the specified signature and casting was found for ufunc isfinite

This error occurred in the following expression:
	col("v").rolling_map()
1 failed, 26 passed in 0.85s
