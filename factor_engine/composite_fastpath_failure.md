.......F......F.........                                                 [100%]
=================================== FAILURES ===================================
_________________ test_native_polars_parity[ATR_WILDER-args0] __________________
factor_engine/tests/operators/test_composite_fastpath.py:127: in test_native_polars_parity
    np.testing.assert_allclose(
E   AssertionError: 
E   Not equal to tolerance rtol=1e-07, atol=1e-07
E   
E   nan location mismatch:
E    ACTUAL: array([[nan, nan],
E          [nan, nan],
E          [nan, nan],...
E    DESIRED: array([[nan, nan],
E          [nan, nan],
E          [nan, nan],...
_________________ test_native_polars_parity[volatility-args7] __________________
factor_engine/tests/operators/test_composite_fastpath.py:127: in test_native_polars_parity
    np.testing.assert_allclose(
E   AssertionError: 
E   Not equal to tolerance rtol=1e-07, atol=1e-07
E   
E   nan location mismatch:
E    ACTUAL: array([[      nan,       nan],
E          [      nan,       nan],
E          [      nan,       nan],...
E    DESIRED: array([[      nan,       nan],
E          [ 6.550202,  2.904134],
E          [ 9.141563,  3.948085],...
=========================== short test summary info ============================
FAILED factor_engine/tests/operators/test_composite_fastpath.py::test_native_polars_parity[ATR_WILDER-args0] - AssertionError: 
Not equal to tolerance rtol=1e-07, atol=1e-07

nan location mismatch:
 ACTUAL: array([[nan, nan],
       [nan, nan],
       [nan, nan],...
 DESIRED: array([[nan, nan],
       [nan, nan],
       [nan, nan],...
FAILED factor_engine/tests/operators/test_composite_fastpath.py::test_native_polars_parity[volatility-args7] - AssertionError: 
Not equal to tolerance rtol=1e-07, atol=1e-07

nan location mismatch:
 ACTUAL: array([[      nan,       nan],
       [      nan,       nan],
       [      nan,       nan],...
 DESIRED: array([[      nan,       nan],
       [ 6.550202,  2.904134],
       [ 9.141563,  3.948085],...
2 failed, 22 passed in 0.72s
