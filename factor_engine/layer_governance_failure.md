...............F.........                                                [100%]
=================================== FAILURES ===================================
_________ test_native_polars_parity_for_retained_composites[ADX-args2] _________
factor_engine/tests/operators/test_composite_fastpath.py:67: in test_native_polars_parity_for_retained_composites
    np.testing.assert_allclose(
E   AssertionError: 
E   Not equal to tolerance rtol=1e-07, atol=1e-07
E   
E   nan location mismatch:
E    ACTUAL: array([[ nan,  nan],
E          [ nan,  nan],
E          [ nan,  nan],...
E    DESIRED: array([[ nan,  nan],
E          [ nan,  nan],
E          [ nan,  nan],...
=========================== short test summary info ============================
FAILED factor_engine/tests/operators/test_composite_fastpath.py::test_native_polars_parity_for_retained_composites[ADX-args2] - AssertionError: 
Not equal to tolerance rtol=1e-07, atol=1e-07

nan location mismatch:
 ACTUAL: array([[ nan,  nan],
       [ nan,  nan],
       [ nan,  nan],...
 DESIRED: array([[ nan,  nan],
       [ nan,  nan],
       [ nan,  nan],...
1 failed, 24 passed in 0.71s
