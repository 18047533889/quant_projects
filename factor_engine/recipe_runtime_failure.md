...................F....................                                 [100%]
=================================== FAILURES ===================================
____________ test_recipe_compiler_rejects_unsafe_syntax_and_cycles _____________
factor_engine/tests/operators/test_recipe_runtime.py:54: in test_recipe_compiler_rejects_unsafe_syntax_and_cycles
    with pytest.raises(RecipeExpansionError, match="cyclic"):
E   AssertionError: Regex pattern did not match.
E     Expected regex: 'cyclic'
E     Actual message: 'dunder names are forbidden'
=========================== short test summary info ============================
FAILED factor_engine/tests/operators/test_recipe_runtime.py::test_recipe_compiler_rejects_unsafe_syntax_and_cycles - AssertionError: Regex pattern did not match.
  Expected regex: 'cyclic'
  Actual message: 'dunder names are forbidden'
1 failed, 39 passed in 0.83s
