# Performance Audit: factor_preprocess

================================================================================
PERFORMANCE OPTIMIZATION AUDIT RESULTS
================================================================================

Total Issues: 99
  CRITICAL: 0
  HIGH:     86
  MEDIUM:   13



================================================================================
🟠 HIGH PRIORITY ISSUES
================================================================================

#1 [nested_loops] /home/shw/quant_projects/factor_preprocess/tests/neutralization/test_ols.py:26
   Impact: Nested loop depth 2
   Code: 

#2 [nested_loops] /home/shw/quant_projects/factor_preprocess/tests/neutralization/test_ols.py:59
   Impact: Nested loop depth 2
   Code: 

#3 [nested_loops] /home/shw/quant_projects/factor_preprocess/tests/neutralization/test_ols.py:89
   Impact: Nested loop depth 2
   Code: 

#4 [nested_loops] /home/shw/quant_projects/factor_preprocess/tests/neutralization/test_ols.py:211
   Impact: Nested loop depth 2
   Code: 

#5 [nested_loops] /home/shw/quant_projects/factor_preprocess/tests/neutralization/test_regularized.py:33
   Impact: Nested loop depth 2
   Code: 

#6 [nested_loops] /home/shw/quant_projects/factor_preprocess/tests/neutralization/test_regularized.py:112
   Impact: Nested loop depth 2
   Code: 

#7 [nested_loops] /home/shw/quant_projects/factor_preprocess/tests/neutralization/test_diagnostics.py:81
   Impact: Nested loop depth 2
   Code: 

#8 [nested_loops] /home/shw/quant_projects/factor_preprocess/tests/neutralization/test_diagnostics.py:471
   Impact: Nested loop depth 2
   Code: 

#9 [nested_loops] /home/shw/quant_projects/factor_preprocess/tests/neutralization/advanced/test_pca_neutralization.py:25
   Impact: Nested loop depth 2
   Code: 

#10 [nested_loops] /home/shw/quant_projects/factor_preprocess/tests/neutralization/advanced/test_pca_neutralization.py:103
   Impact: Nested loop depth 2
   Code: 

================================================================================
🟡 MEDIUM PRIORITY ISSUES
================================================================================

Found 13 medium priority issues
  - missing_numba: 8 occurrences
  - excessive_copies: 5 occurrences