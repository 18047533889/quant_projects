# Performance Audit: quant_evaluator

================================================================================
PERFORMANCE OPTIMIZATION AUDIT RESULTS
================================================================================

Total Issues: 216
  CRITICAL: 0
  HIGH:     215
  MEDIUM:   1



================================================================================
🟠 HIGH PRIORITY ISSUES
================================================================================

#1 [nested_loops] /home/shw/quant_projects/quant_evaluator/benchmark_numba_speedup.py:67
   Impact: Nested loop depth 2
   Code: 

#2 [nested_loops] /home/shw/quant_projects/quant_evaluator/benchmark_numba_speedup.py:248
   Impact: Nested loop depth 2
   Code: 

#3 [nested_loops] /home/shw/quant_projects/quant_evaluator/benchmark_numba_speedup.py:265
   Impact: Nested loop depth 3
   Code: 

#4 [nested_loops] /home/shw/quant_projects/quant_evaluator/tests/test_batch_plan.py:222
   Impact: Nested loop depth 2
   Code: 

#5 [nested_loops] /home/shw/quant_projects/quant_evaluator/tests/test_batch_plan.py:223
   Impact: Nested loop depth 3
   Code: 

#6 [nested_loops] /home/shw/quant_projects/quant_evaluator/tests/test_batch_plan.py:224
   Impact: Nested loop depth 4
   Code: 

#7 [nested_loops] /home/shw/quant_projects/quant_evaluator/tests/test_numba_parity.py:269
   Impact: Nested loop depth 2
   Code: 

#8 [nested_loops] /home/shw/quant_projects/quant_evaluator/tests/test_numba_parity.py:291
   Impact: Nested loop depth 2
   Code: 

#9 [nested_loops] /home/shw/quant_projects/quant_evaluator/tests/test_numba_parity.py:314
   Impact: Nested loop depth 2
   Code: 

#10 [nested_loops] /home/shw/quant_projects/quant_evaluator/tests/test_cupy_backend.py:293
   Impact: Nested loop depth 2
   Code: 

================================================================================
🟡 MEDIUM PRIORITY ISSUES
================================================================================

Found 1 medium priority issues
  - missing_numba: 1 occurrences