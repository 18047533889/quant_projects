# Performance Audit: factor_optimizer

================================================================================
PERFORMANCE OPTIMIZATION AUDIT RESULTS
================================================================================

Total Issues: 6
  CRITICAL: 0
  HIGH:     6
  MEDIUM:   0



================================================================================
🟠 HIGH PRIORITY ISSUES
================================================================================

#1 [nested_loops] /home/shw/quant_projects/factor_optimizer/tests/adapters/test_integration.py:267
   Impact: Nested loop depth 2
   Code: 

#2 [nested_loops] /home/shw/quant_projects/factor_optimizer/factor_optimizer/policy/decisions.py:281
   Impact: Nested loop depth 2
   Code: 

#3 [nested_loops] /home/shw/quant_projects/factor_optimizer/factor_optimizer/search/lineage.py:152
   Impact: Nested loop depth 2
   Code: 

#4 [nested_loops] /home/shw/quant_projects/factor_optimizer/factor_optimizer/search/lineage.py:340
   Impact: Nested loop depth 2
   Code: 

#5 [nested_loops] /home/shw/quant_projects/factor_optimizer/factor_optimizer/search/lineage.py:353
   Impact: Nested loop depth 2
   Code: 

#6 [nested_loops] /home/shw/quant_projects/factor_optimizer/factor_optimizer/search/pareto.py:221
   Impact: Nested loop depth 2
   Code: 