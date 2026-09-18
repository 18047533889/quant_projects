# R36 incremental history repair

## Actual defects
- A planner allocation floor (two rows) incorrectly classified pointwise
  fin_ratio as finite-window. Incremental classification now uses semantic own
  dependency where no authoritative history factory is registered.
- fin_surprise_zscore lost its daily prior-window mapping in the central
  execution contract. Its shift(1).rolling(window_days) baseline needs exactly
  window_days prior rows, not two or window_days-1. Central backward history
  and forward impact now both reflect this.
- Unknown/full/unbounded and malformed history remain full replay, including
  production fiscal-calendar uncertainty. Explicit history factories remain
  authoritative; checkpointed recursive behavior is preserved.

## Evidence
Root combined run: evidence/r36-history-root-final.log, 35 passed.
Files: test_r44_incremental_contract.py, test_analyzer_semantic_warmup_v2.py,
test_r10_execution_contract.py, test_incremental_history_contract_v2.py.
Includes independent NumPy prior-window mean/ddof=1 standard-deviation
oracle, windows 10 and 252, default and keyword calls, exact overlap versus
one-row-short overlap, production stateless pointwise behavior, and
unknown/fractional/malformed history fail-closed cases.

Sampled watchdog wall 55.957s; process-family peak RSS 466477056 bytes.
Not a hard memory cap or a full-catalog evaluation.
