# R36 minute-session returns and recipe parameters

## Repairs
- Session return helper explicitly disables price forward-fill. A missing
  current or lagged price remains missing, rather than inventing zero returns.
- Session keys preserve timezone-aware DatetimeIndex dtype; no conversion to
  timezone-naive datetime64 is forced.
- Legacy research recipes recipe_micro_vpin and recipe_micro_trade_imbalance
  consume keyword window/min_periods and panel arguments consistently with
  positional arguments. No production eligibility or canonical promotion changed.
  recipe_micro_vpin remains a legacy absolute-return proxy, not true VPIN.

## Evidence
- r36-micro-session-before.log: 9 failures. Seven are helper failures;
  two were test registry-mode mistakes (production rather than any), not
  evidence that kernels failed.
- r36-micro-keywords-before.log: after correcting the test mode, 2 real
  keyword-versus-positional mismatches reproduced.
- r36-micro-session-fixed.log: 13 passed.
- r36-micro-session-final.log: 13 passed, including all-keyword calls and
  duplicate parameter rejection; legacy semantic tests included.
- Includes explicit ratio/rolling oracle, 1/2-bar returns, timezone-naive,
  Hong Kong/New York timezone-aware indices, gaps and session boundaries.
- Final sampled watchdog peak family RSS: 460697600 bytes; wall 56.992s.
  This is not a hard memory cap.

Not a claim of complete operator coverage, all backends, or full-catalog
production evaluation. The 110000-factor CSV was not edited in this repair.
