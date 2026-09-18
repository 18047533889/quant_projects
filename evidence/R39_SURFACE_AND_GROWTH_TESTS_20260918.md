# R39 truthful backend and report-growth regression tests

Three stale test assumptions corrected without promoting any operator:
- group_normalize Polars is truthfully a pandas delegate, not native.
- Deprecated cs_robust_resid resolves to cs_trimmed_ols_resid, an extended
  canonical; it is not automatically a daily production primitive.
- report_change_breadth measures centered standardized relative growth.
  Rising linear levels have declining relative growth and can legitimately
  produce negative breadth. Replaced the old 'all rising implies positive'
  assertion with an independent median/MAD oracle for linear versus
  accelerating growth; coherence is checked separately.

Root combined regression: evidence/r39-daily-surface-growth-root.log:
**52 passed**, 3 warnings. No source kernel or governance change in this commit.
