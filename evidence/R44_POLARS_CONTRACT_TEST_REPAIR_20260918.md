# Polars window contracts and labeled parity tests

Updated stale window-statistics tests to the actual public contract:
- z-score default non-finite propagation and explicit ignore policy both have independent expectations.
- Public z-score exposes 8 declared parameters; undeclared keywords still reject.
- Trimmed mean uses trim_ratio, not the nonexistent trim_pct. Wrong-key rejection is tested through calculate (private kernels deliberately bypass binding).
- Parity compares unique labeled series after sorting, not storage row order. Group fixture now has real two-member groups and independent sample-zscore expectations; linear decay has explicit weighted-mean values.

Root evidence/r44-window-stats-root.log: window statistics + nonfinite reference + tier3 regressions passed (see log for exact test count). This does not reactivate retired rollout tiers or convert their 25 skips into certifications.

Separately evidence/r44-auto-resource-root.log reruns auto-DAG pipeline/admission, research auto run_many, memory contracts and observed/post-read budgets; no code change was needed in those components during this batch. These are targeted tests, not an 110k-factor production performance claim.
