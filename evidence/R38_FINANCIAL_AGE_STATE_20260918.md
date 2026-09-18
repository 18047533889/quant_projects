# R38 financial age requires stateful replay

Three previously stateless canonicals now self-declare recursive state and
required_full_history:
fin_days_since_update, fin_staleness, fin_days_since_expectation_revision.

max_days caps the output; it does not bound the history needed to know whether
a real update/revision has ever started the age clock. An identical final
three-row suffix can legitimately produce [1,2,3] or all NaN depending on an
older prefix. No checkpoint implementation is claimed.

Tests cover left-censoring, provider gaps, distinct period-transition behavior,
full-history and unbounded-forward production contracts.
Root combined regression: evidence/r38-financial-age-root.log: **65 passed**,
including independent explicit expected values and existing incremental/history
and left-censor suites. Sampled watchdog wall 53.538s, family RSS 441524224 bytes.

Correctness takes priority over treating these as finite windows; full replay
can cost more until a separately validated checkpoint implementation exists.
No production publication or full-catalog certification.
