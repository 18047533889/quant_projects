# R31 locked 40-operator campaign

The 40 recipes in `recipes.json` were locked before execution. They are disjoint
from the root-reviewed 384 previously passed canonicals (R24 canonicals plus
R25/R26/R27/R28 new sets and R30 windows) and from the separate 30-canonical
R21 reserved list. Incidental names in incomplete or unrelated ledgers are not
used as the passed-count denominator.

The campaign covers both ordinary daily authoring operators and explicitly
extended filters/robust statistics. In particular `ts_robust_ema`,
`ts_hampel_filter_causal`, `ts_median3_causal`, and `ts_qn_scale` are recorded
as extended/filter coverage; dual-backend execution does not imply promotion to
the default authoring surface.

Execution is bounded to 96 rows, two columns, batches of at most 16 operators,
a 180-second process guard, single-threaded BLAS, and `POLARS_MAX_THREADS=2`.
