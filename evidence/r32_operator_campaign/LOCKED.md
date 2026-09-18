# R32 locked 60-operator campaign

The 60 recipes in `recipes.json` were locked before execution. They are
disjoint from the root-reviewed 454-canonical baseline: 384 earlier passed
canonicals, the 30 R30-extra canonicals, and the 40 R31 canonicals.

The selection prioritizes still-uncovered rolling AR, lag-correlation, robust
signal, fractional-difference, polynomial, Huber, ridge, quantile, expectile,
and multivariate regression families. `ts_kama` and
`ts_adaptive_noise_kalman` are recorded as extended signal coverage; execution
does not promote them to the default authoring surface.

The fixture is deterministic and bounded to 96 rows and two columns. Regression
families use up to four non-collinear deterministic explanatory panels. Runs
use batches of at most 15 canonicals, a 180-second guard, single-threaded BLAS,
and `POLARS_MAX_THREADS=2`.
