# R30 window/statistical operator lock

The 32 canonicals in `recipes.json` were locked before execution. They were
checked against all earlier campaign ledgers and the separate R21 reserved
list. The deterministic campaign uses 96 rows and two columns and exercises
both effective `pandas_numpy` and `polars` registry slots.

The original recipes and evidence are immutable. In particular, the initial
distance-correlation recipes with illegal `min_periods=5` remain preserved;
their corrected `min_periods=10` recipes are in `recipes_retry.json`. The
single corrected AR recipe is in `recipes_ar_retry.json`.

Locked result accounting is 29 initial successes, plus two corrected distance
recipes, plus one repaired AR implementation: 32/32 effective successes.

