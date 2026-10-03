# RAW series cache: bounded runtime probe (2026-10-04)

Status: performance hotspot verified; acceleration not implemented or measured.

Formal tree: server-c `/home/sunhaiwei/quant_projects`.
Source: `factor_optimizer/research_fitness.py`, `paired_series` RAW cache key.
The existing `tests/test_pair_ic_cache.py` fixture supplied the inputs.
Three calls used the same RAW/labels and candidates `-x`, `-2*x`, `-3*x`,
with 60 selected dates and 40 assets. An invocation-local SHA256 wrapper
counted update bytes; a RawSeriesCache subclass counted successful hits.

| Call | Cumulative digest bytes | Cumulative cache hits |
| --- | ---: | ---: |
| 1 | 38442 | 0 |
| 2 | 76884 | 1 |
| 3 | 115326 | 2 |
All three RAW outputs were exactly equal; process exit code was zero.
