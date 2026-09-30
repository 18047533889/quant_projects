# 2026-09-30 runtime regression checkpoint

These checks used `/home/sunhaiwei/quant_projects/.venv/bin/python` on
`qs-server-c`, directly in the shared main working tree. This is a regression
checkpoint, not a certification that every input is bug-free or fastest.
The published starting commit was `6984c311f`; concurrent new GPU/FE recipe
changes require their own post-change tests before publication.

| Scope | Result | Time |
| --- | --- | --- |
| Public auto routing: core, F5 mixed, F32 Pearson, F8 and mixed backend | 140 passed | 5.26 s |
| Public factor source, GPU tile source, COS adapter, streamed configuration hash | 61 passed | 4.32 s |
| CPU daily IC vectorized equivalence | 20 passed | 0.56 s |
| FactorAssets full test directory | 1535 passed | 50.70 s |

Source tests emitted 69 warnings, including all-missing `nanmean`/CuPy
reductions. FactorAssets emitted 47 warnings, including deprecated legacy
optimizer APIs, UTC timestamp APIs and existing FE physical-spec warnings.
These are not erased or reclassified as a warning-free run.

Auto evidence currently covers explicit measured shapes, metric combinations,
dtypes and VRAM gates. A caller snapshot identifier is not proof of source
provenance. Single-object COS profiling and GPU microbenchmarks do not prove
whole-request superiority; real multi-factor A/B remains required for route changes.
