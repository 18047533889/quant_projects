# Factor catalog continuation — 2026-09-16

Worktree: server-c `/home/sunhaiwei/quant_projects`. No branch, commit, push, deployment, or production-factor publication.

## Verified root changes

- `factor_engine/api/intraday_daily.py`: reject fractional, boolean, NaN and infinite integer controls; retain existing integral integer/float/string inputs; normalize `minutes` as well as bar/history controls. RED 11 failed / 2 passed; GREEN 24 passed / 2 warnings. Logs: `intraday-integer-controls-{red,green}.log`.
- `factor_engine/tools/catalog_migration.py`: preserve explicit official raw-price leaves in known exchange-limit price argument positions, including legacy uppercase spelling. Keep unrelated prices adjusted and preserve field control flags. Full price→recipe→reimport regression RED 8 failed / 1 passed; GREEN 44 passed / 2 warnings. Logs: `catalog-price-pipeline-{red,green}.log`. Root frozen hash before later authorized intraday extension: `20fb6af3cf2be7947a47bc05fb98607881903f3e84fc1fdd410b7bde6bfb4170`.
- Root temporary patch/test transfer files removed after application. Formal tests retained.

## Additional executed checks

- DataAccess PIT + authoritative missingness: 96 passed. `dataaccess-pit-missingness-r7.log`.
- DataAccess parallel admission + frozen physical scan scope + unknown-cost handling: 23 passed. `dataaccess-resource-scope-r7.log`.
- Quant Evaluator actual CuPy GPU parity: 11 passed, not skipped. `quant-evaluator-gpu-parity-r7.log`; watchdog return 0, peak process-family RSS 701751296 bytes. Hardware: NVIDIA L20. This certifies these small GPU tests only, not all factor operators or GPU memory exhaustion safety.
- Fresh read of closed `open-limit-failed17-fixed-r18.summary.json`: 17/17 executed, 2025-01-01..2026-04-30, eight symbols. Its deadline kind is legacy-cooperative, not a hard external timeout.
- Auto/pandas real first-two parity: identical result hashes per factor, each 2088 finite values. Auto batch 10.280938s; pandas batch 9.662257s. These are single-run timings, not a statistically controlled speed comparison.

## Incomplete work / next actions

- Latest completed daily sweep ends at source offset 12503 (source_row = offset + 2). Resume there after shared-runtime freeze is released.
- R6 checkpoint has 5211 EXECUTED, 67 EXECUTED_ALL_NONFINITE, 17 EXECUTION_FAILED, 14 PREPARE_FAILED, 81 execution-COMPILE_FAILED, 108742 NOT_RUN. Historical formula-matched evidence is not current-code certification. Most factors remain unexecuted.
- 1000-factor synthetic auto+sink probe passed. 10000 failed before execution at default monolithic DAG-width gate (adaptive limit 1754); 100000 was not attempted. Tail agent is repairing automatic bounded waves, not raising the width limit. Default `result_policy=return` compatibility and cross-wave reuse limits must be stated honestly.
- Row6170 small-scale liquidity regression: real pandas rerun is executed; shared pandas/Polars/SQL numeric fix and primitive recertification owned by intraday agent. Do not claim recertification until its final fresh check/hash bracket completes.
- User authorized semantic redesign of 99 old intraday-limit formulas using current formal definitions with original formula and explicit change ledger retained. Alpha owns an opt-in redesign path. This is NOT an equivalent rename. Whole-factor unresolved fields (e.g. pledge data) remain unresolved even if a rewritten subexpression passes.
- IndexReturn benchmark choice remains unanswered; do not invent it.
- No claim of all-operator availability, all-backend parity, globally fastest performance, or OOM-proof execution.

The separate external summary workbook has already been delivered and is not the DSL review table. Original source CSV remains untouched.
