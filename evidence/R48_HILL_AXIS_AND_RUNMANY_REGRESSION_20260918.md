# R48 Hill axis/domain repair and run_many regression

- Active batch5 Hill implementation inherited the authoritative parameter contract: legal window>=2, explicit side/min_tail_count/defaults, no divergent window>=20 restriction.
- Polars metadata/time columns are preserved instead of entering the Hill kernel; multi-stock long frames under standard identity-column names fail explicitly.
- Actual shared CPU estimator retained, not relabeled native or GPU.
- New reproducer evidence/r48-hill-axis-before.log: 5 failures.
- Repaired evidence/r48-hill-axis-after.log: 16 passed including independent Hill formula, four time-coordinate aliases, scale/domain regressions, fresh-process binding.
- evidence/r48-runmany-da-stream-root.log: 69 passed covering DataAccess snapshot cache across waves, manifest invalidation, stream controls, CSE, fusion control.
- Sampled RSS test guard only, not a hard memory cap. No 110k-factor production run or CSV change in this patch.
