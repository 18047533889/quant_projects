# R31 run history

The locked 40-operator campaign used 96 deterministic daily rows and two
columns. All commands ran from `/home/sunhaiwei/quant_projects` with the project
`.venv`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `POLARS_MAX_THREADS=2`, and a 180-second process guard.

- Campaign SHA-256: `0d1921dbb78accdc674c6a76d72a73f2f7b57b9b75e33a42aeab52774f090f06`
- Recipes SHA-256: `27064bdd0079915c48ec76a582db0d7df388ca1b3a66e342bbd92dab2bc1aeb6`
- Ledger: `ledger.jsonl`
- Logs: `run_0.log`, `run_16.log`, `run_32.log`

```text
timeout 180s .venv/bin/python evidence/r31_operator_campaign/campaign.py --recipes evidence/r31_operator_campaign/recipes.json --ledger evidence/r31_operator_campaign/ledger.jsonl --selection recipes --offset 0 --limit 16 --rows 96
timeout 180s .venv/bin/python evidence/r31_operator_campaign/campaign.py --recipes evidence/r31_operator_campaign/recipes.json --ledger evidence/r31_operator_campaign/ledger.jsonl --selection recipes --offset 16 --limit 16 --rows 96
timeout 180s .venv/bin/python evidence/r31_operator_campaign/campaign.py --recipes evidence/r31_operator_campaign/recipes.json --ledger evidence/r31_operator_campaign/ledger.jsonl --selection recipes --offset 32 --limit 8 --rows 96
```

All 80 backend executions produced finite values (154 to 192 finite cells),
all 80 preserved the all-input future-prefix invariant, and all 40 paired rows
passed Pandas/Polars equality bound to the exact backend fingerprints. There
were no failed execution or parity rows.

## Independent formula oracle

- Oracle SHA-256: `836043dbf199964f19d9ad3c19f5e2da53e04d4cb622a6da06a0fec0ec72d4af`
- Script: `oracle.py`
- Log: `oracle.log`
- Result: `oracle_results.json`

```text
timeout 180s .venv/bin/python evidence/r31_operator_campaign/oracle.py
```

Thirteen representative formulas passed on both backends (26/26 checks), with
maximum absolute error `6.494804694057166e-15`. Coverage includes rolling MAD,
quantile/range, upside deviation, zero ratio, positive streak, transition
count, nth value, one-step ratio, causal median-3, and OLS slope/intercept/R2.

No production operator source was changed during R31.
