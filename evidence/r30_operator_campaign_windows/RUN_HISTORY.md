# R30 run and triage history

All commands were run from `/home/sunhaiwei/quant_projects` with the project
`.venv`, a 180-second process guard, single-threaded BLAS, and
`POLARS_MAX_THREADS=2`.

## Initial locked run

Script: `evidence/r23_operator_campaign/campaign.py`

- Script SHA-256: `397cd52916fd522376a9d4e6a60baa2072ccbf566800274dd62ca7d8df8e5203`
- Recipe SHA-256: `b35c8bdf8c19c2a45217e1a3c5355adcb69ca72706f9d7c440361c0047a80d36`
- Logs: `run_0_initial.log`, `run_16_initial.log`
- Ledger: `ledger.jsonl`

Commands (with the common environment prefix shown above):

```text
timeout 180s .venv/bin/python evidence/r23_operator_campaign/campaign.py --recipes evidence/r30_operator_campaign_windows/recipes.json --ledger evidence/r30_operator_campaign_windows/ledger.jsonl --selection recipes --offset 0 --limit 16 --rows 96
timeout 180s .venv/bin/python evidence/r23_operator_campaign/campaign.py --recipes evidence/r30_operator_campaign_windows/recipes.json --ledger evidence/r30_operator_campaign_windows/ledger.jsonl --selection recipes --offset 16 --limit 16 --rows 96
```

Twenty-nine canonicals passed finite output, all-input future-prefix invariance,
and exact-backend-fingerprint-bound parity. `ts_distance_corr` and
`ts_distance_cov` did not execute because the locked recipe's `min_periods=5`
violated their formal minimum of 10. `ts_ar_coefficient` executed but exposed a
real Polars implementation defect: 172 overlapping finite cells all differed.

## Distance recipe retry

- Recipe: `recipes_retry.json`
- Recipe SHA-256: `f9c79e336bcdc5888f8e726d53710ff47911abcbb5a384ab2b1907a597e988f7`
- Log: `run_retry.log`
- Ledger: `ledger_retry.jsonl`

```text
timeout 180s .venv/bin/python evidence/r23_operator_campaign/campaign.py --recipes evidence/r30_operator_campaign_windows/recipes_retry.json --ledger evidence/r30_operator_campaign_windows/ledger_retry.jsonl --selection recipes --offset 0 --limit 2 --rows 96
```

Both distance operators passed both backends with 174 finite cells per backend,
future-prefix invariance, and Pandas/Polars parity.

## AR repair and retry

The canonical is a single economic lag with an intercept, pairwise-finite
support, and `max(3, min_periods)` pairs. The prior Polars class incorrectly
fit an intercept-free AR(p) and ignored support/warmup policy. The correctness
repair is covered by `factor_engine/tests/runtime/test_r30_ar_coefficient_parity.py`
(lag 1/2, expanding/full, NaNs, independent OLS oracle, prefix stability).

- Recipe: `recipes_ar_retry.json`
- Recipe SHA-256: `2e13a73486b755e451d359ce134aa3c7f236a837263a02bdf8cf3bf1b831a689`
- Test log: `ar_test.log` (4 passed)
- Campaign log: `run_ar_retry.log`
- Ledger: `ledger_ar_retry.jsonl`

```text
timeout 180s .venv/bin/pytest -q factor_engine/tests/runtime/test_r30_ar_coefficient_parity.py
timeout 180s .venv/bin/python evidence/r23_operator_campaign/campaign.py --recipes evidence/r30_operator_campaign_windows/recipes_ar_retry.json --ledger evidence/r30_operator_campaign_windows/ledger_ar_retry.jsonl --selection recipes --offset 0 --limit 1 --rows 96
```

The retry passed with 172 finite cells per backend, future-prefix invariance,
and parity. Historical ledgers were never overwritten.

## Independent major-formula oracle

- Script: `oracle.py`
- SHA-256: `80ea1b70cf06f02cfce2af7c8b7deec9428c8743f2141e98d30e14df219decf1`
- Result: `oracle_results.json`

```text
timeout 180s .venv/bin/python evidence/r30_operator_campaign_windows/oracle.py
```

Fourteen formulas across both backends passed 28/28 checks; maximum absolute
error was `3.3306690738754696e-16`.

## Native Polars performance prototype (not shipped)

- Script: `ar_polars_prototype.py`
- SHA-256: `9de5ae6200f2b7b1bff073ac8aa21356d4163ccf990419b41eb4a2bae0b6fca9`
- Result: `ar_polars_prototype_results.json`

```text
timeout 180s .venv/bin/python evidence/r30_operator_campaign_windows/ar_polars_prototype.py
```

Native `rolling_cov/rolling_var` matched ordinary and pairwise-NaN cases and
was about 17.28x faster in a bounded 512x8 benchmark. It failed all four
`+1e12` offset-stability cases, with maximum absolute error about `9.54e10`.
It was therefore rejected and was **not** applied to the AR source. The correct
NumPy implementation and `ledger_ar_retry.jsonl` remain authoritative.
