# GTJA185 batch evaluation

AutoFactorEvaluation contains a versioned pack of all **185 deliverable GTJA191 factors**.
The pack is generated from `gtja191/lib/catalog.py`; every formula and the whole pack carry
SHA-256 fingerprints. CI rejects missing factors, duplicate formulas, stale generated files,
and VWAP formulas that substitute a typical-price proxy for the real `col('vwap')` field.

## Architecture

The public entry point remains `evaluation.gtja185_batch`, while implementation is separated:

```text
evaluation/
├── gtja185_models.py   # immutable configuration and audit records
├── gtja185_metrics.py  # vectorized, leakage-safe statistics
├── gtja185_runner.py   # FactorEngine, DataAccess, resume and reports
└── gtja185_batch.py    # backward-compatible facade
```

All formulas use the repository-root current `factor_engine`. Market data, snapshots, universe
membership and optional factor-lake writes use the repository-root `data_access`.

## Methodology contract

1. Resolve one immutable market-data snapshot.
2. In production, require point-in-time universe membership and tradability. Missing membership
   means ineligible; the evaluator never substitutes the current security master.
3. Compile every formula with the current FactorEngine and PIT audit.
4. Execute in bounded batches with shared-expression reuse. A failed batch is isolated to
   single-factor runs so one error cannot hide the status of the remaining factors.
5. Replace non-finite values, then apply vectorized daily cross-sectional MAD winsorization and
   z-scoring.
6. A signal observed at `t` enters no earlier than `t+1`. Forward labels are purged when their
   exit date crosses a train/validation/test boundary.
7. Choose factor direction from **training RankIC only**.
8. Use **validation only** for routing and ranking. The test sample is a locked final holdout and
   never influences sign, thresholds, routing or order.
9. Correct the validation family with Benjamini-Hochberg FDR before a factor may enter core or
   satellite tiers.
10. Report Pearson IC, RankIC, ICIR, ordinary and Newey-West/HAC t-statistics, positive ratio,
    coverage, equal-weight quantile long-short gross and transaction-cost-adjusted net returns,
    weight turnover, ordinary and HAC-adjusted Sharpe, non-overlapping worst-sleeve drawdown,
    hit rate and yearly stability.
11. Routing uses validation RankIC and **net HAC-adjusted performance**.
12. Optional materialization writes to `factor_lake_staging`; publication remains explicit.

## Outputs and reproducibility

```text
<output>/
├── run_manifest.json
├── summary.json
├── ranking.csv
├── ranking.parquet
└── factor_reports/
    └── gtja191_alpha_XXX.json
```

The run manifest records the pack hash, DataAccess snapshot identity, configuration hash and split
boundaries. Resume is accepted only when formula hash, data snapshot and configuration hash all
match. Valid resumed factors skip compilation and execution, then participate in the current run's
FDR family with newly computed factors.

## Deterministic smoke run

From the repository root:

```bash
export PYTHONPATH="$PWD:$PWD/factor_engine:$PWD/AutoFactorEvaluation-RECONSTRUCT:$PWD/gtja191"

python -m evaluation.gtja185_batch \
  --synthetic \
  --horizons 1,5 \
  --min-assets 6 \
  --n-quantiles 4 \
  --output-dir /tmp/gtja185_smoke
```

## Real production run

```bash
export PYTHONPATH="$PWD:$PWD/factor_engine:$PWD/AutoFactorEvaluation-RECONSTRUCT:$PWD/gtja191"

python -m evaluation.gtja185_batch \
  --run-mode production \
  --dataset ashare_stock_daily \
  --universe-id A_SHARE_ALL_A_EX_ST \
  --require-point-in-time-universe \
  --universe-dataset ashare_universe_daily \
  --start-date 2014-01-01 \
  --end-date 2026-06-25 \
  --horizons 1,5,21 \
  --entry-lag 1 \
  --cost-bps 10 \
  --fdr-alpha 0.10 \
  --batch-size 8 \
  --output-dir /tmp/gtja185_eval
```

The registered universe dataset must contain `TradeDate`, `Symbol`, `is_member` and
`is_tradable`; a static dataset may additionally contain `universe_id`, while a parameterized
dataset may use `universe_id` as a path parameter. Both forms are supported and become part of the
snapshot/config identity.

To persist values, add `--materialize-staging`. Add `--publish` only after reviewing staging and
the evaluation report. Production code never writes directly to the published factor lake.
