# GTJA185 batch evaluation

AutoFactorEvaluation ships a versioned bundle of all **185 deliverable GTJA191 factors**.
The bundle is generated from `gtja191/lib/catalog.py`; each formula and the complete pack
carry SHA-256 fingerprints, and CI rejects stale or semantically drifted copies. GTJA catalog,
candidate manifests, formula files, materialization YAML and the AutoFactorEvaluation pack are
regenerated from the same canonical source. Formulas that reference VWAP use the actual
`col('vwap')` market field rather than a typical-price proxy.

## Official execution path

`evaluation.gtja185_batch` is the only supported GTJA185 batch path:

1. Resolve one immutable `DataAccess` market-data snapshot.
2. Compile every formula with the repository-root current `FactorEngine` and PIT audit.
3. Execute factors in bounded batches with shared-expression reuse; isolate a failed batch
   to single-factor runs so one bad factor cannot hide the other results.
4. Replace non-finite values, apply daily cross-sectional MAD winsorization and z-scoring.
5. Build forward returns from the configured price field, defaulting to real `vwap`.
   A signal observed at `t` enters no earlier than `t+1`; same-bar execution is forbidden.
6. Split dates chronologically into train, validation and test samples and purge any row
   whose label exit date crosses the next split boundary.
7. Choose signal direction from **training RankIC only**. Validation and test observations
   never participate in sign selection or parameter fitting.
8. Report Pearson IC, RankIC, ICIR, ordinary and Newey-West/HAC t-statistics, positive
   ratio, coverage, equal-weight quantile long-short gross and transaction-cost-adjusted
   net returns, weight turnover, annualized return/volatility, Sharpe, max drawdown, hit
   rate and yearly stability for every configured horizon. Routing uses net performance.
9. Production mode requires a registered point-in-time universe dataset with explicit
   membership and optional tradability flags; missing membership fails closed.
10. Produce deterministic routing recommendations (`tier3a_core`, `tier3b_satellite`,
   `tier2_research`, `rejected`) without silently publishing a factor.
11. Optionally upsert factor values to `factor_lake_staging`; publication remains a separate,
    explicit action.

Every run writes:

```text
<output>/
├── run_manifest.json       # pack hash, data snapshot, config hash, split dates
├── summary.json            # success/failure record for every requested factor
├── ranking.csv
├── ranking.parquet
└── factor_reports/
    └── gtja191_alpha_XXX.json
```

Resume is allowed only when the factor formula hash, DataAccess snapshot ID and evaluation
configuration hash all match. A changed formula, dataset snapshot or metric configuration
forces recomputation.

The committed GTJA materialization YAML files use repository-relative defaults:

```text
data/factors/plan_cache/gtja191
data/factors/lake/gtja191
```

They therefore remain portable across CI, research servers and developer workstations; runtime
orchestration may override the output root without regenerating the formulas.

## Full real-data run

Run from the repository root so the shared `factor_engine` and `data_access` modules are
resolved consistently:

```bash
export PYTHONPATH="$PWD:$PWD/factor_engine:$PWD/AutoFactorEvaluation-RECONSTRUCT:$PWD/gtja191"

python -m pipeline --gtja185 \
  --gtja-start-date 2014-01-01 \
  --gtja-end-date 2026-06-25 \
  --gtja-horizons 1,5,21 \
  --gtja-batch-size 8 \
  --gtja-output /tmp/gtja185_eval
```

For a deterministic data-free smoke run, append `--gtja-synthetic`. To persist calculated
values, add `--gtja-materialize-staging`. Published factor values are never written directly;
add `--gtja-publish` only after the staging output and evaluation report have been reviewed.
