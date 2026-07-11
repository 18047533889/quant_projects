# GTJA185 batch evaluation

AutoFactorEvaluation ships a versioned bundle of all 185 deliverable GTJA191 factors.
The bundle is generated from `gtja191/lib/catalog.py`, and CI rejects stale copies.

The official path is `evaluation.gtja185_batch`: one DataAccess snapshot, the current
FactorEngine, PIT compilation, cross-sectional MAD winsorization/z-scoring, train-only
direction selection, validation/test IC, RankIC, ICIR, quantile long-short, turnover,
annualized performance, drawdown, yearly stability, deterministic routing, resumable
reports, and optional `factor_lake_staging` writes.

```bash
python -m pipeline --gtja185   --gtja-start-date 2014-01-01   --gtja-end-date 2026-06-25   --gtja-output /tmp/gtja185_eval
```

For a data-free deterministic smoke run, append `--gtja-synthetic`. Published factor
values are never written directly: use `--gtja-materialize-staging`; add
`--gtja-publish` only after review.
