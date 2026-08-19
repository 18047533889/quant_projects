# R2 Resume State (compact)

**Updated:** 2026-08-19
**HEAD:** `bce4c0a9`
**Detail:** `LOOP_ENGINEERING_STATUS.md` (compact) · archive `docs/R2_HISTORY_ARCHIVE.md`
**Queue:** `AUTONOMOUS_MASTER_QUEUE.md`

## Constraints

- **LOCAL ONLY:** no GitHub/`gh`/fetch/push/remotes.
- ≤15 GiB total; ≤2 disjoint low-memory Writers + 1 read-only Auditor; serial tests with BLAS/OpenMP/MKL/Polars threads=1, no xdist.
- Working tree is truth: no `git checkout|restore|stash|clean|reset`; no bulk AST rewrite.
- Never claim PASS for `NOT_RUN`; never load/prompt `docs/R2_HISTORY_ARCHIVE.md`.

## Active priorities

1. Keep Q fail-closed; continue residency / PlanNode / evidence gates with **small scoped diffs**.
2. DataAccess: PIT / catalog identity local evidence only until live COS gates run.
3. Polars selectable math: use evidence YAMLs, not narrative status dumps.
4. FactorAssets / Modeling / QE: focused tests + manifests; broad suites `NOT_RUN`.
5. When compacting docs: archive history, keep pointers (ID → evidence path).

## Truth

Q / FactorAssets / Modeling: `NOT_PRODUCTION_CERTIFIED`
Polars / DuckDB: `PARTIALLY_VERIFIED`
Root CI / live COS / real Redis / broad QE: `NOT_RUN`
