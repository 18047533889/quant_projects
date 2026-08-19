# R2 Agent Status (compact)

**Updated:** 2026-08-19 · **HEAD:** `bce4c0a9`

## Active

| Role | Scope |
|---|---|
| (idle) | Prefer 1 Writer + optional read-only Reviewer; avoid loading full LOOP archive |

## Policy

- LOCAL ONLY; no GitHub/`gh`/fetch/push/remotes; no destructive Git.
- ≤15 GiB: max 2 disjoint low-memory Writers + 1 read-only Auditor.
- Serial tests, all BLAS/OpenMP/MKL/Polars threads=1, no xdist; heavy/full suites `NOT_RUN` unless run.
- Review with scoped files + evidence only; on prompt/524 failure shrink input or reduce to 1 Writer—never load archive.

## Next

1. Scoped Q / DA / Polars repairs with evidence YAMLs
2. Keep status files compact; append one-liners only
3. Broad gates remain `NOT_RUN` until run
