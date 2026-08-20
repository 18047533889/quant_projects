# R2 Agent Status (compact)

**Updated:** 2026-08-20 · [`orchestration.md`](orchestration.md)  
**CONTINUE:** yes — delete this word and `touch loop/HALT` to allow Claude to stop.

## Active roles (fill each cycle)

| Role | Agent | Scope this cycle |
|---|---|---|
| Coordinator | main | spawn only |
| Finder | finder.md | (idle until spawned) |
| Dispatcher | dispatcher.md | (idle) |
| Writer-A | writer-platform or writer-fe | — |
| Writer-B | optional disjoint | — |
| Tester | tester.md | — |
| Reviewer | reviewer.md | — |

## Policy
- Continuous cycles until user stops. LOCAL ONLY. ≤2 Writers, disjoint files.
- Serial pytest, BLAS/OpenMP/MKL/Polars=1. Unrun=`NOT_RUN`.
- Review: diff + evidence only.
