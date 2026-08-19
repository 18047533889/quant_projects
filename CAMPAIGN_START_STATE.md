# Campaign start state (stale pointer)

**Status:** COMPLETED / SUPERSEDED (2026-08-14 6h campaign ended).  
**Do not** treat this file as live orchestration or reload its old role list into prompts.

## Live SoT
- `LOOP_ENGINEERING_STATUS.md`
- `RESUME_STATE.md`
- `AGENT_STATUS.md`
- `AUTONOMOUS_MASTER_QUEUE.md`
- `CLAUDE.md` (this repo) + `/home/shw/CLAUDE.md`

## Standing constraints (still valid)
- ≤15 GiB; ≤2 low-memory Writers; serial pytest; BLAS/OpenMP/MKL/Polars threads=1
- No `git checkout|restore|stash|clean|reset`; no bulk AST rewrites
- Single-writer on shared authorities (`operator_catalog.py`, `operator_policy.py`)
- Unrun gates = `NOT_RUN`
