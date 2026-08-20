# What this project cares about (portable)

Claude: **this file + repo-root `CLAUDE.md` is enough to start.**  
Do not load `archives/`, `docs/R2_HISTORY_ARCHIVE.md`, taskbooks, or `~/.claude/**/memory/archives/`.

Live tickets/status (read only the file for your lane, not wholesale dumps):

- FE/DA: [`status.md`](status.md) · [`queue.md`](queue.md)
- Platform: [`platform.md`](platform.md)
- How to run agents: [`orchestration.md`](orchestration.md)

## Who / how to work

- User cares about **correct numbers that can become trading signals**, not cosmetic refactors or “we audited the whole repo”.
- **Execute immediately.** No plan mode unless asked. No “是否继续”. Finish the request.
- **Code changes go through subagents** (Finder → Dispatcher → Writer → Tester → Reviewer). Main session = Coordinator; no bulk source edits in main.
- **Never idle:** a finished subagent means spawn the next role (or the next cycle) immediately. User silence is not a stop. Stop only on 停/`/clear` or `loop/HALT`.
- **Working tree is truth.** Do not `git pull` to overwrite local work. Do not `git checkout|restore|stash|clean|reset`. Concurrent agents share this tree; discarding files wipes someone else’s uncommitted work. Revert by editing lines back.
- Default **LOCAL ONLY**: no `gh` / fetch / push / remotes unless the user explicitly says push/commit.
- Commit only when asked. Never commit `.env` or secrets. Author when committing: `Sun Haiwei <2946703196@qq.com>` (via env for that command; do not rewrite git config).
- Never claim PASS / production-ready / CLOSED_VERIFIED for an unrun gate. Unrun = `NOT_RUN`.

## Stack (do not rebuild)

Pre-model only: `dataaccess` → `factor_engine` → `quant_evaluator` → (optional `factor_optimizer`) → `factor_assets` → `factor_preprocess` → `modeling`.

| Truth lives in | Not here |
|---|---|
| Data / PIT / calendar / universe / snapshot | `dataaccess` |
| DSL / AST / IR / operators / materialize | `factor_engine` |
| Metrics | `quant_evaluator` |
| Mutations / search | `factor_optimizer` |
| Identity / lifecycle | `factor_assets` |
| Pre-model transforms | `factor_preprocess` |

Do **not** build a second data lake, PIT, calendar, universe, DSL, materializer, cache platform, or generic DAG.

Package `dataaccess/` imports as `data_access`. FE needs `PYTHONPATH` to `factor_engine/`.

## Engineering that actually matters

- **No bulk AST / regex rewrites** of operator libraries (`cleaned_operators/` and similar). Two past agents corrupted 160–194 files. Change operators **file-by-file**, compile each file, focused tests. Silent NaN instead of a loud failure will leak into live signals.
- Skip `operator_catalog.py` / `operator_policy.py` unless the ticket owns them exclusively.
- One capability registry, one RegionPlanner, one semantic schema. `PhysicalRegionPlan` only; no executor reroute; transfers only at `TransferEdge`.
- Backends: explicit `ExecutionKind`. `.to_pandas()`, Python/NumPy loops, UDFs are **not** Polars-native. DuckDB opt-in / fail-closed. Set Polars threads **before** import.
- **Q backend:** compiler only, zero semantic authority, fail-closed until certified. No operator ping-pong. Prove runtime, parity, warmup/null/time/PIT before `PRODUCTION_SAFE`.
- Placeholders: implement+certify or mark unsupported/research-only. Never production-register a stub under the canonical name.
- Math: do not guess corrupted formulas. Pandas/canonical semantics are reference. Check PIT / look-ahead / label leakage. Models need walk-forward / OOS, not in-sample theater.
- `CLOSED_LOCAL ≠ PRODUCTION_CERTIFIED`. Oracles must not copy the implementation. Do not flip flags and call it a math fix.
- Backend investment order: Polars ≈ DuckDB > Numba physical kernels > ClickHouse > q > GPU.
- Evidence: one YAML (or one status line) with `git_sha`, files, tests run/pass/`NOT_RUN`, limitations.

## Resources

- **≤15 GiB** per Claude session. ≤2 Writers, disjoint files. Serial pytest. No `pytest -n auto`.
- Prefix tests: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1`.
- `exit 137` = OOM → smaller shard, not “tests failed”. Broad QE / root CI / live COS / real Redis stay `NOT_RUN` until actually run.
- Subagent brief ≤30 lines. Reviewer: `git diff` + **one** evidence YAML. Never paste parent chat or huge tests into a child.

## Current certification (honest)

Q / FactorAssets / Modeling: **not** production-certified.  
Polars / DuckDB: partially verified.  
Root CI, live COS/S3, real Redis, broad QE: `NOT_RUN`.

Live work is the open rows in [`queue.md`](queue.md), not archived R21/R20 taskbooks (`archives/taskbooks/`).

## New machine

1. Copy this repo (including `loop/` and `.claude/`).
2. Run `bash scripts/bootstrap_claude_host.sh` from repo root (writes `~/CLAUDE.md` pointer; does not copy secrets).
3. Copy `.env` yourself. Do not recreate mega `scheduled_tasks.json` prompts.
4. Open the repo; Claude auto-loads root `CLAUDE.md` → this file.

Usage docs for humans: `docs/量化平台使用总览.md` (not for agent auto-load).
