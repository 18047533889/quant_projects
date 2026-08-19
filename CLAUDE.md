# quant_projects — Claude entry (keep short)

## Execute
- No plan mode unless asked. No “是否继续”. Implement end-to-end.
- Working tree is truth. Never `git checkout|restore|stash|clean|reset` or bulk AST rewrites.
- Push/GitHub only when the user explicitly asks.

## Context budget
- Status SoT: `LOOP_ENGINEERING_STATUS.md` (+ `RESUME_STATE.md` / `AGENT_STATUS.md` / `AUTONOMOUS_MASTER_QUEUE.md`).
- **Never** load into prompts: `docs/R2_HISTORY_ARCHIVE.md`, `archives/**/*_full_*`, `archives/taskbooks/*`, Claude `memory/archives/*`, multi-100KB dumps, or root `FactorEngine_*.md` / `DataAccess_*.md` stub targets.
- Append status as one-line `ID → result → evidence path`. Subagent briefs ≤30 lines + owned files + evidence paths.
- **Independent review:** scoped-reviewer agent; input = `git diff -- <files>` + evidence YAML only (no full tests, no DIRECTUSE matrix, no parent history). After skill changes: `bash scripts/sync_claude_worktree_skills.sh`.
- FE skill: `factor_engine/.claude/skills/factor-engine-rules` (already compact). Blueprint package work: skim `quant_factor_platform_blueprint/CLAUDE.md`; load `AI_GUIDE/*.md` only for the package you touch.

## Resources
- ≤15 GiB total; ≤2 disjoint low-memory Writers + 1 read-only Auditor.
- Serial tests: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1`; no xdist.
- Unrun gates stay `NOT_RUN`. Never claim production-ready without release-grade provenance.

## Session hygiene
- No `/loop` or `scheduled_tasks` unless user explicitly asks **this turn**.
- Max 5 active todos; cancel completed Wave-0 / “78 subagents” platform tasks.
- Bloat cleanup: `bash scripts/cleanup_claude_bloat.sh` then `/clear`.

## Lanes
- **Platform packages** (QE / FO / FA / FP / modeling): primary focus — see `PLATFORM_LOOP_STATUS.md`. FE/DA edits OK when narrowly needed to unblock platform work.
- **FE / DA / Q / Polars**: can be a separate deep-dive session; don’t let them swallow the whole platform loop by default.
