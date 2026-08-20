---
name: factor-engine-rules
description: Mandatory compact FactorEngine/DataAccess governance — always on
alwaysApply: true
---

# Rules

- **LOCAL ONLY:** no GitHub/gh/fetch/push/remotes. Working tree is truth. Never `git checkout|restore|stash|clean|reset|rebase`. Never bulk AST/formula rewrites. No bulk loads of `archives/`.
- **Context:** Read `loop/care.md` first, then `loop/status.md`, `loop/queue.md`, `loop/resume.md`. Brief ≤30 lines per file. Never load `docs/R2_HISTORY_ARCHIVE.md`.
- **Loop:** `loop/orchestration.md` + `loop/start_fe.md`. Agent roster: `.claude/agents/`. Coordinator does not bulk-edit.
- **Reviewer:** `git diff` + one evidence YAML. Template: `.claude/agents/reviewer.md`.
- **Workflow:** reproduce → narrow edit → non-vacuous oracle → syntax/import smoke → manifest → independent scoped review → local integration. No real/full datasets unless explicitly requested.
- **Math:** never guess corrupted formulas/denominators. Use trusted pre-image if available; otherwise mark manual review. Pandas/canonical semantics are reference. Check numerical edge cases and PIT/look-ahead/label/temporal leakage. Model work needs walk-forward/OOS evidence.
- **Authorities:** one capability registry, one RegionPlanner, one semantic schema. `PhysicalRegionPlan` only; no executor reroute; transfers only at `TransferEdge`.
- **Backends:** explicit `ExecutionKind`; `.to_pandas()`, Python/NumPy loops, or UDFs are not Polars-native. DuckDB is opt-in/fail-closed. Set Polars threads only before import.
- **Q compiler:** canonical-IR only, zero semantic authority, fail-closed until certified; region residency / no operator ping-pong; prove runtime, parity, warmup/null/time/PIT semantics before `PRODUCTION_SAFE`.
- **Placeholders/TODO proxies:** implement and certify or mark unsupported/research-only. Never production-register under the canonical name.
- **Evidence:** YAML records current `git_sha`, timestamp, owned/modified files, tests run/pass/not-run, limitations. Any unrun full compile/import, registry/duplicate-authority, ABI/placeholder, backend parity, PIT poison, Q null/time/runtime, CI, or production gate is `NOT_RUN` — never PASS/production-ready.

## GitHub Push

After every significant change (new operator, planner fix, backend change, test addition), run:

```bash
cd /home/shw/quant_projects
git add factor_engine/ dataaccess/ evidence/ tests/
git diff --cached --stat
git commit -m "..."
git push origin main
```

Token is stored. If push fails due to diverged branch, fetch+merge, resolve conflicts, then push.

## Bloat Cleanup

If loop/session grows huge: `bash /home/shw/quant_projects/scripts/cleanup_claude_bloat.sh`
