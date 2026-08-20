---
name: factor-engine-rules
description: Mandatory compact FactorEngine/DataAccess governance
alwaysApply: true
---

# Rules

- **LOCAL ONLY:** no GitHub/`gh`/fetch/push/remotes. Working tree is truth; never `git checkout|restore|stash|clean|reset`; no bulk AST/formula rewrites.
- Context: `loop/care.md` then `loop/status.md` `loop/queue.md` `loop/resume.md`. Never load archives. Brief ≤30 lines.
- **Loop:** `loop/orchestration.md` + `loop/start_fe.md`. Agents: `.claude/agents/` (`writer-fe`, finder, dispatcher, tester, reviewer). Coordinator does not bulk-edit.
- **Reviewer:** `git diff` + one evidence YAML. Template: `.claude/agents/reviewer.md`.
- Workflow: reproduce → narrow edit → non-vacuous oracle → syntax/import smoke → manifest → independent scoped review → local integration. No real/full datasets unless explicitly requested.
- Math: never guess corrupted formulas/denominators. Use trusted pre-image if available; otherwise mark manual review. Pandas/canonical semantics are reference. Check numerical edge cases and PIT/look-ahead/label/temporal leakage; model work needs walk-forward/OOS evidence.
- Authorities: one capability registry, one RegionPlanner, one semantic schema. `PhysicalRegionPlan` only; no executor reroute; transfers only at `TransferEdge`.
- Backends: explicit `ExecutionKind`; `.to_pandas()`, Python/NumPy loops, or UDFs are not Polars-native. DuckDB is opt-in/fail-closed. Set Polars threads only before import.
- Q: canonical-IR compiler only, zero semantic authority, fail-closed until certified; region residency/no operator ping-pong; prove runtime, parity, warmup/null/time/PIT semantics before `PRODUCTION_SAFE`.
- Placeholders/TODO proxies: implement and certify or mark unsupported/research-only; never production-register under the canonical name.
- Evidence: YAML records current `git_sha`, timestamp, owned/modified files, tests run/pass/not-run, limitations. Any unrun full compile/import, registry/duplicate-authority, ABI/placeholder, backend parity, PIT poison, Q null/time/runtime, CI, or production gate is `NOT_RUN`—never PASS/production-ready.
