---
name: factor-engine-rules
description: Mandatory compact FactorEngine/DataAccess governance
alwaysApply: true
---

# Rules

- **LOCAL ONLY:** no GitHub/`gh`/fetch/push/remotes. Working tree is truth; never `git checkout|restore|stash|clean|reset`; no bulk AST/formula rewrites.
- Context: read compact status/resume/queue only; never load `docs/R2_HISTORY_ARCHIVE.md` or full dumps. Agent brief ≤30 lines: task, owned files, evidence pointers.
- Resources: ≤15 GiB total; ≤2 disjoint low-memory Writers + 1 read-only Auditor. Serial pytest, no xdist; set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1` before imports.
- Ownership: declare files; no overlap; shared authorities (`operator_catalog.py`, `operator_policy.py`) single-writer. On prompt/524 instability shrink input/reduce to one Writer.
- **Scoped review:** read-only reviewer gets `git diff` + one evidence YAML only—not full test modules, not DIRECTUSE matrices, not parent session. Template: `.claude/agents/scoped-reviewer.md`.
- Workflow: reproduce → narrow edit → non-vacuous oracle → syntax/import smoke → manifest → independent scoped review → local integration. No real/full datasets unless explicitly requested.
- Math: never guess corrupted formulas/denominators. Use trusted pre-image if available; otherwise mark manual review. Pandas/canonical semantics are reference. Check numerical edge cases and PIT/look-ahead/label/temporal leakage; model work needs walk-forward/OOS evidence.
- Authorities: one capability registry, one RegionPlanner, one semantic schema. `PhysicalRegionPlan` only; no executor reroute; transfers only at `TransferEdge`.
- Backends: explicit `ExecutionKind`; `.to_pandas()`, Python/NumPy loops, or UDFs are not Polars-native. DuckDB is opt-in/fail-closed. Set Polars threads only before import.
- Q: canonical-IR compiler only, zero semantic authority, fail-closed until certified; region residency/no operator ping-pong; prove runtime, parity, warmup/null/time/PIT semantics before `PRODUCTION_SAFE`.
- Placeholders/TODO proxies: implement and certify or mark unsupported/research-only; never production-register under the canonical name.
- Evidence: YAML records current `git_sha`, timestamp, owned/modified files, tests run/pass/not-run, limitations. Any unrun full compile/import, registry/duplicate-authority, ABI/placeholder, backend parity, PIT poison, Q null/time/runtime, CI, or production gate is `NOT_RUN`—never PASS/production-ready.
