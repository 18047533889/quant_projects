# Resume / Notes

## Active Blockers

- **⚠️ LIVE GitHub PAT leaked (2026-08-22, found by CI security scan):** token `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` is embedded in (1) `.claude/skills/github-push/SKILL.md` — tracked and already pushed to both remotes (commit 91b62620) — and (2) both git remotes (`origin`, `hkust-origin`) via `x-access-token:...@` URLs. **Should be revoked** in GitHub settings (it's a `ghp_` PAT) and replaced with a credential helper / scoped token. Coordinator did NOT modify remote URLs or the skill file (user decision).

## Findings from Last Session

- **Bare `python` is NOT on PATH** in this shell — use `/home/shw/quant_projects/.venv/bin/python` (verified `.venv/bin/python` imports polars/numpy). Backgrounded bash runs must use the venv path.
- **QE numerical oracle collection error**: repo-root `quant_evaluator/` stub shadows `build/lib/quant_evaluator` under pytest; `test_numerical_oracle.py` only prepended build/lib to sys.path, so `quant_evaluator.contracts.factor_batch` resolves to the stub. Fix = full sys.modules bootstrap from `test_metamorphic.py`.
- **FO checkpoint/resume 5 failures**: (1) `SearchSpace.to_array` does `float(point['c'])` on a choice param ('b') → ValueError; (2) `SearchRunner.resume` raises "cannot resume a finished session" for a fully-budgeted legacy session — test semantics need a defined contract. Fix agent dispatched.
- **VerificationManifest stale**: manifest_id rel-v1:2026-08-21T12:58:18 was generated at git_sha b05a888f (current 4a27a8b2), ALL 14 gates NOT_RUN. Refresh agent dispatched with real gate evidence.
- ClickHouse SQL execution should not share a concrete backend class with DuckDB.
- `build_backend('clickhouse_sql')` previously relied on runtime attribute injection to convey adapter identity; this creates a fragile boundary for certificates and telemetry.
- Cross-dialect certificate rejection requires both the certificate and runtime event to declare a dialect; event-only dialect is not rejected by current validation logic.
- Production execution is planner-certified-plan only (no executor silent pandas fallback); documented in `api/mining_integration.py` top docstring.

## Next Steps

- Await the 10 dispatched agents; review their reports, then commit+push the accumulated approved changes.
- After each commit, update VerificationManifest source_snapshot git_sha (4a27a8b2 → new sha) via the manifest refresh script.
- Continue to the next audit sweep (DETERMINISM, LEAKAGE, PIT, CHECKPOINT_RESUME gates) once current batch lands.

## Session Log

- `<timestamp>` — Loop initialized
- `2026-08-21` — Completed ClickHouse adapter isolation locally and added `evidence/r2/R21-CLICKHOUSE-ADAPTER.yaml`.
- `2026-08-21` — Writer DW: production no-silent-fallback doc + regression tests + `evidence/r2/R21-PRODUCTION-NO-SILENT-FALLBACK.yaml`. Verified pre-existing `tests/api/test_mining_fastpath.py` failures are independent of this change (pre-dating, from R24 bare-field requirement); the fix is pending Writer DH's original R21-FASTPATH-MARKET task.
