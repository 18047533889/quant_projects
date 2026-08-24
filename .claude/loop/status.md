# FactorEngine Loop Status

## Current State

```
coordinator: idle
active_agents: none
last_update: 2026-08-21
```

## Active Items

- Writer DW: corrected production doc in `api/mining_integration.py` (no executor silent pandas fallback).

## Recent Activity

- 2026-08-21: Added `ClickHousePushdownBackend` and isolated adapter boundary for `clickhouse_sql`.
- 2026-08-21: Added adapter regression tests and `evidence/r2/R21-CLICKHOUSE-ADAPTER.yaml`.
- 2026-08-21: Confirmed adapter/factory/certificate tests pass (7 tests).
- 2026-08-21: Updated `loop/queue.md` / `loop/status.md` / `loop/resume.md` with current adapter work.
- 2026-08-21: Writer DW: `api/mining_integration.py` top docstring now documents planner-certified-plan production routing + typed failure + explicit replan; no silent pandas fallback. Added `tests/api/test_production_no_silent_fallback.py` (5 tests) + `evidence/r2/R21-PRODUCTION-NO-SILENT-FALLBACK.yaml`.
