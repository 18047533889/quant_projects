# FactorEngine Loop Status

## Current State

```
coordinator: idle
active_agents: none
last_update: 2026-08-21
```

## Active Items

- Writer CG: completed ClickHouse adapter isolation locally (working tree).

## Recent Activity

- 2026-08-21: Added `ClickHousePushdownBackend` and isolated adapter boundary for `clickhouse_sql`.
- 2026-08-21: Added adapter regression tests and `evidence/r2/R21-CLICKHOUSE-ADAPTER.yaml`.
- 2026-08-21: Confirmed adapter/factory/certificate tests pass (7 tests).
- 2026-08-21: Updated `loop/queue.md` / `loop/status.md` / `loop/resume.md` with current adapter work.
