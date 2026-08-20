# Resume / Notes

## Active Blockers

(None yet)

## Findings from Last Session

- ClickHouse SQL execution should not share a concrete backend class with DuckDB.
- `build_backend('clickhouse_sql')` previously relied on runtime attribute injection to convey adapter identity; this creates a fragile boundary for certificates and telemetry.
- The compiler IR / emitter / executor chain is still safe to share; only the adapter runtime boundary needs separation.
- Cross-dialect certificate rejection requires both the certificate and runtime event to declare a dialect; event-only dialect is not rejected by current validation logic.

## Next Steps

- Consider promoting adapter metadata defaults into a small `SqlPushdownAdapterSpec` dataclass if additional SQL engines are added later.
- If DuckDB/ClickHouse runtime divergence grows, extract adapter-owned query-budget/connection settings into adapter methods.

## Session Log

- `<timestamp>` — Loop initialized
- `2026-08-21` — Completed ClickHouse adapter isolation locally and added `evidence/r2/R21-CLICKHOUSE-ADAPTER.yaml`.
