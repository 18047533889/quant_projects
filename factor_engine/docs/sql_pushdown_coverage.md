# SQL pushdown coverage

Generate the current active-canonical report with:

```bash
python factor_engine/scripts/sync_backend_docs.py
```

The report distinguishes SQL emitter implementation, real DuckDB parity verification, and DuckDB production-safe routing. ClickHouse certification is independent and is never inferred from DuckDB support.
