# ARCH-P0 DataAccess I/O Boundary Known Violations

## Status: 30 violations documented (not silently blessed)

These modules currently perform direct physical I/O operations outside the
authorized DataAccess boundary. Each represents technical debt that bypasses:
- Point-in-time (PIT) validation
- Security/credential governance  
- Schema versioning
- Resource accounting
- Audit trails

## Migration Plan

All violations must be migrated to use DataAccess abstractions. No new
violations are permitted (enforced by pre-commit hook).

## Known Violations (30 total)

### Backend (3 violations)
- `backend/fastpath_plan_probe.py:146` - duckdb.connect
- `backend/sql_pushdown/duckdb_capabilities.py:58` - duckdb.connect

**Migration:** Use DataAccess query interface instead of direct DuckDB connection.

### Export (2 violations)
- `export/importers.py:141` - pq.read_table
- `export/serializers.py:219` - .to_parquet

**Migration:** Use DataAccess read/write interfaces with proper governance.

### Runtime (7 violations)
- `runtime/intermediate_registry.py:59` - pd.read_parquet
- `runtime/multibackend/batch_transfer_optimizer.py:190` - duckdb.connect
- `runtime/reconcile/snapshot_reconcile.py:31` - pq.read_table
- `runtime/resource_calibration_store.py:328` - pd.read_parquet
- `runtime/shard_executor.py:68` - pd.read_parquet
- `runtime/spill_store.py:192` - pd.read_parquet

**Migration:** Use DataAccess governed reads/writes. Runtime should not directly
access physical storage.

### Scripts (18 violations)

#### Audit/Certification Scripts (6)
- `scripts/audit_r37_hard_gates.py:130, :155` - pl.read_parquet (2x)
- `scripts/certify_factor_operator_evidence.py:56` - pd.read_parquet
- `scripts/certify_intraday_parity.py:67` - duckdb.connect
- `scripts/certify_primitive_evidence.py:264` - pd.read_parquet
- `scripts/sql_certification_factory.py:223` - duckdb.connect

**Migration:** These are MAINTENANCE scripts. Can be categorized as ExemptionCategory.MAINTENANCE
in PHYSICAL_IO_AUTHORITY_POLICY if they are one-time/infrequent operations.

#### Benchmark Scripts (11)
- `scripts/duckdb_parallel_tuning_benchmark.py:103, :142, :179, :219, :266, :308, :353, :395, :436` - duckdb.connect (9x)
- `scripts/storage_tuning_benchmark.py:122, :154` - pq.read_table (2x)

**Migration:** These are BENCHMARK scripts. Should be categorized as ExemptionCategory.BENCHMARK.

#### Data Comparison/Testing Scripts (2)
- `scripts/compare_lqtp_golden.py:15` - pd.read_parquet
- `scripts/calibrate_backend_costs.py:179` - duckdb.connect

**Migration:** MAINTENANCE/BENCHMARK category depending on usage frequency.

#### Evidence Generation Scripts (1)
- `scripts/generate_model_layer_redesign_evidence.py:269` - .to_parquet

**Migration:** MAINTENANCE category if one-time, or refactor to use DataAccess.

## Scanner Infrastructure Failures (2)

These files have syntax errors and cannot be scanned:
- `cleaned_operators/lqtp_compat.py:11` - SyntaxError: unexpected indent
- `cleaned_operators/technical/polars_signal.py:100` - SyntaxError: unmatched ')'

**Action:** Fix syntax errors immediately. These files may contain hidden violations.

## Next Steps

1. Fix 2 scanner infrastructure failures (P0)
2. Categorize script violations appropriately in PHYSICAL_IO_AUTHORITY_POLICY
3. Migrate runtime/ and backend/ violations to DataAccess (P1)
4. Migrate export/ violations to DataAccess (P2)
5. Enforce pre-commit hook to prevent new violations

## Hard Gate Status

✓ ARCH-P0-001: Violations cause exit code 1 (fail-closed)
✓ ARCH-P0-002: Scanner failures cause exit code 2 (infrastructure failure)
✓ ARCH-P0-003: Single authority policy established

**Current state:** Honest boundary enforcement with documented technical debt.
**Production readiness:** NOT READY until runtime/backend violations migrated.
