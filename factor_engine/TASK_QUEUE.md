# Loop Engineering Task Queue
**Updated:** 2026-08-14T00:05:00Z

## DISCOVERED (from BackendParityChecker)

### P0 Tasks (2)

**P0-01: Registry initialization error blocks testing**
- Type: BLOCKER
- Details: `ts_mean_abs_deviation` alias error prevents test execution
- File: Likely in operator_catalog.py or registry initialization
- Impact: Blocks all parity test runs
- Assignee: Pending triage
- Evidence: BackendParityChecker report

**P0-02: Technical indicators untested (8 operators)**
- Type: NO_TEST
- Operators: EMA, SMA, RSI, MACD_line, MACD_signal, bollinger_upper/mid/lower
- Impact: Production-critical indicators lack backend parity verification
- Severity: P0 - widely used in production
- Assignee: Pending triage
- Evidence: BACKEND_PARITY_FINDINGS.md:72-82

### P1 Tasks (5)

**P1-01: Statistical operators untested (5 operators)**
- Type: NO_TEST
- Operators: correlation, covariance, skew, kurt, quantile
- Impact: Core statistical operations lack parity tests
- Assignee: Pending triage
- Evidence: BACKEND_PARITY_FINDINGS.md:58-69

**P1-02: cs_zscore missing parity test**
- Type: NO_TEST
- Operator: cs_zscore
- Impact: Cross-sectional family incomplete
- Assignee: Pending triage
- Evidence: BACKEND_PARITY_FINDINGS.md:45

**P1-03: Q backend parity unverified**
- Type: MISSING_INTEGRATION
- Details: Q backend implemented but not in four-way parity suite
- Impact: Q backend may diverge silently
- Assignee: Pending triage

**P1-04: Known ewm divergences**
- Type: DOCUMENTED_BUG
- Operators: ewm_std, ewm_mean, EMA, ts_kurt
- Details: Polars NaN hole behavior differs from pandas recursive calculation
- Assignee: Pending triage
- Evidence: Memory files

**P1-05: Polars .to_pandas() fake native (7 instances)**
- Type: PERFORMANCE
- Files: polars_panel.py, panel_polars.py, long_frame.py, duckdb_optimizer_integration.py, duckdb_performance.py
- Impact: Performance regression, not truly native
- Assignee: Pending triage
- Evidence: BACKEND_SCAN_FINDINGS.txt

### P2 Tasks (2)

**P2-01: Q backend test infrastructure TODO**
- Type: TODO
- File: /home/shw/quant_projects/factor_engine/backend/q_backend/q_capability_evidence.py
- Details: compile/runtime/parity verification infrastructure needed
- Assignee: Pending triage

**P2-02: Base backend abstract method implementation**
- Type: NOT_IMPLEMENTED
- File: /home/shw/quant_projects/factor_engine/backend/base.py
- Details: Abstract method raises NotImplementedError if called directly
- Assignee: Pending triage

## TRIAGED (0)

## IMPLEMENTED (0)

## VALIDATED (0)

## APPROVED (0)

## REJECTED (0)

---

## Summary
- Total discovered: 9 tasks (2 P0, 5 P1, 2 P2)
- Awaiting: OperatorUsabilityAuditor, AuditMiner-Operator results
- Next action: Launch TriageSpecialist when queue >= 10 or critical mass reached
