# U05 dependency-source review — 2026-09-12

Static and synthetic verification only in server-c `/home/sunhaiwei/quant_projects`; no unauthorized real-data read was attempted.

## Findings and narrow repair

- Managed logical dependencies require a dataset-specific approved content digest; a daily-source approval cannot substitute for missing financial/minute approval.
- Child DataAccess sources inherit date/instrument scope, production/PIT/strict mode, semantic filters, approved snapshot/content maps, and the same parent broker. Binding failure closes the child.
- Financial batches use `prepare_read(..., run_mode="production", snapshot_policy="fail_if_changed", mode="event")`, retain pre-end PIT history and instrument scope, compare the frozen prepared dataset/content digest exactly, and execute only after that comparison.
- Digest mismatch releases the preparation reservation without execute. Success and execute failure rely on DataAccess `execute_prepared_read`, whose `finally` releases exactly once; synthetic counters verified both paths.
- `MissingDataDependencyError.reason_code=DATA_SOURCE_MISSING` was previously flattened to `UNKNOWN_FIELD`/`SOURCE_SCOPE_UNAVAILABLE` in physical preflight. The narrow repair preserves typed `reason_code`, while unknown fields remain `UNKNOWN_FIELD` and unrelated infrastructure errors remain `SOURCE_SCOPE_UNAVAILABLE`.
- Managed artifact reads were additionally hardened per follow-up: manifest metadata gets a broker lease before bounded allocation and every invocation uses unique UUID lease ids for metadata and value leases.

## Tests

```text
U05 dependency approvals + physical source binding + artifact purpose
34 passed, 3 warnings in 2.00s

U05 dedicated dependency tests
5 passed in 0.25s

legacy LQTP runtime suites after the capability serializer repair
28 passed, 5 warnings in 47.11s
```

The previously failing capability-manifest node now passes; no Legacy LQTP node is deselected.

No commit, push, deployment, branch, worktree, dependency installation, or production publication was performed.
