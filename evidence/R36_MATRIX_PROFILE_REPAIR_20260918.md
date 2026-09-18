# R36 matrix-profile performance repair

Directly modified the server-c main worktree; no branch, copied repository,
deployment, or production-factor publication.

## Production change
`factor_engine/cleaned_operators/intraday/topology_manifold.py` now
precomputes rolling statistics and evaluates nearest-neighbor distances in
NumPy candidate tiles instead of recalculating each candidate in Python.
No N-by-N distance matrix is materialized. The 8 MiB setting is a transient
tile target, not a hard total memory limit: linear support arrays, an
O(window) query and optional <=4 MiB cache coexist; reductions can use
additional temporaries.

Existing ddof=1, exclusion zone, tie and degenerate-result behavior is
preserved. Existing finite-return compaction across gaps is also preserved,
not certified as an appropriate missing-data model for every use case.

## Verification
Root independent rerun:
`evidence/r36-matrix-profile-root.log`: **26 passed**.
Command: pytest -q --tb=short
factor_engine/tests/operators/test_r32_matrix_profile_kernel.py
factor_engine/tests/test_intraday_topology_manifold.py

Includes naive-formula comparison, missing/degenerate inputs, forced 256-byte
tile target, and a 1500-bar synthetic smoke test under 5 seconds.
Watchdog wall 57.167 seconds; sampled process-family peak RSS 798769152 bytes.
This sampled guard is not a kernel hard memory cap. This is not a benchmark
of full run_many, nor evidence that all operators or 110000 factors pass.
