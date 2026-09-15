# Default durable HTTP service repair evidence

Date: 2026-09-12 (Asia/Shanghai)

These files preserve the original pytest completion output captured in the
agent tool transcript. Tests were not rerun merely to create evidence files.

## Environments

- System: Python 3.12.3, pandas 3.0.5, NumPy 2.5.2, Polars 1.44.1.
- Project `.venv`: pandas 2.3.3, NumPy 2.2.6, Polars 1.42.1.

## Results and commands

- Full service suite: 84 passed. The exact command and completion output are
  in `R3-service-default-durable-service84-20260912.log`.
- Focused default-durable and receipt-protocol tests under the system Python:
  29 passed. See `R3-service-default-durable-system29-20260912.log`.
- The same focused tests under the project virtual environment: 29 passed.
  See `R3-service-default-durable-venv29-20260912.log`.

## Contract boundary

- Submission binds the approved deployment, resolved default policy, timeout,
  operator catalog, and execution purpose; dispatch rejects identity drift.
- Owner access compares identity, tenant, and project. The explicit ADMIN
  bypass remains unchanged.
- An ABORTED receipt is retained only after bounded reading, approved-root
  containment, durable run association, and receipt identity verification.
- Manual retry is an explicit new-job attempt: a new run ID, incremented
  lineage, and fresh deadline are created. It is not same-run resume. The new
  attempt keeps the original immutable request, resolved policy and timeout.
  These are service contract tests, not approved-data default-run acceptance.
