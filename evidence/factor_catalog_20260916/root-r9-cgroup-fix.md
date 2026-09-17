# Shared cgroup / process-cap domain correction

Formal worktree: server-c `/home/sunhaiwei/quant_projects`. No branch, commit, push, deployment or production factor publication.

## Root cause

The live server's SSH cgroup is unlimited and shared: its memory.current exceeded 29 GiB. A broker with an 8 GiB process cap incorrectly subtracted that shared usage from the process cap, producing a false zero. Independently, an unlimited leaf stopped the cgroup limit scan before finite ancestors.

## Implementation

- `factor_engine/runtime/resource_governor.py`: scan past unlimited leaves; pair each finite ancestor's own memory.max/current; finite limit with unreadable usage remains denied. Live headroom no longer pairs a process cap with shared-group usage.
- `factor_engine/runtime/resource_broker.py`: retain shared current as telemetry; snapshot separately carries paired cgroup remaining; the budget formula receives paired headroom without the mismatched current fallback. Unknown host availability remains denied.
- Pure budget formula and its genuine-zero safety tests are unchanged. No cap inflation or admission bypass.

## Verification

- `test_cgroup_memory_domains.py`: RED 6 failed / 1 passed before code changes; afterward all 7 passed.
- Combined domain, auto-budget, broker, authority regressions: 60 passed, exit0. `cgroup-memory-domains-green-r2.log`; watchdog sampled peak747716608 bytes.
- One obsolete global-boundary test had expected COMPUTE to consume the entire shared pool, conflicting with the existing protected egress reserve. It now verifies that COMPUTE is denied, RESULT_QUEUE can use the complete pool, the next byte is denied, and release permits readmission. External measurements are frozen; real admission/accounting remains exercised.
- Fresh live-server probe after fix: process cap8589934592, shared-group current31556149248, unlimited paired cgroup remainingNone, host available37539213312, process-family RSS157552640, execution budget6745905561, KNOWN_NONZERO. This is a timestamp-specific observation, not a fixed budget promise.
- Python compilation and scoped diff check passed. Small owned patch transfer files removed.

Further independent review and the broader CSE regression run remain separate evidence. This correction is not a guarantee of all-factor correctness or zero OOM risk.
