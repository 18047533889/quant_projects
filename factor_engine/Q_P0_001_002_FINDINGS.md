# Q-P0-001/002 Capability Authority (compact)

**Audit:** 2026-08-14 historical snapshot · **production:** NO.

## Finding

A manual declared-native list diverged from executable Q lowerings: 110 declared vs 80 lowered; 30 declarations lacked lowering. This made capability claims unsafe and could permit silent fallback/cost-model error.

Historical gap: `bfill`, `cs_clip`, `cs_percentile_rank`, `cs_quantile`, `cs_winsorize`, `group_{count,max,mean,median,min,std,sum}`, `replace`, `resample`, `time_bucket`, `true_range`, `ts_{argmax_age,argmin_age,decay_exp,decay_linear,distance_to_high,distance_to_low,max_drawdown,moment,new_high,new_low,percentile,quantile,sum_decay}`, `vwap`.

## Required authority

`production_safe = declared ∩ lowering ∩ compile ∩ runtime ∩ parity/null evidence`, bound to current HEAD and runtime identity. Missing evidence means unsupported/fail-closed—not fallback under a native claim.

## Artifacts

- `backend/q_backend/q_capability_evidence.py`
- `tests/q_backend/test_q_capability_evidence.py`
- Current compiler/registry evidence under `evidence/r2/`

## Current-use warning

Counts and code locations above are historical; re-run scoped gates against the working tree before acting. Later repairs changed compiler/registry authority. Do not reintroduce a second authority or bulk-implement unproved lowerings.

## Gates

- Q native-without-lowering: historical **FAIL (30)**; current result `NOT_RUN` here.
- Manual-authority removal: historical **FAIL**; current result `NOT_RUN` here.
- Compile/runtime/parity/null/PIT/root CI/production certification: `NOT_RUN` unless a current-SHA evidence artifact says otherwise.

Q remains `NOT_PRODUCTION_CERTIFIED`.
