# TRAIN layer decay: reuse QE quantile assignments by source date

## Scope and interpretation

The diagnostic compares a stale signal with the same scoring-date forward
label. For scoring row `t` and lag `l`, it bins the signal observed at source row
`s = t - l`, then evaluates those memberships against `y[t]`. This is a
predictive decay diagnostic, not a holding-period portfolio backtest. It runs
on TRAIN rows only. The selected decay values are locked before any
VALIDATION or TEST evaluation; those partitions do not select, refit, or tune
the decay rule.

QE remains the authority for both membership assignment and return
aggregation. FO gathers previously assigned membership IDs and calls the QE
preassigned aggregation API; FO does not implement a separate rank or return
calculation.

## Membership and return definition

Let $x_{s,n}$ be the factor value for source date $s$ and asset $n$, and let
$b^{\mathrm{QE}}_{20,\max}(x_{s,:})_n$ denote asset $n$'s ID from QE's
twenty-bin assignment of the source row with the `max` tie policy. The
membership is

$$
m(s,n) =
\begin{cases}
b^{\mathrm{QE}}_{20,\max}(x_{s,:})_n,
& \text{if }x_{s,n}\text{ is finite and at least 20 source values are finite},\\
-1, & \text{otherwise.}
\end{cases}
$$

In API terms, QE computes this row with
`assign_quantiles_batch(x[s, :], n_quantiles=20, method="max")`.

`m(s,n)` is `-1` when the source factor value is non-finite or when QE cannot
form the requested bins under its normal rules (including too few finite
values for the requested bins). Otherwise it is a bin ID in `[0, 20)`. The
assignment depends only on the source factor row, its finite values, and the
QE tie policy. It does not depend on forward labels.

For scoring date $t$, lag $l$, and bin $q$, let $v_{t,n}$ be the label
validity flag and $y_{t,n}$ the forward return. The contributing asset set,
count, and mean return are

$$
\mathcal{A}_{t,l,q}
= \left\{n : m(t-l,n)=q,\;v_{t,n}=1,\;y_{t,n}\text{ is finite}\right\},
$$

$$
C_{t,l,q}=\left|\mathcal{A}_{t,l,q}\right|,
\qquad
R_{t,l,q}=
\begin{cases}
\displaystyle\frac{1}{C_{t,l,q}}\sum_{n\in\mathcal{A}_{t,l,q}}y_{t,n},
& C_{t,l,q}\geq a_{\min},\\
\mathrm{NaN}, & C_{t,l,q}<a_{\min},
\end{cases}
$$

where $a_{\min}$ is `min_assets`.

The return is `NaN` when the count is below `min_assets`; the count still
reports the valid labeled observations, matching the standard QE aggregation
contract. A missing or invalid label removes that asset from aggregation only.
It never changes `m(s,n)` or the bin cut points. For source rows before the
available history (`t-l < 0`), every membership is `-1`, so no asset enters a
bin.

## Layer excess and sampled half-life

All lags use the same scoring dates: let $G$ be the set of TRAIN dates for
which every lag and all twenty layer returns are finite. For each date, layer,
and lag, subtract the cross-layer mean return; then average over $G$:

$$
E_{t,l,q}=R_{t,l,q}-\frac{1}{20}\sum_{j=0}^{19}R_{t,l,j},
\qquad
\mu_{l,q}=\frac{1}{|G|}\sum_{t\in G}E_{t,l,q}.
$$

The initial direction is $d_q=\operatorname{sign}(\mu_{0,q})$. A layer is
stable only when that direction is nonzero, each of the three TRAIN folds has
at least ten common dates, and its mean lag-zero excess has the same direction
in every fold. For a stable layer, sampled half-life is the first evaluated
lag after zero at which the signed excess falls to at most half its initial
magnitude:

$$
h_q=\min\left\{l\in\mathcal{L}\setminus\{0\}:
d_q\mu_{l,q}\leq\frac{1}{2}|\mu_{0,q}|\right\},
\qquad
\mathcal{L}=\{0,1,\ldots,10,15,20\}.
$$

If no sampled lag crosses the threshold, the half-life is right-censored at
the final sampled lag; it is not treated as infinite. If the layer is not
stable, no half-life is reported. The two fixed scale proposals use the
median of the outer-layer half-lives, substituting the final sampled lag for a
censored outer layer, and keep only `scale/2` or `scale` in the range 3 to 60.
No scale is fitted against VALIDATION or TEST results.

## QE typed API

The lagged IDs passed to QE must already be aligned to scoring dates and have
shape `(T, N, F)`, including the singleton factor dimension. Axes carry exact
coordinates. The label bundle must use the same scoring decision times and
asset coordinates; QE rejects mismatches instead of inferring or reordering
them.

```python
from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.quantile_assignments import QuantileAssignmentBatch
from quant_evaluator.contracts.quantile_policy import QuantileTiePolicy
from quant_evaluator.metrics.quantile import (
    assign_quantiles_batch,
    compute_quantile_returns_from_assignments,
)

# score_times and asset_ids are coordinate arrays. score_times exactly equal
# label_bundle.decision_time; asset_ids exactly equal label_bundle.asset_axis.values.
score_time_axis = AxisRef(
    "time", str(score_times.dtype), len(score_times), score_times
)
asset_axis = AxisRef("asset", str(asset_ids.dtype), len(asset_ids), asset_ids)

# source_values has shape (source_rows, assets). QE bins each source row once.
source_ids = assign_quantiles_batch(
    source_values, n_quantiles=20, method="max"
)

# Gather IDs for a particular lag onto score times; negative-history rows use -1.
# lagged_ids has shape (score_rows, assets, 1), dtype int32.
lagged = QuantileAssignmentBatch(
    assignments=lagged_ids,
    time_axis=score_time_axis,       # AxisRef coordinates equal label decision_time
    asset_axis=asset_axis,           # coordinates equal label_bundle.asset_axis
    factor_ids=("delayed",),
    n_quantiles=20,
    tie_policy=QuantileTiePolicy.MAX,
)
returns, counts = compute_quantile_returns_from_assignments(
    lagged, label_bundle, min_assets=10
)
# returns and counts have shape (score_rows, 20, 1).
```

`QuantileAssignmentBatch` owns an immutable int32 copy and accepts only `-1`
or IDs in `[0, n_quantiles)`. Its explicit axes and factor IDs are part of the
contract, not hints. Assignment provenance carries the tie policy; aggregation
does not alter or reassign those IDs.

## FO entry point and result status

Call the TRAIN diagnostic as follows. Its optional
`assignment_cache_budget_bytes` defaults to `64 * 1024 * 1024` (64 MiB) and
must be a positive integer. Boolean values are rejected even though Python
treats `bool` as an integer. Invalid budgets raise `ValueError` before the
diagnostic evaluates whether the available TRAIN data is sufficient.

```python
record = diagnose_layer_decay(
    batch, labels, split, config, factor_index,
    assignment_cache_budget_bytes=64 * 1024 * 1024,
)
```

The returned `status` is `available` only when the diagnostic has enough
common TRAIN dates and coverage to report layer curves. Otherwise it remains
`unavailable`, with `reason` describing insufficient history, coverage, or a
budget refusal. `assignment_cache_status` is `bounded_unique_sources` after
the lag panels have been built, or `budget_exceeded` when the configured
budget cannot admit the minimum working set. It may be absent when the
diagnostic returns before assignment work begins. Per-layer output includes
`mean_excess_returns`, `stable_initial_direction`, `half_life_bars`, and
`right_censored`; `proposed_half_lives` contains only the fixed TRAIN-derived
scale proposals. These proposals are research inputs to be locked before
VALIDATION and TEST, not post-hoc values fit on those partitions.

## Source-date reuse and bounded working set

For each chronological scoring block, FO forms the sorted union of valid
source rows `idx_block - lags`. It invokes QE assignment on each unique source
row once in that block, retains the immutable ID chunks, and gathers the
appropriate source row for every `(score row, lag)` pair. Sparse or
noncontiguous TRAIN indices are supported through source-row lookup; the
implementation does not assume contiguous dates. A source date that is also
needed by a later score block can be binned again after the prior block is
released.

The algorithm does not build a full `lags x TRAIN x assets` membership cube.
It retains the compact `T_train x 20 x 13` float64 diagnostic result cube and
the unique source memberships needed for one scoring block. The 64 MiB
assignment-cache budget uses a conservative incremental array-work estimate
for that block. It includes the result cube, immutable cached IDs, gathered
and owned lag IDs, label/validity normalization headroom, QE sorting buffers,
and estimated boundary/interpolation scratch. Source assignments are split
into smaller chunks when the estimated working set requires it.

This is an estimator for the arrays accounted for by the diagnostic, not a
strict process-RSS limit. It does not include the already existing raw factor
panel, labels and metadata owned by the caller, the Python interpreter, or
unrelated process memory. If even the result cube plus one score row and its
minimum source working set exceed the configured budget, the diagnostic
fails closed with a budget-exceeded result. It does not retry by retaining a
full membership cube or by switching to an unbounded fallback.

For scale only, with `T_train=150`, `N=3000`, 13 lags and 20 bins, the result
cube occupies about 0.30 MiB. A 64-row block on contiguous indices uses at most
84 unique source rows, about 0.96 MiB for immutable int32 IDs, plus the other
estimated working arrays. Actual block sizing uses the exact unique-source
count for the selected indices. These arithmetic examples are not measured
peak RSS or performance results.

## Validation and performance status

The implementation preserves QE's assignment and aggregation semantics. The
CPU oracle checks compare assignment-based returns and counts with the
legacy per-lag route, including missing labels, invalid factors, sparse source
indices, pre-history rows, and axis alignment. The bounded-cache checks cover
budget admission and refusal when the minimum working set cannot fit.

Before the final output-buffer follow-up, the FO suite completed with 1,697
passing tests in 125.71 seconds, and the QE metrics/public quantile paths had
1,502 passing tests and 24 skipped in 155.97 seconds. Those are initial
shared-aggregation snapshot results, not final-source full-suite results or
benchmarks. The skipped cases require specialised inputs, so this does not
claim coverage of every QE metric path. After the output-buffer follow-up,
206 focused regression tests passed in 41.08 seconds. The final independent
oracle/output-buffer tests passed: 28 tests in 0.49 seconds. Counts overlap
across these runs and must not be summed into a unique-test total.

### CPU A/B measurements

The seeded synthetic benchmark used one warmup per arm and three ABBA rounds,
for six timed samples per arm. Every cached sample matched the scalar
percentile/bincount oracle and pinned baseline exactly before timing was
reported. These are CPU measurements on synthetic panels; they did not use
COS data or a GPU.

| TRAIN rows $T$ | Assets $N$ | Old median (s) | Reuse median (s) | Median ratio |
|---:|---:|---:|---:|---:|
| 300 | 200 | 0.121836 | 0.067549 | 1.804x |
| 500 | 1,000 | 0.572268 | 0.217685 | 2.629x |
| 1,200 | 5,461 | 7.353322 | 2.044372 | 3.597x |

Each case used its own fixed seed (81030, 81031, and 81032, respectively).
These measurements describe only these workloads and environment; they do
not establish expected performance on COS panels, other shapes, or production
hardware. Full samples, correctness details, and source hashes are recorded
in [`research_decay_assignment_reuse_20261003.json`](benchmarks/research_decay_assignment_reuse_20261003.json).
The runner is [`benchmark_research_decay_assignment_cache_oct03.py`](../scripts/benchmark_research_decay_assignment_cache_oct03.py).

### Standard QE value-path follow-up

A separate standard-path A/B used synthetic `T=512`, `N=5461`, `F=16`,
`Q=20`, `min_assets=2`, seed `81033`: one warmup per arm and three ABBA
rounds (six samples per arm), against optimized baseline `000eadc70`.
All returns and counts matched exactly in every timed sample.

The first shared-aggregation snapshot (`b92641b7…682ad`) measured 3.9023123
seconds versus 3.8224774 seconds for the old implementation (~2.1% slower).
It allocated and copied temporary per-factor output arrays. The follow-up
writes into the existing result views instead; shares-memory regression
tests enforce that property. Final QE source `bc27e018…24693d` measured
3.7848970 seconds versus 3.8354281 seconds for the pinned old implementation.

The final timings are close (~1.3% lower median), not proof of a significant
or universal standard-QE speedup, nor proof that allocation alone explains
the timing difference. FO decay gains must not be generalized to ordinary
QE calls. Full initial/final samples are in the QE library's
`docs/benchmarks/quantile_assignment_output_reuse_20261003.json`; reproduce
with `quant_evaluator/scripts/benchmark_quantile_assignment_reuse_oct03.py`.

FO decay A/B evidence above records the initial shared-kernel source hash.
The subsequent output-buffer change affects the standard value entry point;
FO uses the from-assignments entry point without an output buffer. The
reported FO ratios remain measurements of the recorded snapshot, not a new
timing certification for every later source revision.
