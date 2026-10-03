# Reusable quantile memberships and aggregation

QE now separates factor-only membership assignment from label aggregation.
This supports bounded reuse in TRAIN diagnostics (including FO layer decay)
without reimplementing QE quantile semantics outside this library.

## Formula and missingness

For source cross-section $x_s$, factor-valid finite values determine the
percentile boundaries and memberships $m_{s,n}\in\{-1,0,\ldots,Q-1\}$.
Tie policies retain the existing QE `min` / `max` percentile semantics.
Missing or invalid labels do not change boundaries or memberships.

For memberships aligned to scoring row $t$, define

$$
A_{t,q,f}=\{n:m_{t,n,f}=q,\ V^y_{t,n}=1,\ y_{t,n}\text{ is finite}\}.
$$

$$
C_{t,q,f}=|A_{t,q,f}|,\qquad
R_{t,q,f}=
\begin{cases}
\frac{\sum_{n\in A_{t,q,f}}y_{t,n}}{C_{t,q,f}},
& C_{t,q,f}\ge\mathrm{min\_assets},\\
\mathrm{NaN},&\text{otherwise}.
\end{cases}
$$

Counts report every eligible labelled member, even below the return threshold.
A missing membership is represented only by `-1`. For a lagged signal,
gather memberships from source row $t-\ell$ onto score row $t$ before calling
aggregation; pre-history rows use `-1` for every asset.

## Explicit typed API

```python
from quant_evaluator.contracts.quantile_assignments import QuantileAssignmentBatch
from quant_evaluator.metrics.quantile import (
    assign_quantiles_batch,
    compute_quantile_returns_from_assignments,
)

# values: (T,N,F); invalid factor cells must already be masked.
ids = assign_quantiles_batch(values, n_quantiles=5, method="max")
memberships = QuantileAssignmentBatch(
    assignments=ids,              # int32, including singleton F dimension
    time_axis=time_axis,          # explicit coordinates
    asset_axis=asset_axis,        # explicit coordinates
    factor_ids=factor_ids,
    n_quantiles=5,
    tie_policy="max",
)
returns, counts = compute_quantile_returns_from_assignments(
    memberships, label_bundle, min_assets=10,
)
```

The contract owns an immutable byte-backed copy without modifying caller
writeability. It rejects wrong dtypes/shapes, duplicate or empty factor IDs,
invalid quantile counts (including bool), and out-of-range IDs. Aggregation
requires exact score-time equality with label decision times and exact asset
coordinate equality. It does not reorder or infer axes.

The existing `compute_quantile_returns(FactorBatch, LabelBundle, ...)` API
remains supported. It uses the same aggregation kernel but writes directly
into its final output views, avoiding temporary per-factor result copies.
The output-buffer option is private kernel plumbing, not a new public
backend-selection setting. It does not change CPU/GPU `auto` qualifications.

Empty time or asset axes preserve output shapes instead of indexing empty
sort buffers. All-missing cross-sections preserve NaN returns and zero counts.

## Evidence and limits

Final independent oracle/output-buffer tests: 28 passing tests (0.49s).
The final focused quantile/FO diagnostic regression run: 206 passing tests
(41.08s). These overlap and are not unique-test totals; they do not establish
that every specialised metric or every backend input is covered.

[Controlled CPU A/B samples](benchmarks/quantile_assignment_output_reuse_20261003.json)
include both the initial shared-kernel slowdown (~2.1%) and the subsequent
output-buffer reuse result. On synthetic 512 days x 5461 assets x 16 factors,
final median was 3.784897s versus pinned old baseline 3.835428s. This small
difference is not evidence of a significant general evaluator speedup.

Reproduce with
`python quant_evaluator/scripts/benchmark_quantile_assignment_reuse_oct03.py`
from the full project root using its virtual environment. The runner loads
the pinned old optimized module into memory, copies no checkout/data, checks
every return and count, performs one warmup per arm and three ABBA rounds,
and limits requested work to 60 million factor cells. These are synthetic
CPU measurements, not COS/GPU or whole-evaluator end-to-end results.
