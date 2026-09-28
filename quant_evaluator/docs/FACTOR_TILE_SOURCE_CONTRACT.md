# Bounded factor-tile source contract

A full-history A-share panel can have thousands of dates and stocks; copying every
factor into one (time, asset, factor) array can exhaust host RAM before evaluation
starts. The tile-source contract establishes the input boundary for a future
single-request evaluator that reads a bounded number of factors at a time.

The contract is in `quant_evaluator.contracts.factor_tile_source` and is also
exported by `quant_evaluator.contracts`:

- `FactorTileSource` declares ordered, unique factor IDs; full time and asset
  axes; the actual real numeric dtype; a nonempty source snapshot identity;
  a positive maximum tile width; `read_tile(start, end)`; and `close()`.
- `FactorTile` wraps one immutable `FactorBatch` for an exact half-open factor
  range and repeats the source snapshot identity.
- `capture_factor_tile_source` validates and freezes source metadata before
  reads. `read_validated_factor_tile` rejects gaps, reordering, wrong ranges,
  axis or dtype drift, oversized tiles, and changed snapshots. The source
  metadata is checked both before and after each read.
- `iter_validated_factor_tiles` yields contiguous ranges covering every
  declared factor exactly once. Its optional `max_tile_size` argument caps
  in-memory tile width below the source limit. The caller owns the source
  lifetime and must call `close()` in a `finally` block.

The snapshot identity is an opaque trust-root supplied by a source adapter. A
COS adapter must bind it to a verified manifest/object set; merely inventing a
string does not prove object integrity. The internal GPU executor now has
`run_source_tiled(source, label, metrics, max_tile_size=...)`: one device session
retains labels while it reads and uploads bounded factor tiles, preallocates one
columnar `BatchEvaluationBundle`, and retries smaller reads after CUDA OOM when
policy permits. The caller must validate label timing and metric admission
before using this low-level primitive. CUDA tests compare scalar, series,
vector and count results against the existing in-memory executor.

The contract does not download COS objects, merge public `EvaluationBundle`
objects, or implement a public `evaluate_source`/`auto` API yet. In particular,
running `evaluate()` independently on each tile and concatenating the results
is **not** a single batch evaluation: request identity, shared labels, metric
artifacts, resource accounting, and provenance must be aggregated by one
orchestrator before that claim is valid.

Minimal source usage:

```python
from quant_evaluator.contracts import iter_validated_factor_tiles

try:
    for tile in iter_validated_factor_tiles(source):
        assert tile.batch.factor_ids == source.factor_ids[tile.start:tile.end]
        # Feed the validated tile to a future single-request orchestrator.
finally:
    source.close()
```

The dedicated contract tests exercise complete coverage, immutable values,
invalid ranges, reordered IDs, axis/dtype/snapshot mismatch, and metadata drift.
They do not certify throughput or the correctness of a future batch evaluator.

## Bounded full-history smoke test (synthetic, not route certification)

On the server L20, an on-demand deterministic source was exercised at
2586 dates × 5461 assets × 8 factors for `rank_ic` and `rank_ic_series`.
Widths 2 and 1 covered all eight factors in four and eight reads respectively;
scalar, series and count arrays matched at absolute tolerance 1e-12. The
single-process RSS high-water mark was 2,047,292 KiB, with reported GPU peaks
2,991,313,408 and 1,615,029,760 bytes. Wall times of 3.229 and 2.737 seconds
include random factor generation, so they are **not** evidence that width 1 is
faster on real COS data or that public `auto` should select this path.
