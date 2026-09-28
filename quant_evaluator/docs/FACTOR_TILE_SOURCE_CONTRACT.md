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
string does not prove object integrity. The contract does not download COS
objects, merge `EvaluationBundle` objects, or implement a public
`evaluate_source` API yet. In particular, running `evaluate()` independently
on each tile and concatenating the results is **not** a single batch evaluation:
request identity, shared labels, metric artifacts, resource accounting, and
provenance must be aggregated by one orchestrator before that claim is valid.

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
