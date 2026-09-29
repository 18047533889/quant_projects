# Bounded factor-tile source contract

A full-history A-share panel can have thousands of dates and stocks; copying every
factor into one (time, asset, factor) array can exhaust host RAM before evaluation
starts. The tile-source contract establishes the input boundary for the
bounded public core-metric API described in [Factor-source batch evaluation](FACTOR_SOURCE_BATCH_API.md).

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

The generic contract does not download COS objects or turn a columnar
`BatchEvaluationBundle` into the richer public `EvaluationBundle`. The bounded
`evaluate_factor_source_batch()` API supports its explicitly admitted
factor-separable source metrics; see the current list, options, auto route and
limitations in the linked guide.
Running `evaluate()` independently on each tile and concatenating the richer
bundles is **not** a single `EvaluationBundle`: request identity, diagnostics,
artifacts, resource accounting, and provenance need a separate orchestrator.

Minimal source usage:

```python
from quant_evaluator.contracts import iter_validated_factor_tiles

try:
    for tile in iter_validated_factor_tiles(source):
        assert tile.batch.factor_ids == source.factor_ids[tile.start:tile.end]
        # Inspect one validated range; the public batch API manages this loop itself.
finally:
    source.close()
```

The dedicated contract tests exercise complete coverage, immutable values,
invalid ranges, reordered IDs, axis/dtype/snapshot mismatch, and metadata drift.
They do not by themselves certify throughput or the richer EvaluationBundle path.

## Bound COS source and prefetch choice

`quant_evaluator.adapters.cos_factor_tile_source.CosFactorTileSource.from_data_access`
is a research COS-specific implementation of this source contract. The caller
supplies ordered factor IDs, full axes, dtype, a tile assembler, factories for
registered DataAccess read contexts, and a `BoundManifestHelpers` value. The
manifest factory creates one controller context; the factor factory creates a
fresh context with its own store/engine per read. Neither factory may invent a
bucket or bypass dataset authorization. QE contains no dependency on the
library that implements bound-manifest reads;
the integration layer injects that library's existing
`read_bound_manifest`, `read_bound_factor` and `verify_bound_manifest_unchanged`
callables. The adapter delegates binding and source validation to those helpers,
then compares selected object URI, SHA-256 and byte count and verifies the
manifest again on completion/close. No binding validation logic is duplicated
in QE.

For an integration that already uses factor_optimizer, bind its existing
helpers at the boundary:

```python
from factor_optimizer.research_manifest import (
    read_bound_factor, read_bound_manifest, verify_bound_manifest_unchanged,
)
from quant_evaluator.adapters.cos_factor_tile_source import (
    BoundManifestHelpers, CosFactorTileSource,
)

bound_helpers = BoundManifestHelpers(
    read_bound_manifest=read_bound_manifest,
    read_bound_factor=read_bound_factor,
    verify_bound_manifest_unchanged=verify_bound_manifest_unchanged,
)
```

Other integrations can inject equivalent helpers without installing or
importing factor_optimizer. The lower-level `CosFactorTileSource` constructor
also remains independent of both factor_optimizer and DataAccess.

The factory's `prefetch` option is `"auto"` (default), `"on"` or `"off"`.
For this COS adapter, `auto` and `on` both use at most two independent reads
in flight; `off` reads serially. Results are assembled and returned in source
order. This choice belongs to the source adapter, not to the generic
`backend="auto"` CPU/CUDA route. The public evaluation call remains:

```python
from quant_evaluator.adapters.cos_factor_tile_source import CosFactorTileSource
from quant_evaluator.api.factor_source import evaluate_factor_source_batch

source = CosFactorTileSource.from_data_access(
    bound_manifest_helpers=bound_helpers,
    factor_ids=selected_factor_ids,
    time_axis=time_axis,
    asset_axis=asset_axis,
    dtype="float64",
    make_tile=assemble_verified_tile,
    manifest_context_factory=registered_manifest_context,
    factor_context_factory=registered_factor_context,
    expected_manifest_sha256=known_manifest_sha256,
    prefetch="auto",  # use "off" for a serial diagnostic run
)
try:
    result = evaluate_factor_source_batch(
        source, labels, metrics=selected_metrics, backend="auto",
        max_tile_size=16,
    )
finally:
    source.close()
```

The context factories and tile assembler above are application-provided
callables, not built-in names. `max_source_memory_bytes` and
`max_prefetch_memory_bytes` bound the adapter's conservative source estimate;
they do not cap labels, result arrays, DataAccess cache or total process RSS.
Invalid identity, manifest drift, budget violation and worker failure fail
closed. See the real-COS full-request measurements in
[runtime capabilities](PUBLIC_RUNTIME_CAPABILITIES.md); they certify only the
measured shape/metrics and research source, not arbitrary COS datasets or PIT.

## Bounded full-history smoke test (synthetic, not route certification)

On the server L20, an on-demand deterministic source was exercised at
2586 dates × 5461 assets × 8 factors for `rank_ic` and `rank_ic_series`.
Widths 2 and 1 covered all eight factors in four and eight reads respectively;
scalar, series and count arrays matched at absolute tolerance 1e-12. The
single-process RSS high-water mark was 2,047,292 KiB, with reported GPU peaks
2,991,313,408 and 1,615,029,760 bytes. Wall times of 3.229 and 2.737 seconds
include random factor generation, so they are **not** evidence that width 1 is
faster on real COS data or that public `auto` should select this path.
