# COS tile source memory admission

`CosFactorTileSource` estimates simultaneously live logical arrays before starting workers.
For C = times × assets × min(max_tile_size, factor_count), value item size B,
and caller-declared scratch E bytes/cell, assembly estimate is C × (3B + 2 + E).
This covers retained source panels, mutable dense values, immutable values,
and mutable/immutable boolean validity masks. Bounded prefetch is added separately.
The default source budget remains 4 GiB; it is not automatically increased.

This estimate is not a hard process RSS limit. Python/pandas metadata, indexes,
allocator overhead, labels, evaluator intermediates and GPU staging have separate costs.
Callbacks with larger source panels than the requested axes, reindexing copies,
or other temporary allocations must account for additional scratch explicitly.
`extra_assembly_bytes_per_cell` is a nonnegative integer accepted by both constructors.

The 2586 × 5461 × 16 float64 case estimates 5,874,812,736 assembly bytes,
plus 536,870,912 bytes with prefetch: 6,411,683,648 bytes total.
A 4 GiB configuration therefore rejects this case before tile reads. Historical
benchmarks made before the corrected admission check do not prove it fits 4 GiB.

`prefetch_workers` accepts 1, 2 or 4 in both source constructors (default 2).
Enabled prefetch uses the same bound for active workers and pending payloads.
The prefetch estimate is workers × 256 MiB; four workers require an explicit
`max_prefetch_memory_bytes=1024**3` or larger, as well as sufficient source budget.
`prefetch="off"` remains serial and has no prefetch-memory reservation.
This is a configurable option, not evidence that four workers are always faster;
the default remains two until representative IO/memory A/B supports a change.
