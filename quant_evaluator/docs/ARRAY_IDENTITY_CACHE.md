# Array identity cache

QE computes its authoritative array identity from the original dtype, original
shape and exact C-order payload. This identity is independent of a caller's
`FactorBatch.value_hash`. The public import from `runtime.evaluator` remains
available; `contracts.array_identity` owns the implementation.

The raw-payload cache has 32 entries and weak references. Arrays must meet the
shared cache's 8 MiB threshold, have a C-contiguous numeric layout and derive
from an immutable `bytes` owner. A readonly view of mutable storage does not
qualify. Shape, dtype, strides, byte count and hash prefix participate in lookup.
The cache stores SHA state, not another copy of the factor panel.

Mutable, strided, scalar and unsupported layouts retain the uncached path.
Object arrays retain the element codec; structured arrays fail closed.
Fork handling and synchronization use the shared `ArrayHashStateCache` contract.

A 64 MiB microbenchmark on 2026-10-01 measured approximately 0.037 seconds for
the initial raw scan and 11–59 microseconds for repeated eligible-array calls.
The historical and cached implementations returned the same SHA-256 digest.
These numbers measure the hash helper, not end-to-end evaluation. Use the real
COS batch benchmark to measure backend selection and whole-request elapsed time.
