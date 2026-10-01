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

## Cache-aware coverage auto

For the certified float64 shape `(2586, 5461, 32)` and the single `coverage`
metric, the static auto selector checks raw-array identities and the exact
configuration-stream cache before probing CUDA resources. A missing or unknown
state selects CPU with reason
`identity_hash_cache_not_fully_warm_real_cos_f32_coverage`. A resident state
allows the existing CUDA precision/device/free-memory guards to decide.

The selector uses non-promoting cache peeks. It does not scan a missing factor
payload to estimate a backend. It bounds metadata traversal and treats unsupported
metadata as cold. It checks `EvaluationRequest` fields in the same shared field
builder that computes the final config hash; another request configuration
cannot borrow a warm state from a different hash prefix.

An omitted backend and `backend="auto"` use this policy. You can select
`backend="cpu"` for the reference path or `backend="cuda_strict"` to request
CUDA explicitly. The existing measured-auto registry and explicit calibration
options remain available; cache state affects the static coverage route only.

Cache eviction or concurrent evaluations can change residency between selection
and execution. The selector makes a deterministic decision for the observed
state, not a guarantee of the fastest elapsed time under changing host load.
The F32 raw-cache A/B report records the cold/warm evidence behind this split.
