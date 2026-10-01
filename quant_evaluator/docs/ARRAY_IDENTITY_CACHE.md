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

## Coverage auto and identity reuse

For the certified float64 shape `(2586, 5461, 32)` and the single `coverage`
metric, the static auto selector admits CUDA through the existing precision,
device and free-memory guards, including on the first evaluation. Identity
cache residency no longer selects the backend. QE still computes and validates
the same authoritative array and configuration hashes for CPU and CUDA.

`EvaluationRequest` fields remain part of the shared configuration identity.
Eligible immutable arrays can reuse hash states; mutable or unsupported arrays
retain their existing hashing path. Cache eviction changes hashing cost without
changing this static route.

An omitted backend and `backend="auto"` use this policy. You can select
`backend="cpu"` for the reference path or `backend="cuda_strict"` to request
CUDA explicitly. The existing measured-auto registry and explicit calibration
options remain available. Resource rejection still selects CPU under auto.

We verified the new route with six rounds and 18 complete evaluations on the
real COS panel. See [cold-auto verification](benchmarks/real_cos_f32_coverage_cold_auto_cuda_20261001.md).
The certificate covers this exact request and tested resource conditions.
Other shapes, metrics, parameters and devices require their own evidence.
