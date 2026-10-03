# Source-profile output identity (2026-10-04)

## Output identity contract

Same-backend ABBA qualification requires stable `values_sha256`, `finite_mask_sha256`, and `observation_counts_sha256`. The live-auto guard extends that requirement to the selected backend's actual outputs. Counts use little-endian signed 64-bit integers (`int64`). Value digests cover raw canonical C-order bytes: NaN payload bits and signed zero are not normalized away. Scalar outputs have shape `(F,)`; series outputs have shape `(T, F)`. The live guard binds the ordered factor-ID axis to source metadata, rather than merely checking its length.

CPU and CUDA may produce different value hashes. Qualification compares their outputs under the declared per-metric numerical tolerances and mask/count requirements; it does not require CPU and CUDA value digests to be identical. Auto-verification is instead bound to the output hashes of the qualification-selected backend.

## Reader and trust boundary

The new linear-shape reader requires selected-backend auto value-hash binding for both explicit-profile and default-auto verification receipts. Historical Pearson report parsing retains its prior structure and does not gain that requirement. Runtime hashing lives in `runtime/source_profile_output_identity.py`, shared with measurement conversion without importing benchmark scripts into runtime. On a mismatch, the public API sets `source_qualification_applied=False`, reports `qualified_profile_output_identity_deviated`, and discards the cached route. This revokes the performance qualification; it does not independently certify the returned numerical values or recompute them. Public-entry regressions cover changed values, counts, and ordered factor IDs, including cache removal.

A parsed report is unsigned caller-provided data: schema and digest checks do not authenticate its producer or establish provenance. Live use must rebuild the current context and qualify against it (`expected_context`); a report qualified against an earlier source closure must not be reused after that closure changes. The genuine producer-to-reader test exposed a missing `expected_context` argument; the producer now supplies it, and the combined producer/oracle/reader regression passed 59 tests (session 21304).

## Scope and status

Current joint regression passed 231 tests with two existing multiprocessing/fork deprecation warnings (session 74612). Scope includes hashing, live execution guards, public provider/default cache and tampering, qualification DTOs, reader/producer/oracle, source catalog, and progress-writer checks. Two opt-in real CUDA glue cases passed independently (session 99513); their intentionally stubbed receipts correctly lose qualification, so this is GPU numerical/routing evidence, not real-COS qualification. A small hashing probe confirmed identical raw digests for contiguous, transposed, and reversed/sliced arrays, with update chunks no larger than 524,288 bytes for float64 values and 65,536 bytes for masks (session 51566). These are scoped correctness and scratch-buffer checks, not whole-process memory or performance certification. The full F48 six-metric shape ABBA run remains outstanding; unrelated live CPU/GPU jobs must be accounted for before timing. No all-path bug-free, fastest, or production-readiness claim is made.
