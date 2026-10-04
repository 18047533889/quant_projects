# Research feature receipt: scope and identity

Status: research bridge under behavioral verification; not production intake integration.
Formal source: server-c `/home/sunhaiwei/quant_projects`.

## Purpose and ownership

`factor_assets.contracts.research_feature_receipt` owns the immutable, payload-free
research receipt and source declaration. The narrow adapter
`factor_assets.adapters.research_feature_receipt.build_research_feature_receipt`
binds an explicit caller embedding to exactly one QE FactorBatch.

This is a research identity boundary, not a production similarity fingerprint.
No profile, preprocessing policy, governed snapshot, universe, fitted state,
authoritative missing-reason plane, or point-in-time investability is inferred.
The receipt never assigns a cluster, persists a factor, downloads data or
publishes a production factor. No existing package entry point is changed.

## Identity and calculation

The source binding contains factor ID, manifest URI/SHA-256, source URI/SHA-256,
and expected QE authoritative hashes for the full one-factor values
(T × N × 1), exact validity array or explicit absence, and time/asset coordinates.
The adapter recomputes and compares the array hashes before returning a receipt.

For non-object arrays, QE authoritative identity is SHA-256 over dtype,
shape and exact C-order payload bytes. Object coordinates use QE's existing
lossless scalar codec, not raw Python object-pointer bytes. Unsupported schemas
fail closed; supported Decimal coordinates are not rejected merely for being
objects. Values are not cast or cleaned: NaN/Inf and validity remain exact
identity information. Absent validity is different from an all-valid mask.

Receipt content identity uses FactorAssets' typed canonical structural encoder
and includes every declared source field, actual value/validity dtype, shape,
axis names and actual coordinate dtype, embedding, embedding specification,
model version, schema version and research-only scope. Reordered allocation
of equal arrays does not change identity; changes in array contents, dtype,
axis order, masks or embedding configuration are not silently aliased.

## Trust boundary

`source_binding_status = caller_supplied` is intentional. Matching in-memory
array hashes proves that the caller declaration refers to the supplied slice;
it does not prove that a COS object was downloaded, that its file bytes hash
matches, or that a manifest record authenticates the source. URI/file-hash
changes are recorded in content identity rather than falsely authenticated
against matrix bytes. A future loader integration must supply genuine bound
source evidence; it must not relabel a manifest SHA as a data snapshot.

The embedding is supplied by the caller, not trained or computed by this
adapter. It must be finite, nonempty and bounded to 4096 dimensions. Its
specification and model version are explicit. A receipt is not evidence that
the embedding is economically useful or that its similarity metric agrees
with legacy intake Spearman clustering.

## Usage

Import the two new modules directly; no package-global registration is needed.
Construct ResearchSourceBinding from an actual, single-factor source slice and
the genuine loader's source metadata, retaining the caller-supplied status.
Call the builder with the same QE FactorBatch and explicit embedding,
embedding_spec and embedding_model_version. Use the returned content_hash as
a research cache identity, never factor_id alone.

Do not build an all-valid mask from NaN checks and call it authoritative DA
missingness. Do not pass this DTO to production fingerprint, cluster-assignment
or registry-write APIs. Do not pad 16 real factors into thousands of claimed
real members for a benchmark.

## Verification and remaining integration

Offline tests exercise genuine QE FactorBatch and authoritative hashing with
bounded arrays; no COS or registry access is needed. Tests must cover altered
factor/axis/value/mask binding, absent validity, equal independent allocations,
supplied metadata changes, embedding boundaries and unsupported encodings.

The initial public-behavior test was observed failing on the formal worktree
before the builder existed (one assertion failure, not collection failure).
Implementation and regression results must be recorded separately before a
release claim. Real loader integration, persisted receipt admission, actual
batch-versus-scalar real-factor scan and randomized paired timing are still
required. This module alone is not an end-to-end performance result.

## Recorded bounded verification (2026-10-04)

The identical initial behavioral test passed after implementation (0.99 seconds).
The expanded receipt suite passed 25 tests in 1.00 seconds, covering scalar
embedding normalization, metadata identity, supported Decimal coordinates,
unsupported structured coordinates, exact NaN/Inf/mask identity, and rejection
by the production fingerprint bridge. Extra boundary tests were added after
initial GREEN and are not claimed as separate fail-first cycles.

A broader 114-test adapter/scan/extreme-value run returned 106 passed, 8 failed
and 3 warnings in 38.27 seconds. All eight failures reached the existing FE
bootstrap owner-identity guard for intra_same_slot_zscore; the new receipt tests
passed in that same run. A separate cold-process single-test run reproduced
the failure (33.60 seconds). This is unresolved integration evidence, not a
green full-library result. No FE source or guard was changed by this tranche.
