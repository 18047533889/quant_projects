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

## Strict loader cohort bridge (2026-10-04)

`factor_assets.adapters.research_cohort_receipts` provides
`build_cohort_research_feature_receipts(batch, provenance, *, embeddings,
embedding_spec, embedding_model_version, max_slice_bytes=128 * 1024**2)`.
It returns a tuple of the existing single-factor research receipts in input
factor order. It performs no COS I/O, embedding inference or registry writes.

Pass the FactorBatch and provenance returned by
`factor_optimizer.examples.cos_batch_audit.load_cos_sample`, explicitly using
`coverage_policy="strict"`, and a mapping from every retained factor ID to
its supplied embedding. The bridge accepts only strict, zero-quarantine
cohorts: retained IDs and source rows must match the batch exactly in order.
The loader's default isolate mode is deliberately unsupported here because
its source list can retain rows for subsequently quarantined factors.
This adapter does not change the loader or silently discard source rows.

The manifest URI must end in `<lowercase SHA-256>/landing_manifest.json`,
and each row's manifest digest must agree. Each source row supplies `factor`,
`uri`, `sha256`, `manifest_sha256` and positive integer `downloaded_bytes`;
other genuine loader fields are allowed. Downloaded byte count is admission
metadata, not receipt content identity. Source file SHA and URI remain caller
declarations: consistency checks do not authenticate downloads. Loader-derived
finite-value masks are preserved exactly, not relabelled as authoritative
missing-reason evidence. Receipt status remains `caller_supplied`.

All source metadata, embedding IDs and bounded finite embedding vectors are
validated before constructing a child. Each child receives strided views of
one factor's values and optional validity; QE FactorBatch owns their immutable
C-order copy. Hashes are computed on that child, never on the whole cohort.
A child is released before constructing the next; outputs retain metadata
only. Missing validity stays absent. No preliminary contiguous whole-cube
copy is made by this adapter.

`max_slice_bytes` bounds the one-child values plus validity payload copies.
It does not bound total RSS, the caller-owned parent, coordinate hashing
scratch, Python metadata or the tuple of embedding-bearing receipts. Choose
it against actual memory headroom and account for those other allocations.
Each embedding is capped at 4096 dimensions; no universal cohort/RSS bound
is claimed.

Formal initial behavioral RED was one expected missing-API assertion failure
in 0.92 seconds. Initial implementation plus scalar suite passed 43 tests in
0.95 seconds; absent-mask and non-contiguous-parent coverage expanded that to
45 passing tests in 1.00 seconds. These are bounded offline correctness tests,
not real multiyear/full-universe throughput or GPU qualification.

Independent review exposed a padded-factor-ID admission gap. The corrected
focused fixture failed before the fix (one assertion failure in 0.89 seconds,
showing entry into child construction); global trimmed-ID validation fixes it.
The final scalar plus cohort suite passed 48 tests in 0.99 seconds. Held-out
tests also verify that an infinite embedding generator stops at 4097 items,
child freeze inputs share the corresponding parent slice memory, and weak
references to child arrays are dead after the metadata-only return. These
checks establish the inspected copy/lifetime boundary, not a total RSS limit.
