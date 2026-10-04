# Research receipt scan audit protocol

Status: implemented and bounded offline verification passed. No real
cohort timing, registered-factor intake, production clustering or GPU result is
claimed by this document.

## Scope

The independent Python callable
`factor_assets.scripts.audit_research_receipt_scan_oct04.run_receipt_scan_audit`
compares the existing exact single-query and batched member scan helpers. It
accepts caller-supplied research receipts; it does not download data, infer an
embedding, assign a cluster, write a registry or publish a factor.

These are embedding-cosine scans, not raw-value Spearman correlation. A named
embedding specification and model version declare comparability but do not
authenticate model execution or prove economic usefulness.

## Input and pairing

The outer container must be an exact built-in tuple or list. Container subclasses
are rejected before iteration: overridden length/iteration could otherwise bypass
the cohort admission cap or materialize an unbounded input.
Supply 2–16 exact ResearchFeatureReceipt objects with unique factor IDs, finite
nonzero equal-width embeddings, identical specification/model version,
manifest URI/digest, time/asset coordinate hashes and array shape. Status stays
`research_only` and `caller_supplied`. Each receipt's canonical identity must
match its contents.

Input order is fixed: the first ceiling-half are members, the remaining
receipts are queries. The sets are disjoint, preventing trivial self matches.
Both arms receive the same ordered members and normalized queries. Exact ties
select the first member in input order. No full query-by-member score matrix
is constructed.

The callable runs two scalar/batch warmup pairs, then 3–10 seeded randomized paired
blocks. Every pair must agree on winner IDs and Float64 scores; disagreement
raises instead of producing a successful report. Source and receipt identities
are checked around calls. Source hashing is explicitly scoped to inspected
audit/scan/normalization/receipt modules, not a transitive dependency closure.
Timings describe this input and environment only; they do not select a backend
or establish universal fastest behavior.

Each reported time includes the source/receipt identity guards around the scan;
it excludes source loading, embedding production and initial query normalization.
It is not a bare cosine-kernel timing or an end-to-end intake measurement.

The report records input-ordered receipt content hashes, the shared
manifest/coordinate/shape/embedding identity, random seed, pair count, warmup
pair count, scan limits and vector width. The content hashes bind complete
receipt declarations; they do not authenticate source files or model execution.
Call the function with an existing tuple of receipts, for example:

```python
from factor_assets.scripts.audit_research_receipt_scan_oct04 import run_receipt_scan_audit

report = run_receipt_scan_audit(
    receipts, paired_blocks=3, seed=20261004,
    max_chunk_rows=4, max_chunk_bytes=1024 * 1024,
    max_batch_queries=16,
)
```

This callable has no automatic loader or receipt-file reader. Supplying
receipts remains a separate, explicit caller action.

## Bounded verification

The formal initial behavioral test failed for the expected missing callable
(one assertion failure, 0.84 seconds). The identical test passed after
implementation (1.21 seconds). The expanded protocol, existing batched member
scan and scalar/cohort receipt suites passed 89 tests in 2.04 seconds.

Coverage includes literal winner/tie order, disjoint query/member IDs, mixed
specification/model/manifest/axes rejection, invalid option limits, forged
receipt identity, scalar/batch disagreement and source-binding drift during
scans. Additional boundary tests were added after initial RED; they are not
each claimed as separate recorded fail-first cycles. Offline fixtures are not
real COS factors and these passing tests do not establish production readiness.

Independent review exposed missing report reproducibility fields. The focused
report test failed on the absent receipt hashes (1.26 seconds) and the identical
test passed after adding the fields (1.25 seconds). The final expanded protocol,
member-scan, extreme-vector, sample-size and receipt suites passed 117 tests in
1.93 seconds. No real-data throughput or total-memory measurement was run.

A subsequent container-admission review found that tuple/list subclasses could
lie about length or supply custom iterators. Two bounded rejection tests first
failed (1.42 seconds), then passed with the identical command (1.13 seconds).
The complete protocol suite passed 30 tests in 1.24 seconds after the fix.

## Memory limits

Receipts cap embeddings at 4096 dimensions. The audit caps its cohort at 16,
query count at 16, member chunks at 256 rows and core chunk budget at 4 MiB.
The underlying helper reserves at least one row even when the requested byte
budget is smaller than that row's workspace. Query arrays, caller-held receipt
tuples, transient member-vector conversion, Python metadata and result lists
are separate allocations. These limits are not a total RSS guarantee.

## Actual sample locator and unresolved registered intake

The existing FactorOptimizer `examples/cos_batch_audit.py::load_cos_sample`
reads an explicitly bound candidate-pool manifest through DataAccess. The
inspected manifest is:

`cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/metadata/00e545d254ca742305a37405a69ffe55e9a04448d0f176282c4f07997f1feb66/landing_manifest.json`

Factors live under the loader's declared
`cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/factor_values` root. Do not
derive new bucket names, credentials or unsupported object paths.

The prepared optimizer audit reloads a strict 16-factor, 500-day, 256-asset
cohort through that loader; no saved prepared matrix is referenced by that
runner. Those candidate-pool statuses are not FactorAssets REGISTERED lifecycle
proof. Real registered intake still requires an authoritative registration or
snapshot identity joined to the exact DataAccess/FactorEngine value sources.
This audit does not close that requirement.

Real sample execution remains gated on the coordinator's timing/resource slot,
actual available RAM/disk checks and existing configured DataAccess targets.
No real-data result may be fabricated by padding or duplicating these factors.
