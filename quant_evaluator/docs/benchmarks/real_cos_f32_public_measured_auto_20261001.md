# Real COS F32 measured-auto review (2026-10-01)

## Result

The persisted run reports a successful calibration for a `2586 × 5461 × 32`
batch (`float64`) and the four Pearson metrics: `pearson_ic`,
`pearson_ic_series`, `pearson_ic_std`, and `pearson_ic_ir`. Calibration selected
strict CUDA on the measured steady-state medians:

| Route or phase | Seconds | What the timer includes |
| --- | ---: | --- |
| CPU median | 27.057 | Two measured whole public-evaluation calls |
| Strict CUDA median | 16.208 | Two measured whole public-evaluation calls |
| Input fingerprint | 2.240 | Exact batch, label, and metric request fingerprint |
| First calibrated public call | 133.064 | Setup plus six whole route calls (one warmup pair and two measured pairs) |
| Calibrated cache-hit call | 18.531 | Fresh fingerprint/source check plus reevaluation on the cached route |
| Ordinary `backend="auto"` call | 16.172 | One ordinary routed evaluation; receipt confirms CUDA for all four metrics |

The two measured CPU/CUDA pairs alternate route order; the policy uses one
warmup and two repetitions. The median comparison is therefore evidence for
this exact materialized request and runtime. The ordinary-auto call is slightly
shorter than the CUDA median, but it is one call and does not establish a
general speed ranking. In particular, this run does not support a claim that
CUDA or auto is fastest for every metric, shape, input, or device state.

## Parity and provenance

Calibration records `parity: pass` with no mismatch. The calibration loop
checks the CPU and strict-CUDA bundles after each pair, including the warmup
pair. The saved `comparison_mismatches` are also null for the calibrated
result versus both the cache-hit result and ordinary auto. The comparator
covers factor IDs and available metric values, grouped metrics, diagnostics,
artifacts, factor artifacts, versions, instance results/specifications, and
warnings. Numeric arrays use the configured `rtol=1e-10`, `atol=1e-12`, and
must have matching finite masks; non-floating metadata is exact.

The evidence binds 32 distinct factor IDs to one manifest digest
(`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`). Each
source record includes its COS URI, object SHA-256, byte count, ETag, and
source status; all listed object digests occur in their URI. The batch covers
2016-01-04 through 2026-08-25 and records the label definition, finite ratios,
and no missing adjusted-VWAP partitions. These fields support identifying
the tested inputs, but do not certify upstream point-in-time or investability
lineage.

## Limits of the saved evidence

The comparison is a bounded, alternating run with only two measured
repetitions, not randomized repeated I/O or a universal performance study.
`cache_hit_seconds` includes request/source identity work as well as the
cached-route evaluation; `ordinary_auto_seconds` is a single ordinary
evaluation, so these wall times have different setup and sampling scopes.
`calibration_seconds` includes six whole route calls and should not be read as
steady-state latency. The source-check receipt says `strict_full_content`,
but reports `files_hashed` and `bytes_hashed` as null. The evidence also omits
an explicit hardware/software/runtime inventory and a source-control revision;
the cache key is opaque and is not a substitute for those details. The
process peak RSS is recorded, but it does not describe GPU memory use.

The helper's saved `pass` predicate checks calibration parity, cache status,
and result comparisons, but does not itself assert ordinary-auto routing
metadata. This particular receipt shows CUDA for every listed metric, with
reason `certified_batch_real_cos_f32_pearson_chain`. For a future
unknown-envelope case, add an explicit route requirement: assert each metric's
backend is GPU, its reason identifies measured selection, and no static shape
profile or certification was used. The helper's ordinary call as currently
written cannot validate that condition by itself.

There is also a comparator caveat for future runs: `_parity_mismatch` skips a
field when either bundle lacks that attribute. A null mismatch therefore
certifies the fields that were compared, not field-presence symmetry for every
possible bundle schema. This does not show a mismatch in this saved run, but
the current evidence does not inventory which optional fields were present.

The recorded `pass` supports this exact run's parity and cache-hit checks. It
does not certify that every auto route is fastest, nor establish production
readiness.
