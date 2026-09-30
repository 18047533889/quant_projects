# Joint block bootstrap plan-hash reuse

Date: 2026-10-01

## Measurement setup

All three recorded stages used the same CPU-only .venv and A/B input: a (1250, 5) array from default_rng(12).normal(0, 0.05, ...), five factor IDs, and one ResamplingPlan with 200 replicates, block length 10, seed 0, and one segment. Both functions ran once for warmup. The benchmark measured seven alternating pairs, switching which function ran first on each pair. Each timed function call included index generation, plan and sample identity hashing, sample computation, and artifact construction. Timed calls reused the fixed plan; they did not include plan construction.

The performance-guard fixture uses make_ic(1250, 5, 0.0, 11), while these A/B runs used seed 12. The shape and plan settings match, but the input values differ.

For each A/B stage, returned samples matched bit for bit using uint64 views. The metric ID, factor names, and full provenance also matched exactly. The plan-hash count below counts evaluations of ResamplingPlan.content_hash per function call. It excludes separate hashes for the time grid, draw matrix, and common mask.

## Results

Times are milliseconds. Each row records one implementation stage; do not read the rows as a single controlled end-to-end speedup estimate.

| Stage | Plan-hash evaluations per call | New samples (7 alternating calls) | Reference samples (7 alternating calls) | New median | Reference median | New / reference |
|---|---:|---|---|---:|---:|---:|
| Before reuse | 202 each | 472.046, 466.579, 468.273, 467.376, 466.775, 468.883, 462.603 | 471.007, 462.537, 465.015, 465.047, 465.109, 473.288, 447.387 | 467.376 | 465.047 | 1.005 |
| Property-level reuse | 3 each | 18.747, 18.679, 20.432, 18.572, 18.804, 18.612, 19.471 | 16.181, 16.148, 16.151, 16.222, 16.151, 16.116, 18.604 | 18.747 | 16.151 | 1.161 |
| Producer-level reuse | New: 1; reference: 3 | 13.784, 14.217, 13.871, 13.738, 13.468, 14.724, 14.624 | 17.732, 16.047, 16.071, 16.119, 16.195, 16.281, 15.644 | 13.871 | 16.119 | 0.861 |

The original replicate_ids property evaluated content_hash once for each of 200 IDs. Property-level reuse computes the digest once per property access and builds all IDs from that local value. It does not cache across accesses. At that stage, each bootstrap function still evaluated the plan hash three times: twice for its provenance fields and once through replicate_ids.

The public producer now computes one local plan digest and uses it for both provenance fields and the replicate IDs. The reference retains its prior structure, so it evaluates the plan hash three times per call after the property-level change. The final 0.861 ratio describes the current public producer against that same-version reference. It is a 14 percent relative advantage, not the total gain from the original 467 ms observation. The first row records the old 200-hash-per-property overhead as a separate historical stage.

## Verification

The targeted run passed 14 tests, including the joint_bootstrap performance guard, bitwise equivalence, known values, provenance, one digest per property access, and one digest in the public producer. The performance-guard threshold did not change.
