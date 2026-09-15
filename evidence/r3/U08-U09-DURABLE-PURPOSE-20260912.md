# U08/U09 durable purpose/artifact evidence — 2026-09-12

Scope: direct edits in server-c `/home/sunhaiwei/quant_projects` main working tree. Synthetic fixtures only; they grant no real DataAccess profile or production publication authority.

## Implemented contracts

- Existing v2 writer, manifest, per-factor receipt, run identity and final/abort receipt preserve the immutable execution purpose and canonical run-identity digest.
- `SUCCEEDED` remains a delivery result. Research and production-compute artifacts both retain `assurance=UNVERIFIED` and `publication_authorized=false`; neither is converted to an execution certificate.
- Parent checks run-identity purpose against the parent engine before factor iteration. Direct workers reject an artifact plan whose purpose differs from their rebuilt engine. Parent validates purpose/digest fields rather than trusting worker labels.
- Resume validates persisted artifact purpose and run-identity digest. A research run cannot resume as production. Older no-purpose artifacts remain explicitly `legacy_unspecified` / `LEGACY_UNSPECIFIED` and are never upgraded.
- `read_verified_factor_artifact` reuses the managed manifest/chunk hashes, schema/policy identity, decoded-memory budget, contiguous coverage, two-axis uniqueness and DQ checks. Its typed projection contains globally scoped artifact id, manifest digest, execution purpose and run-identity digest for downstream `FactorValueRef.metadata`.
- Managed reads require a real broker memory lease (an integer budget alone is rejected); that lease stays alive through ndarray/index physical owners and is released only after the returned buffers die. Manifest reads are bounded before allocation, duplicate chunk filenames are rejected, and receipt run/ordinal/factor/policy plus exact row/byte coverage are checked.
- Lazy cache copies release an acquired lease if copying fails and explicitly deep-copy their index, so cached/output values do not retain shared index buffers.

## Tests

```text
test_r3_artifact_purpose + receipt protocol + resume validation + direct recovery
57 passed, 11 warnings in 3.95s

same set plus full test_v6_bounded_pipeline (before final reader-projection-only patch)
78 passed, 52 warnings in 170.88s

final receipt/resume/direct recovery/full bounded pipeline after lease hardening
71 passed, 49 warnings in 163.73s

final artifact ownership plus R04 lazy focused selection
34 passed, 5 warnings in 46.07s

normal-path R04/R05 full files after registry recovery
54 passed, 2 warnings in 44.42s
```

Warnings were existing multiprocessing fork deprecations and unsupported-Polars classification notices. `git diff --check` passed.

## Honest limits

- No approved real business profile was available, so real DataAccess execution, durable readback and QE consumption remain `NOT_RUN`.
- QE files were not modified; the reader returns the typed projection needed by the separate QE integration task.
- U10 compile errors remain string/None protocol outcomes and AU33–AU36 were not claimed or completed here.
- No commit, push, deployment, dependency installation, new writer, branch or worktree was created.
