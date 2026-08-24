# R25 — Source Authority / Evidence Aggregation / Physical Isolation fix round

Local-only: DO NOT `git push`, `git commit`, `git add`, `git checkout/restore/stash/clean/reset/rebase`.
Working tree is truth. Edit files in the working tree. Evidence must be honest — no fabricated PASS.

## Verified current state (R25 findings, 2026-08-23)

- Repo HEAD = `ec4ef105f5613e0889575f4f362f38c86020d267` (main lineage).
  Evidence commit the user referenced, `2f383185`, is 4 commits back.
- `evidence/VerificationManifest.json` binds `source_snapshot.git_sha = 49b81f00` (stale vs HEAD),
  records `git_dirty: true`, and lists submodule SHAs for factor_engine/dataaccess.
- NO `.gitmodules` and NO git submodules in main. FE/DA/QE/FO/FP are plain tracked dirs in the monorepo.
- `quant_evaluator/` is real full source (`contracts/ metrics/ kernels/ runtime/ registry/ reporting/ ...`).
  There is NO `quant_evaluator/build/` dir on this server.
- Every QE source test (`quant_evaluator/tests/*.py`) has a "bootstrap" that force-loads the package from
  `quant_evaluator/build/lib/quant_evaluator` and asserts `__file__` startswith that path. With no build/lib,
  these tests FAIL AT COLLECTION. P0-1 is real and current.
- `factor_preprocess/` package root = `factor_preprocess/factor_preprocess/`. `factor_optimizer/` root =
  `factor_optimizer/factor_optimizer/`. Importable when the package dir is on `sys.path`.
- `dataaccess/` modules import `data_access.core...` but there is no `dataaccess/data_access/` dir and no
  packaging config in the tree; `import dataaccess` fails today. Not importable from source as-is.
- `factor_assets/` is NOT on disk, NOT in HEAD; source lives on branch `r2-q-empty-authority-gates` at
  `360836b5` (140 files). `.gitignore` line 54 ignores `factor_assets/`.
- Root `cleaned_operators/` (459 tracked files) and `factor_engine/cleaned_operators/` both exist and differ
  (blob sha of `__init__.py`: root `4e5abacc`, FE `8762e4c9`). `evidence/current.py:_module_source_digest`
  resolves `cleaned_operators` from the ROOT copy. The factor-engine wheel builds from the repo ROOT
  (`_build_backend` + `scripts/wheel_clean_install_smoke.py`), so runtime and hashing both point at root —
  the `factor_engine/cleaned_operators/` duplicate is the leftover that must be deauthorized/archived.
- `requirements-production.lock` = same in root and `factor_engine/`. It contains `==missing` placeholders
  and `python==3.10.12`.

## Canonical test runner on this server

    PY=/tmp/fe2/bin/python          # has numpy 2.5.2, pandas 3.0.5, scipy, polars, duckdb, statsmodels, PyWavelets, sqlglot, pytest 9.1.1
    export PYTHONPATH=/home/sunhaiwei/quant_projects:/home/sunhaiwei/quant_projects/factor_preprocess:/home/sunhaiwei/quant_projects/factor_optimizer:/home/sunhaiwei/quant_projects/dataaccess

Use `$PY -m pytest <file> -q -p no:cacheprovider --tb=short` from `/home/sunhaiwei/quant_projects`.
QE/FP/FO are importable from source with that PYTHONPATH. Do NOT run the slow gates
(concurrency ~104s, leakage ~82s, PIT ~81s) — run only targeted/fast tests.

## Work cluster boundaries (disjoint files — no cross-agent file overlap)

### Cluster A1 — gate runner + QE test bootstrap (P0-1, P0-3, P1-17)
Files: `scripts/gate_runner.py`, `scripts/refresh_gate_evidence.py`, and the 9 files under
`quant_evaluator/tests/` that contain the `build/lib` bootstrap:
  test_qe_p0_03_hash_stability.py, test_axis_refs_and_registry_seal.py, test_sealed_split_overlap.py,
  test_qe_artifact_hardening.py, test_consistency.py, test_numerical_oracle.py, test_qe_serialization.py,
  test_metamorphic.py, test_scale_gates.py
(rewrite ONLY the import-bootstrap block of those tests — do not change test logic).
Deliverables:
1. `GateSpec` gains `expected_test_inventory` and `allowed_skip_inventory`; unknown skips -> FAIL/BLOCKED;
   `CROSS_PACKAGE` defaults `fail_on_skip=True`; `UNIT` enforces "no unexpected skips/errors".
2. Remove `QE_ENV`/`FP_PREPROCESS_ENV` build/lib PYTHONPATH hacks entirely. Source tests import canonical source.
3. `command_template_hash` (hash of the argv BEFORE `{junitxml}` is substituted) recorded alongside
   `rendered_command` and `runtime_temp_paths`; identity hash never includes the temp junit path.
4. Rewrite the 9 QE test bootstraps to import `quant_evaluator` from the repo-root source (drop build/lib).

### Cluster A2 — manifest/matrix/gates (P0-2, P0-5, P0-6, P0-8-manifest, P0-18)
Files: `scripts/gen_verification_manifest.py`, `scripts/gate_matrix.py`, `config/gates.json`.
Deliverables:
1. `apply_verified_evidence`: a gate PASSes only when the OBSERVED command identities == EXPECTED identities
   from GateSpec (no missing / extra / duplicate). Skip inventory enforced (see Cluster A1 schema).
2. Formal `ReleaseIdentity` dataclass; forbid str/dict dual type. No `root.split(" ")[0]`.
3. `EVIDENCE_CURRENT` gate wired: runs `python -m evidence.current --check` (exit!=0 -> FAIL/BLOCKED);
   add to gates.json + gate_matrix required set.
4. gates.json + gate_matrix: add SOURCE_AUTHORITY, SUPPLY_CHAIN, SUBMODULE_REACHABILITY, FRESH_WHEEL_MATRIX,
   EVIDENCE_CURRENT as required checks. Matrix must not print "all green" unless every production-required
   gate PASSes AND evidence SHA == current HEAD AND not dirty.
5. Manifest: submodule_shas only where a real gitlink exists (none in main) — stop recording fabricated
   submodule SHAs. FactorAssets handled by Cluster E.

### Cluster B — QE artifact + metric registry (P1-14, P1-15)
Files: `quant_evaluator/contracts/metric_artifacts.py`, `quant_evaluator/metrics/catalog.py`,
plus NEW test files in `quant_evaluator/tests/`.
Deliverables:
1. Production MetricArtifact path requires a real FactorAxisRef (add a `production: bool = False` flag;
   when True factor_axis must be a FactorAxisRef — research default keeps None allowed).
2. `FrozenMapping.__hash__`: stop using salted builtin `hash(frozenset(...))`; use `stable_hash(stable_content_hex(...))`
   or make unhashable. `_freeze_value` fail-closed on unsupported mutable objects (no silent pass-through).
3. `ImmutableBufferRef`: real ownership transfer (the wrapped buffer must be exclusively owned/read-only; reject
   writable buffers that the caller still mutates after construction via a post-construction mutation check or copy-on-write).
4. `catalog.py`: replace mutable global `CATALOG` dict with BUILDING -> register -> duplicate-detection -> seal ->
   immutable registry. `MetricSpec` gains metric_version, implementation_id, artifact_kind, required_axes, units,
   direction, missing_policy, numeric_policy.
5. Do NOT touch existing test bootstrap blocks (Cluster A1 owns those).

### Cluster C — FP feature-manifest column mapping + FittedState (P0-12, P0-13)
Files: `factor_preprocess/factor_preprocess/contracts/feature_bundle.py`,
`factor_preprocess/factor_preprocess/contracts/state.py`, tests under
`factor_preprocess/tests/contracts/` (extend `test_tiling_channels_axes.py` and/or add new test file).
Deliverables:
1. `_feature_to_col` built from `channel_offset + local_feature_index` while iterating each feature channel,
   NOT `enumerate(feature_ids)`. Regression: layout missing=col0, raw:f1=col1, freshness=col2, raw:f2=col3
   -> `get_feature_values("f1") == values[...,1]`, `get_feature_values("f2") == values[...,3]`.
2. `FittedState.state_kind`: strict — only `StateKind` members, or a constructor that parses the string strictly;
   unknown/`"fitted"` string must resolve to `StateKind.FITTED` and satisfy the FITTED non-empty feature contract.
3. Remove `_stable_repr`'s arbitrary `repr(value)` fallback for unknown objects; reuse a strict canonical codec
   (like QE `_hashutil.canonicalize`) — unsupported objects raise instead of repr.

### Cluster D — FO real search-data isolation + production validator (P0-10, P0-11)
Files: `factor_optimizer/factor_optimizer/search/runner.py`,
`factor_optimizer/factor_optimizer/contracts/splits.py`, NEW capability module under
`factor_optimizer/factor_optimizer/`, new tests under `factor_optimizer/tests/search/`.
Deliverables:
1. Real `TrainDataCapability` / `ValidationDataCapability` / `TestDataCapability` classes. The search-process
   object graph must NOT contain TestDataCapability. `SealedTestExecutor` uses an independent authority
   (separate process or at minimum a strictly separate capability instance that the search graph cannot reach).
2. Malicious-evaluator tests: closure over full_y, global test array, reflection, pickle, checkpoint,
   adapter back-reference — none may reach the test payload.
3. `SearchRunner(PRODUCTION)`: `trial_validator is None` -> constructor FAIL. A legality result with
   `research_only=True` in PRODUCTION -> FAIL. Validator must carry `validator_id`, `validator_version`,
   `implementation_hash`.

### Cluster E — source authority (P0-7, P0-8, P0-9)
Files: `evidence/current.py` (if needed), `.gitignore`, `factor_assets/**` (restored), docs.
Deliverables:
1. cleaned_operators: single canonical source. Keep ROOT `cleaned_operators/` canonical (it is what the wheel
   builds and evidence hashes). De-authorize `factor_engine/cleaned_operators/` — either delete/archive it or
   make it non-importable, and make any identity/hash code unambiguous about the canonical path.
2. factor_assets: restore the 140 files from `git archive 360836b5 factor_assets | tar -x -C .` (read-only,
   no checkout/restore), remove `factor_assets/` from `.gitignore`, confirm it imports and passes its tests
   (`factor_assets/tests/`). This makes FA source-authoritative and pins it to an immutable SHA.
3. SUBMODULE_REACHABILITY: since main has no submodules, the gate must (a) report the truth (no gitlinks)
   and (b) the manifest must stop fabricating submodule_shas.

### Cluster F — full wheel matrix (P1-16)
Files: `scripts/fresh_wheel_matrix.sh` (or .py), plus any missing packaging config needed to build.
Deliverables:
1. Build all 6 wheels (factor_engine, quant_evaluator, factor_optimizer, factor_assets, factor_preprocess,
   dataaccess) from source; install ONLY from pinned lock + hashes into a clean venv; wheel-only
   cross-package integration (no repo root on sys.path).
2. HONEST reporting: any package that cannot build (e.g. dataaccess has no packaging config) is reported
   BLOCKED/NOT_RUN with the reason — never fabricated PASS. DataAccess packaging config may be added if small.
