# DLIB-XPKG cross-package integrity gate — report (2026-08-27)

## 1. Domain → wheel build → fresh-venv install/import → e2e smoke

| Domain | package | wheel build | fresh-venv import | e2e smoke |
|--------|---------|:-----------:|:-----------------:|:---------:|
| QE  | quant_evaluator     | OK (0.0.1a1) | PASS | PASS |
| FO  | factor_optimizer    | OK (0.1.0)   | PASS | PASS |
| FP  | factor_preprocess   | OK (0.1.0)   | PASS | PASS |
| FA  | factor_assets       | OK (0.1.0)   | PASS | PASS |
| P   | quant_platform      | OK (0.1.0)   | PASS | n/a (contracts boundary verified) |

Wheels (all built from the working tree, `python -m build --wheel`):
`/tmp/dlib_wheels/{quant_evaluator,factor_optimizer,factor_preprocess,factor_assets,quant_platform}-*.whl`

Fresh venv: `/tmp/dlib_venv/base` (created `--without-pip`, pip copied from project venv;
deps numpy/scipy/pandas/PyWavelets/PyYAML/statsmodels installed as binaries; statsmodels needs
--only-binary in this env). All 5 wheels installed with `pip install --no-deps`.

Import smoke: `/tmp/dlib_smoke/smoke_base.py` — 13 DLIB artifact symbols all PASS from
site-packages (verified `m.__file__` starts with `/tmp/dlib_venv/base/lib/...`; run from
neutral cwd `/tmp/dlib_run` so the repo root cannot shadow the wheels).

## 2. Cross-package reference integrity (PURE-DTO §6)

No module-level (top-level) cross-domain `import/from` anywhere in the four domains,
and no domain imports `quant_platform`, and `quant_platform/app/contracts` imports no
domain package (verified by grep + runtime import watcher).

Non-package-level cross-domain imports (guarded, optional-integration) — NOT a violation:
- `factor_optimizer/factor_optimizer/adapters/quant_evaluator.py:131-132` — inside
  `create_qe_adapter()` body; proven: `import factor_optimizer` works with QE import blocked.
- `factor_preprocess/factor_preprocess/adapters/factor_assets.py:192` — inside
  `check_factor_assets_available()` try/except; proven: top-level `import factor_preprocess`
  works with factor_assets blocked. FP's module-level imports are stdlib + numpy only.
- `factor_assets/adapters/quant_evaluator.py:14-16` and `residual_novelty.py:24` — inside
  `try: from quant_evaluator ... except (` guarded optional adapters; proven: module imports
  work with QE import blocked (they set `QE_AVAILABLE = False`).

## 3. wheel self-containment

- No tests/build/`__pycache__` leaked into any wheel (inspected via zipfile).
- factor_assets `canonical_data` check: no data-file entries required at runtime (the only
  data-extension match was a `.py` module path with 'data' substring); runtime `open()` reads
  are catalog/universe production paths via the data_access adapter, not ship-with-package.
- FO/FP nested layout preserved: `factor_optimizer/factor_optimizer/...` and
  `factor_preprocess/factor_preprocess/...` resolve correctly from site-packages.

## 4. Shared return convention + deep-freeze

vwap→vwap basis (`Vwap.pct_change().shift(-1)`) is the declared global hard convention:
- `quant_evaluator/contracts/label_bundle.py:18,35` — `price_convention = "vwap_to_vwap"`,
  forward-return measure vwap-to-vwap.
- `factor_assets/aggregation/composite.py` — HorizonMapping aggregates vwap-to-vwap forward
  returns; no close-based return in any of the four domains' non-test code (grep CLEAN).
- `quant_platform/app/contracts/model_version.py:38` — `return_basis: str = "vwap"` with the
  exact `Vwap.pct_change().shift(-1)` documented.

Deep-freeze verification (runtime, fresh venv):
- QE `EvaluationArtifact`  (frozen, recursive FrozenMapping) — hashable, deepcopy-safe. PASS
- FP `TreatmentRecipe`      (frozen, deep_freeze on step params) — hashable, deepcopy-safe. PASS
- FA `FactorLibraryVersionArtifact` — hashable, deepcopy-safe. PASS
- FA `ClusterSetVersionArtifact` / `ClusterVersionArtifact` — hashable, deepcopy-safe. PASS
- FA `ClusterArtifact`      — assignment map frozen to `MappingProxyType`: mutation-guarded
  (immutable), but `copy.deepcopy` raises `TypeError: cannot pickle 'mappingproxy'` and it is
  NOT hashable (its `assignments` maps keys type dict). This matches QE's same MappingProxyType
  under the hood, but QE wraps it in a hashable FrozenMapping; FA's view exposes the raw proxy.
- FA `SimilarityArtifact` — same as ClusterArtifact (`views` backed by MappingProxyType):
  mutation-guarded, `deepcopy` raises, not hashable.

Status: FA cluster-family artifacts are deep-immutable (mutation-blocked) and content-hash
addressed; deepcopy/hash limitation exists but nothing in the four domains' runtime path calls
deepcopy on them (FA tests exercise mutation-guard, not deepcopy). No failure in any required
cross-package runtime path. No violations requiring a domain-code change.

## 5. E2E smoke

`/tmp/dlib_smoke/e2e_smoke.py` — runs from the fresh venv (installed wheels only, neutral cwd):
1. FP: build trivial `TreatmentRecipe` (zscore step). PASS
2. QE: build `EvaluationArtifact` (derived content_hash). PASS
3. FO: `UncertaintyAwareWinnerSelector.select()` over 2 candidates → winner candidate_A. PASS
4. FA: `ClusterArtifact` (modularity, 1 cluster) + `FactorLibraryVersionArtifact` (content_hash). PASS
Result: `E2E_SMOKE_RESULT: PASS`

## 6. BLOCKED / issues
None requiring a fix. The FA deepcopy/hash limitation is recorded under §4 and does not block
any required runtime path.

## Evidence
- Progress log: `/tmp/dlib_xpkg_progress.log`
- Wheels: `/tmp/dlib_wheels/`
- Smoke scripts: `/tmp/dlib_smoke/{smoke_base.py,e2e_smoke.py}`
- Clean venv: `/tmp/dlib_venv/base`
