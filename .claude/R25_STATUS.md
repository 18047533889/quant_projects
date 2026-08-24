# R25 — status notes (coordinator)

## Verified ground truth on server (2026-08-23)
- HEAD = `ec4ef105f5613e0889575f4f362f38c86020d267`; evidence commit `2f383185` is 4 back.
- Manifest binds `49b81f00`, `git_dirty: true` → P0-4/P0-5.
- No `.gitmodules`, no gitlinks (`git ls-files --stage | grep -cE '^160000'` == 0) → P0-8 = "nothing pinned".
- QE source full at `quant_evaluator/`, but NO `build/lib`; all 9 QE tests bootstrap from build/lib and fail at collection → P0-1 current.
- `cleaned_operators` root = canonical (evidence hash + wheel runtime both point at root); `factor_engine/cleaned_operators/` = leftover duplicate (blob shas differ) → P0-7.
- `factor_assets/` absent on disk/HEAD; source at `360836b5` (branch r2-q-empty-authority-gates, 140 files); gitignore line 54 ignores it → P0-9.
- `dataaccess/` no packaging config; `import dataaccess` fails → P1-16 honesty issue.
- Test python = `/tmp/fe2/bin/python` (numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, polars, duckdb, statsmodels, PyWavelets, sqlglot, pytest 9.1.1).
- PYTHONPATH for tests: `/home/sunhaiwei/quant_projects:/home/sunhaiwei/quant_projects/factor_preprocess:/home/sunhaiwei/quant_projects/factor_optimizer:/home/sunhaiwei/quant_projects/dataaccess`

## Agents launched (all background)
- A1 (a61af0b1e432a69e9): gate_runner build/lib removal + QE bootstrap rewrite + skip inventory + command_template_hash
- A2 (aeadf48c32da5bed4): manifest/matrix/gates exact identity, ReleaseIdentity, EVIDENCE_CURRENT, required gates
- B  (af1c04269979c1260): QE artifact production factor_axis, FrozenMapping stable hash, MetricRegistry lifecycle
- C  (aded026aff35c0ff2): FP feature_to_col physical mapping + FittedState strict enum/canonical hash
- D  (a00f205591d6a2d41): FO data capabilities + production validator enforcement
- E  (a8ea8ca8879c4f265): cleaned_operators canonical + factor_assets restore + source authority report
- F  (a19a626243368a621): fresh wheel matrix script

## Known cross-agent conflicts to resolve after completion
- A1 rewrites quant_evaluator/tests/*.py bootstrap blocks; B adds NEW tests (doesn't touch those files). A1's allowed_skip_inventory schema must match what A2's manifest expects — I gave both the same field names.
- E restores factor_assets which F needs to build its wheel. F must wait for E (or report BLOCKED if absent).
- A2 owns config/gates.json + gate_matrix.py + gen_verification_manifest.py; F must NOT touch those.
