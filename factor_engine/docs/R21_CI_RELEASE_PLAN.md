# R21 CI / Release Plan

Status: **SKELETON** (P0-H) — the workflow, gate scripts, and evidence stub are
in place; the self-hosted runner and protected-main branch rules are the
deployment step that makes the gates *enforced*.

---

## 1. Stage list (mirrors `config/gates.json` + CI-only gates)

The release gate workflow `.github/workflows/release_gates.yml` runs **17
stages** in a single `required_gates` matrix job. Each stage runs
`venv/bin/python -m pytest <exact paths> --junitxml=...` with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` pinned.

| # | Stage | Test paths (exact) | Expected runtime (self-hosted) |
|---|-------|--------------------|--------------------------------|
| 01 | source_authority | `tests/source_authority integration_tests/test_cross_package_contracts.py` | ~1 min |
| 02 | unit | `tests/unit factor_engine/tests/r30/test_fe_da_batch_ir_2026_08.py tests/r30/test_fe_da_batch_ir_2026_08.py` | ~2 min |
| 03 | numerical_oracle | `quant_evaluator/tests/test_numerical_oracle.py` | ~1 min |
| 04 | property_metamorphic | `quant_evaluator/tests/test_metamorphic.py quant_evaluator/tests/test_consistency.py tests/r43/test_pass_manager_reachability.py tests/r43/test_model_evidence_gates.py` | ~2 min |
| 05 | leakage | `factor_preprocess/tests/contracts/test_leakage_properties.py tests/modeling/test_evaluation_leakage_evidence.py` | ~1 min |
| 06 | pit | `factor_engine/tests/pit dataaccess/tests/contract` | ~2 min |
| 07 | determinism | `factor_optimizer/tests/search/test_strategy_validation.py` | ~1 min |
| 08 | serialization | `quant_evaluator/tests/test_qe_serialization.py factor_assets/tests/registry/test_serialization_codec.py` | ~1 min |
| 09 | checkpoint_resume | `factor_engine/tests/runtime/test_r10_stateful_checkpoint_2026_08.py factor_engine/tests/operators/test_stateful_checkpoint_hardening.py factor_engine/tests/operators/test_stateful_checkpoint_contract.py factor_engine/tests/operators/test_r13_checkpoint_identity.py` | ~2 min |
| 10 | cross_package | `integration_tests/test_cross_package_contracts.py` | ~1 min |
| 11 | fresh_wheel | `bash scripts/ci_fresh_wheel_run.sh` (wheel build + clean venv + smoke + contract) | ~5 min |
| 12 | scale_1k | `quant_evaluator/tests/test_scale_gates.py` | ~1 min |
| 13 | scale_10k | `quant_evaluator/tests/test_scale_gates.py` | ~1 min |
| 14 | scale_100k | `quant_evaluator/tests/test_scale_gates.py` | ~2 min |
| 15 | real_ashare_shadow | `integration_tests/test_ashare_semantic_golden.py` | ~3 min |
| 16 | security_supply_chain | `bash scripts/security_audit.sh` (pip-audit + hash lock + secret scan + SAST-lite + SBOM) | ~2 min |
| 17 | evidence_current_check | `python -m evidence.current --write` + status assert | ~1 min |

The 14 named gates in `config/gates.json` map 1:1 to stages 01–10 + 12–15;
stages 11, 16, 17 are the CI-only gates (fresh-wheel, security/supply-chain,
evidence-current). `config/gates.json` remains the single source of truth for
gate names/default_status; the workflow never rewrites it.

## 2. `evidence_current_check` blocks merge

Stage 17 runs `python -m evidence.current --write --out evidence/CURRENT.json`
and then asserts the regenerated truth is **CURRENT** (evaluation.status ==
CURRENT, summary.stale == 0, summary.failed == 0, current == total). If the
evidence truth is STALE/UNRESOLVED the step exits non-zero and the job FAILS.
Because `release_sign_off` requires `required_gates` to have no failure, a
STALE evidence truth blocks the release sign-off. This is the P0 red-team
requirement: the repo does **not** rely only on committed evidence JSON — the
CI recomputes the SHA-bound truth from the live tree on every run.

## 3. Protected-main required checks

`main` must be a **protected branch** with the following required status checks
(configured in GitHub branch protection, not in the workflow file):

- `required_gates` (the 17-stage matrix job)
- `release_guards`
- `release_sign_off`

Until these are configured as required checks, the workflow runs but does not
block merges. The plan treats protected-main as a **deployment prerequisite**,
not a code artifact.

## 4. No-PASS-no-release rule

A release is only approved when **every** required gate is green. The
`release_sign_off` job is gated on `!contains(needs.required_gates.result,
'failure')` AND `needs.release_guards.result == 'success'`. There is no
merge-into-prod path with any stage red. A stage that is skipped (e.g. a
`continue-on-error` or a skip-to-green) is treated as a failure for sign-off
purposes — the matrix is the full branching gate set.

## 5. Fresh-wheel-mandatory rule

Stage 11 (`fresh_wheel`) is mandatory. It builds wheels for the six workspace
packages (factor_engine, quant_evaluator, factor_optimizer, factor_assets,
factor_preprocess, dataaccess) into a temp dir, creates a **clean** venv,
installs ONLY the built wheels + the exact locked production deps, and runs
import/smoke + contract tests **without any repo-root on sys.path**. A package
that only works via an editable/`sys.path` hack fails this gate. The local
runner is `scripts/ci_fresh_wheel_run.sh`.

## 6. Dirty-source-no-production-release rule

`release_guards` runs `git status --porcelain` and fails the job if the working
tree is dirty (only CI artifacts may differ). It also prints the pinned
submodule SHAs for `factor_engine` and `dataaccess`. A production release must
come from a clean, pinned tree — never from a dirty working tree.

---

## Deployment prerequisites (not code)

1. Self-hosted runner with the repo checked out and a working `.venv`
   (Python 3.10) at the repo root, with all six workspace packages importable.
2. Protected `main` with the three required checks above.
3. `pip-audit`, `bandit`, `cyclonedx-python-lib` installed in the `.venv`
   (the security script falls back to `pip check` / `py_compile` if absent).

## Security / supply-chain tooling (P1)

- `scripts/security_audit.sh` — pip-audit (or `pip check`), dependency hash
  lock (`pip freeze` → `security/requirements-lock-hashes.txt`), secret scan
  (tracked files; the `ghp_` token in `.claude/skills/github-push/SKILL.md` is
  a **known documented item**), SAST-lite (bandit, else `py_compile` +
  subprocess/eval/import scan), and a CycloneDX 1.5 SBOM stub
  (`security/cyclonedx_packages.json`).
- `security/cyclonedx_packages.json` — **STUB**: components from `pip list`,
  licenses `unknown`, no dependency graph. Real SBOM generation (license
  resolution + dependency edges) is a follow-up.
- `security/requirements-lock-hashes.txt` — hash-pinned freeze (SDIST-anchor
  sha256 of each `name==version` line). This is a **starter** lock; a
  production lock with real artifact hashes is a follow-up.
