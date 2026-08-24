# R21 Source Residency Audit (FE-P0-01 / FE-P0-02)

- Date: 2026-08-22
- Scope: enumerate every importable copy of FactorEngine source; delete only
  what is provably safe; remove build/__pycache__/bytecode from git index.
- Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  /home/shw/quant_projects/.venv/bin/python -m pytest tests/source_authority/
  factor_engine/tests/test_source_residency.py -q --tb=short`

## 1. Duplicate-tree inventory

| Copy | Path | What it is | Status |
|------|------|-----------|--------|
| CANONICAL (authority) | `factor_engine/backend`, `.../planner`, `.../cleaned_operators`, `.../runtime`, `.../planning`, `.../mining`, `.../ir`, `.../expr`, `.../fields`, `.../market`, `.../modeling`, `.../semantic`, `.../storage`, `.../util`, `.../validation`, `.../api`, `.../audit`, `.../cache`, `.../service`, `.../security`, `.../telemetry`, `.../research_operators`, `.../research_tools`, `.../factor_recipes`, `.../export` | Git submodule `github.com/HKUST-QUANT-SOCIETY/factor_engine`, HEAD `fc3e5f3`, all importable | AUTHORITY |
| ROOT duplicate | `/home/shw/quant_projects/backend`, `.../planner`, `.../cleaned_operators`, `.../runtime`, `.../planning`, `.../mining`, `.../ir`, `.../expr`, `.../fields`, `.../market`, `.../modeling`, `.../semantic`, `.../storage`, `.../util`, `.../validation`, `.../api`, `.../audit`, `.../cache`, `.../service`, `.../security`, `.../telemetry`, `.../research_operators`, `.../research_tools`, `.../factor_recipes`, `.../export` | Historically the FE source lived at repo ROOT; after the submodule was introduced the old ROOT copies were kept. Every directory has `__init__.py` and is importable from the repo ROOT. | STALE / importable |
| Build artifact | `factor_engine/build/lib/*` (backend, planner, cleaned_operators, runtime, planning, mining, ir, expr, fields, market, modeling, semantic, storage, util, validation, api, ...) | `pip wheel` output; importable when `factor_engine/build` is on `sys.path`. `factor_engine/build/lib/backend/__init__.py` etc. exist on disk. | STALE / importable on path |
| Build artifact | `factor_engine/build/mining/*.json` (4 manifests) | Build outputs, not importable source. | STALE |

### Staleness of the ROOT copies

The ROOT trees were **actively re-committed in this repo** (they are NOT pure
git-submodule mirrors of `factor_engine/` — they are separate tracked trees).
Last commit dates:

- `backend` 2026-08-22 (R22 numba-region work) — but the working-tree file
  `backend/evidence_provenance.py` is a hand-edited superset (R23 P0-10
  numeric-policy digest) that is **NOT** in `factor_engine/backend/`. This is a
  deliberate fix staged at ROOT only; the submodule canonical copy was NOT
  updated (its `evidence_provenance.py` still predates the digest change).
- `runtime` 2026-08-22, `mining` 2026-08-22, `cleaned_operators` 2026-08-22
  (the last commit touching the ROOT `cleaned_operators/` tree is the pointer
  sync `267d4b56`, i.e. only the superproject pointer; content is older).
- `planner` 2026-08-21, `planning` 2026-08-21, `api` 2026-08-21.
- `ir/expr/fields/modeling/semantic/storage/util/validation/service/security/
  telemetry/research_operators/research_tools/factor_recipes/export/audit/cache`
  all last touched 2026-08-20 (one bulk commit `91189335`).

The ROOT copies are **not byte-identical mirrors** of the submodule: file sizes
differ (`backend/evidence_provenance.py` 36,834 B vs submodule 26,807 B), and
the R23 P0-10 evidence work exists only at ROOT. They are **stale historical
copies** that shadow the canonical tree whenever the repo ROOT precedes
`factor_engine/` on `sys.path`.

### Who imports what (grep across tests/ + scripts/ + integration_tests/ + factor_engine/tests/)

- 638 `.py` files in `tests/` (256), `factor_engine/tests/` (253), `scripts/`
  and `integration_tests/` use top-level `import backend` / `from backend...` /
  `planner` / `cleaned_operators` / `runtime` / `mining` / etc.
- **No test or script imports these names by their ROOT path** — every importer
  uses the bare top-level name, which the conftest resolves to
  `factor_engine/` (tests/conftest.py inserts `factor_engine/` ahead of the
  ROOT). Verified: the source_authority suite + backend/planner/runtime tests
  resolve `backend.__file__` under `/home/shw/quant_projects/factor_engine/`.
- Because every importer uses the bare top-level name, **physically deleting
  the ROOT trees is unsafe today**: bare `import backend` under pytest would
  then resolve to the submodule copy (good) but the ROOT trees contain
  unique, committed working-tree changes (e.g. R23 P0-10
  `numeric_policy_digest` at ROOT only). Deleting them would LOSE those
  changes. Verified: `tests/backend/` collection currently errors with
  `NameError: numeric_policy_digest` under the submodule copy — the ROOT copy
  is still the one the R23 evidence work targets.

## 2. What was REMOVED vs LEFT+FLAGGED

### Removed (safe, index-only — working dir untouched)

- `git rm -r --cached` on **all** `factor_engine/build/**` (1217 files:
  `build/lib/*` 1209, `build/mining/*` 4, quoted-CJK-doc names 4) — build
  artifact, regenerated by `pip wheel`, confirmed NOT imported by any test
  (no test puts `factor_engine/build` on `sys.path`; the residency gate
  isolates it away).
- `git rm -r --cached` on **all** tracked `__pycache__/`, `*.pyc`, `*.nbc`,
  `*.nbi` under `factor_engine/` (2017 pycache/pyc entries + 15 nbc + 15 nbi =
  ~2047 more files; total staged deletions in the submodule index now 3234).
- `factor_engine/.gitignore` already has the standard set (commit was staged
  by a prior writer): `build/`, `__pycache__/`, `*.pyc`, `*.nbc`, `*.nbi` +
  pre-existing `dist/`, `*.whl`. No further edit needed.

Working directory state of `factor_engine/` is **untouched**: `build/lib`,
`build/mining` still exist on disk; all `.py` source files are intact (staged
deletions that are `.py` are 1023, all under `build/`; **zero** `.py` staged
for deletion outside `build/`).

### LEFT + FLAGGED (do NOT delete — documented here)

1. **ROOT duplicate trees** (`/home/shw/quant_projects/{backend,planner,...}`).
   They are importable (all have `__init__.py`), and there is no tracked test
   that imports them *by root path* — but they carry unique committed changes
   not present in the submodule (R23 P0-10 `numeric_policy_digest`; staged
   superproject edits to `backend/evidence_provenance.py`). Deleting them
   would either break `tests/backend/` collection (they currently resolve to
   the submodule copy under conftest, which lacks `numeric_policy_digest` →
   `NameError`) or lose the working-tree changes. **Flagged: physically delete
   only after the R23 P0-10 evidence work is moved INTO the submodule.**
2. **`factor_engine/build/lib/*`** on disk. Removed from git index; kept in
   the working dir because a full `pip wheel` regenerates it. It is made
   non-importable-in-preference: no test puts `factor_engine/build` on
   `sys.path`, and the residency gate asserts a build/lib-only probe resolves
   OUTSIDE build/lib (it currently resolves to the ROOT `backend`), i.e.
   build/lib is not load-bearing under any path this suite uses. Regenerated
   by `pip wheel factor_engine/`.
3. **`backend/evidence_provenance.py`** — DO NOT delete; it is the staged
   R23 P0-10 target.

## 3. FE-P1-01 git-index hygiene — result

In `factor_engine/` submodule (index only; working dir intact):

- staged deletions: **3234** files
  - `build/`: 1217
  - `__pycache__/` + `*.pyc`: 2017 (includes 2 `M` pyc that a prior writer's
    `git rm` converted to staged `D`)
  - `*.nbc`/`*.nbi`: 30
- post-state tracked counts: `__pycache__/` 0, `*.pyc` 0, `*.nbc` 0, `*.nbi` 0,
  `build/**` 0
- `.gitignore` (submodule): already contains `build/`, `__pycache__/`, `*.pyc`,
  `*.nbc`, `*.nbi`, `dist/`, `*.whl` — committed by a prior writer. Confirmed.
- remaining non-ignored submodule status: `M .gitignore` (the committed
  ignore-block), `M runtime/multibackend/batch_transfer_optimizer.py`,
  `M runtime/multibackend/cse_cache_optimizer.py`, and 5 untracked new files
  (numba_region, tests) — all pre-existing FE work, not part of this task.

Superproject index: the gitlink `factor_engine` pointer and the ROOT duplicate
trees are UNCHANGED (no `git add`/`git rm` on them). The pre-existing superproject
modifications (staged `workspace_data/...sqlite` deletions, edits to
`backend/evidence_provenance.py`, `evidence/*`, `loop/care.md`,
`scripts/gen_verification_manifest.py`, the three source_authority test files,
`integration_tests/test_temporal_separation.py` etc.) are untouched.

## 4. FE-P0-02 source-residency gate — result

New test `factor_engine/tests/test_source_residency.py` (32 tests, all PASS):

- `test_canonical_residency_isolated[15 pkgs]` — a fresh subprocess
  (`python -I`, clean `PYTHONPATH`, no cwd on path; venv site-packages kept)
  inserts the canonical `factor_engine/` dir + repo ROOT (the ROOT is required
  for the `factor_engine` namespace package — `factor_engine/` has no
  `__init__.py`) and imports `factor_engine.<pkg>`; asserts `__file__` is under
  the canonical `factor_engine/<pkg>/__init__.py`.
- `test_no_root_resolution_in_isolated_probe[15 pkgs]` — the same isolated
  subprocess with ONLY the repo ROOT on `sys.path` imports the TOP-LEVEL name
  and asserts it does NOT resolve under the canonical tree (documents why
  sys.path-shadowing is load-bearing).
- `test_no_stale_build_lib_tree` — asserts `factor_engine/build/lib/*` is not
  importable under a build/lib-only probe (the probe resolves to the ROOT
  `backend`, i.e. build/lib is NOT load-bearing for this suite).
- `test_build_not_on_sys_path_under_pytest` — host pytest process must not
  have `factor_engine/build` on `sys.path`.

## 5. Source-authority suite + residency gate run

```
cd /home/shw/quant_projects
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/source_authority/ \
  factor_engine/tests/test_source_residency.py -q --tb=short
```

RUN NOTE: the monorepo defines TWO distinct `tests` packages (repo-root
`tests/` with `__init__.py`, and the submodule `factor_engine/tests/` with its
own `__init__.py`).  A single pytest command naming BOTH trees raises
`ImportPathMismatchError('tests.conftest', ...)` at collection because pytest
cannot hold two `tests.conftest` modules with the same module name.  This is a
pre-existing structural clash of the monorepo layout, not a regression of this
task.  The required gate is therefore executed as TWO commands (each tree
alone):

1. Residency gate (submodule tree):
   ```
   cd /home/shw/quant_projects/factor_engine
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     /home/shw/quant_projects/.venv/bin/python -m pytest \
     tests/test_source_residency.py -q --tb=short
   # 32 passed
   ```
2. Source-authority suite (repo-root tree):
   ```
   cd /home/shw/quant_projects
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     .venv/bin/python -m pytest tests/source_authority/ -q --tb=short
   ```

Result (run from `/home/shw/quant_projects`):

- `factor_engine/tests/test_source_residency.py`: **32 passed**
- `tests/source_authority/test_import_resolution.py`: 15 params + 6 core = **21 passed**
- `tests/source_authority/test_import_precedence.py`: **3 passed**
- `tests/source_authority/test_no_root_fe_packages.py`: **PRE-EXISTING FAILURES**
  (25 fail) — every `test_no_root_fe_package[...]` raises because the ROOT
  duplicate trees still exist. These are the P0 finding being tracked; they
  cannot pass until the ROOT trees are deleted (see flag above). The final
  param of the module (`test_root_pyproject_not_factor_engine`) passes (ROOT
  `pyproject.toml` declares `name = "quant-projects"`).
- `test_no_root_fe_packages` uses pytest rootdir `/home/shw/quant_projects/.../tests/source_authority`
  in isolation and `/home/shw/quant_projects` under the standard run; both fail
  for the same root-duplicate reason — this is a genuine, non-fixture failure.

Pre-existing failures recorded separately: the 25 `test_no_root_fe_package`
failures are expected and are exactly the FE-P0-01 red-team finding; they do
NOT regress (they were already failing in the working tree before this task).

## 6. Evidence

- `evidence/r2/R21-FE-SOURCE-RESIDENCY.yaml`
- This audit: `docs/R21_SOURCE_RESIDENCY_AUDIT.md`
