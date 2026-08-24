# Source Authority Report

> Written: 2026-08-23 · HEAD `ec4ef105f5613e0889575f4f362f38c86020d267`
> Working tree: dirty (intended — see per-section dispositions below)

## 1. gitlink inventory (SUBMODULE_REACHABILITY truth, P0-8)

```
$ git ls-files --stage | grep -cE '^160000'
0
```

The repo declares **NO submodules** (zero index gitlinks; `.gitmodules` absent).
There is therefore nothing to fetch/clone, and `SUBMODULE_REACHABILITY` is
**N/A by construction**.  This MUST **NOT** be reported as PASS: the gate is
vacuous, not proven.  The manifest generator (`scripts/gen_verification_manifest.py`)
already honours this — `submodule_shas` is populated ONLY from real mode-160000
gitlinks, so it is now an empty dict and can never fabricate SHAs again.

## 2. cleaned_operators — canonical location + duplicate disposition (P0-7)

- **Canonical (production runtime + evidence hashing authority):**
  `<repo_root>/cleaned_operators/`  (459 tracked files; `__init__.py` blob
  `4e5abacc3e9ed7fcf12870c23d4952acfafea5ae`)
- **Duplicate:** `factor_engine/cleaned_operators/` (451 tracked files;
  `__init__.py` blob `92078c50152b0611ce6b44f9d6d25d7845cdd5f5`) — **archived**.

Both the wheel runtime and `evidence/current.py:_module_source_digest()`
point at the ROOT copy:
- the factor-engine wheel is built from the repo ROOT (`_build_backend.py`,
  `scripts/wheel_clean_install_smoke.py` both run `python -m build` with
  `cwd=<repo_root>`);
- all factor_engine code imports the top-level `cleaned_operators` package,
  which resolves to the ROOT copy on `sys.path` (grep for
  `factor_engine.cleaned_operators` finds **zero** imports);
- `evidence/current.py` resolves operator modules as
  `_REPO_ROOT / "cleaned_operators" / <rest>.py`.

**Disposition:** `factor_engine/cleaned_operators/` was moved (git mv) to
`factor_engine/cleaned_operators_archived/` — nothing deleted, all 451 files
(plus a new `README_ARCHIVED.txt` header) retained.  The archived `__init__.py`
blob is unchanged (`92078c50…`), i.e. the duplicate is preserved byte-for-byte
under the archive name and is no longer importable as
`factor_engine.cleaned_operators`.

`evidence/current.py:_module_source_digest()` is now **fail-closed**: it
asserts the canonical path is `_REPO_ROOT / "cleaned_operators"` and raises
`RuntimeError` if that directory is missing, rather than silently resolving a
non-canonical copy.

## 3. factor_assets — source authority (P0-9)

- **Pinned source:** commit `360836b5` (branch
  `remotes/origin/r2-q-empty-authority-gates`, "Final commit").
- **Tree:** `040000 tree e15976ea63794159e6f6cbeb563c91f86cd30951  factor_assets` (140 tracked files).
- **`__init__.py` blob at pin:** `bad79b23f8a7e48b11dbf40f20ae9190a13f4df3`.
- **Restore command (read-only git archive):**
  `git archive 360836b5 factor_assets | tar -x -C /home/sunhaiwei/quant_projects`
- **Now on disk:** `/home/sunhaiwei/quant_projects/factor_assets/` — 126 `*.py`
  files (140 tracked entries incl. 14 non-`.py` docs/YAML/README files), with
  `pyproject.toml` + `setup.py`.  Import OK
  (`factor_assets.__file__ = /home/sunhaiwei/quant_projects/factor_assets/__init__.py`).
  Tests: **714 passed, 43 skipped** (`PYTHONPATH=/home/sunhaiwei/quant_projects /tmp/fe2/bin/python -m pytest factor_assets/tests/ -q`).
  No missing-dependency failures; 38 warnings are datetime deprecation warnings only.
- **Un-ignored:** `factor_assets/` removed from `.gitignore` (was line 54);
  `git check-ignore factor_assets/` now returns 1 (not ignored); files appear
  as untracked (`?? factor_assets/`).
- **Honest tree hash:** the manifest generator computes the package tree
  SHA-256 from the on-disk files (per-package source-tree digests list
  `factor_assets`), so it can now compute an honest hash — previously the
  directory was absent (hollow).
- **NOTE — must be pinned in a future change:** factor_assets is currently
  UNTRACKED (restored into the working tree via `git archive`, not yet
  committed).  Until a commit lands, its source authority is the commit
  `360836b5` pin recorded above; a future approved change should `git add`
  the restored tree so it is pinned in the parent repo.

## Disposition summary

| Item | Before | After |
|------|--------|-------|
| `cleaned_operators` canonical | ROOT (4e5abacc…) | ROOT (4e5abacc…) — unchanged |
| `factor_engine/cleaned_operators` | live duplicate (92078c50…) | archived → `factor_engine/cleaned_operators_archived/` (92078c50… preserved) |
| `factor_assets/` | absent on disk; gitignored; hollow tree hash | restored from 360836b5; un-ignored; untracked (to be pinned in a future commit) |
| submodule gitlinks | 0 | 0 — SUBMODULE_REACHABILITY N/A-by-construction (NOT PASS) |
| `evidence/current.py:_module_source_digest` | resolved `_REPO_ROOT/"cleaned_operators"` silently | fail-closed: raises if canonical dir missing; docstring updated |
