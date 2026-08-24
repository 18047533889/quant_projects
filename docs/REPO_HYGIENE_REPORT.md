# R46 P1-AA + P1-32 / P1-33 — Repo Hygiene + Secret Scan Report

Date: 2026-08-23 · Agent: non-destructive, read-only audit. No `git rm` / `git mv` /
`rm -rf` / commit / push was performed. Working tree is truth.

---

## 1. Tracked-Hygiene Inventory

All numbers below are from `git ls-files` (tracked files), `du`/`wc -c` (read-only).

Total tracked files: **10,058** · total tracked bytes: **210,289,297 (~200 MB)**.

### (a) `vectorbt_qs/examples/output/**` — P1-32

| Metric | Value |
|---|---|
| Tracked files | **788** |
| Total tracked bytes | **132,679,095 (~126.6 MB)** |
| Share of all tracked bytes | **~63%** |

Breakdown by extension:

| ext | files | bytes |
|---|---|---|
| `.parquet` | 339 | 26,541,694 |
| `.png` | 272 | 66,192,965 |
| `.html` | 52 | 37,663,161 |
| `.csv` | 74 | 2,160,693 |
| `.json` | 51 | 120,582 |

Dominant subdir: `accurate_batch_test` (702 of the 788 files). `B00x_*` backtest
golden dirs hold 4–5 files each.

These are **already tracked** even though `.gitignore` already contains both
`vectorbt_qs/examples/output/` and `*.parquet` — `.gitignore` does not untrack
already-committed files. This is the classic P1-32 shape: backtest artifacts
(parquet weights + png/html reports) live in the repo and are re-generated on
every backtest run.

### (b) `bin/` `*.bak` / `*api-backup*` — **P1-33 (4 files, 22,641 bytes)**

| path | bytes |
|---|---|
| `bin/clean-cos-ro.bak.20260620212412` | 2,997 |
| `bin/cos-api.api-backup-20260817-053454` | 7,282 |
| `bin/cos-api.bak.20260621031149` | 5,234 |
| `bin/cos-api.bak.20260621031508` | 7,128 |

All 4 exist in the **active** `bin/` tree (not just git history). The `bin/`
active entries `clean-cos-ro -> cos-api` and `candidate-cos -> cos-api` are
symlinks (tracked as git symlinks); the `.bak`/`api-backup` copies are dead
backups of the old inline COS client that should not be in the active tree.

### (c) Stale logs — 2 files

| path | bytes |
|---|---|
| `.pytest_full.stale.txt` | 1,866 |
| `factor_engine/.pytest_full.txt` | 1,866 |

Both are stale pytest output capture; `.pytest_full.txt` is explicitly flagged
as a stale release-truth artifact by `scripts/gen_verification_manifest.py`
(`stale_notes`, REL-P0-02). No other `.log`/`.log`-like tracked logs were found
(0 tracked `*.log`).

### (d) Large parquet / csv research outputs (beyond vectorbt)

- Tracked `.parquet` outside `vectorbt_qs/examples/outputs/`: **21 files, ~1.97 MB**.
  Locations are almost all intentional test goldens / evidence fixtures:
  `vectorbt_qs/tests/baselines/…`, `tests/integration/fixtures/golden`,
  `factor_engine/docs/evidence/r37`, `docs/evidence/r37`, `dataaccess/docs/evidence/r30`,
  `factor_engine/docs/evidence/model_operators`. These are small, deterministic
  goldens and are acceptable to keep (moved/organised under
  `tests/fixtures/` as noted below).
- Top large non-vectorbt tracked files worth noting: `runs/inventory/unmapped_existing_values.json` (29.4 MB), `docs/reports/2026-08-23/all_eval_full.json` (24.5 MB), `factor_detail_pages.zip` (320 KB — but zips are gitignored; check whether tracked).
  - `unmapped_existing_values.json` is a research/inventory dump and is a candidate to move off-repo (see remediation).

---

## 2. Secret Scan — Verdict: **CLEAN (no real secret in tracked files)**

Scanned all tracked files (via `git ls-files | xargs grep`) for real-secret
patterns and for literals in the `bin/*.bak` scripts:

- **No `sk-[A-Za-z0-9]{20,}` OpenAI-style keys** anywhere in tracked files.
- **No `AKIA…`/`ASIA…` AWS access-key IDs** anywhere.
- **No literal 40+ char credential values** (`secret/password/token/key = "<40+ alnum>"`) anywhere tracked.

The 66 files matching looser patterns (`password=`, `token=`, `secret_access`)
were all **config/object references, environment lookups, or test-only values**:

- `dataaccess/security/credentials.py` reads keys from env (`os.environ.get("COS_SECRET_KEY", …)`, `COS_SESSION_TOKEN`, `AWS_SESSION_TOKEN`); never a literal.
- The `.bak` COS scripts use `cfg["token"]` / `cfg["endpoint"]` pulled from a
  config dict/env (`cos-api.api-backup` builds `"Authorization": "Bearer " + cfg["token"]`), never a hardcoded literal.
- Test-only literals found: `password="same"`, `password="pass-aaa"/"pass-bbb"`, `token="snap1"`, `token="A"` — clearly synthetic fixture values, not real secrets.

Note on git history: 102 commits exist. History was **not** dumped (per task).
Because the current tracked working tree is clean of real secrets and `.env` is
untracked/gitignored, no rotated-credential remediation is required; if a strict
historical secret-rewrite policy exists, it is a separate follow-up and was not
performed here (out of scope, non-destructive).

---

## 3. Non-Destructive Remediation Plan

Actionable by a future approved push. Working-tree-is-truth; **nothing removed
here**.

### 3.1 `vectorbt_qs/examples/outputs/` (P1-32)

1. **Move artifacts to COS/object store** (`cos://vectorbt/examples/outputs/…`),
   keep a manifest + sha256 hashes + byte sizes in-repo
   (e.g. `vectorbt_qs/examples/outputs/MANIFEST.json`).
2. Keep **only small deterministic golden fixtures** under
   `vectorbt_qs/tests/fixtures/` (the `B00x_*` single-backtest png/csv that tests
   actually assert against). The bulk `accurate_batch_test` parquet/html/png tree
   (702 files) goes to COS.
3. **Recommended push (executes later, by an approved commit):**
   - `git rm -r --cached vectorbt_qs/examples/outputs/`
     (untrack; `.gitignore` already covers it so it will not re-add).
   - Add the manifest; wire the CI/regeneration script to emit only the small
     fixtures into the tree.
4. This will cut tracked bytes ~63% (~200 MB → ~77 MB).

### 3.2 `bin/*.bak` / `*api-backup*` (P1-33)

- These are dead snapshots of the old COS scripts. **Remove from the active tree**
  (`git rm`), keeping them only in git history (they are recoverable from git if
  ever needed).
- Keep the live `bin/cos-api -> /usr/local/bin/research-cos` symlink and
  `mafm2_cos_api_smoke.sh`; optionally gate the smoke script.
- `.gitignore` add `bin/*.bak*` / `bin/*api-backup*` so future backups never
  re-enter tracking.

### 3.3 Stale pytest logs

- Delete `vectorbt_qs/examples/outputs`… no — delete `.pytest_full.stale.txt` and
  `factor_engine/.pytest_full.txt` (`git rm`), and add `*.pytest_full*` /
  `.pytest_full*.txt` to `.gitignore`. `gen_verification_manifest.py` already
  tolerates their absence (records `stale_notes` only when present).

### 3.4 Other

- `runs/inventory/unmapped_existing_values.json` (29.4 MB): move to COS and keep
  only a manifest/summary; gitignore `runs/inventory/*.json` if it is regenerable
  research output.
- The 21 test/evidence `.parquet` goldens under `tests/…`, `docs/evidence/…`,
  `dataaccess/docs/evidence/…` are **small deterministic fixtures** — keep them
  (optionally consolidate under `tests/fixtures/`).

### Proposed `.gitignore` update (append)

```gitignore
# R46 hygiene
vectorbt_qs/examples/outputs/          # already present; keep
bin/*.bak
bin/*api-backup*
bin/*.api-backup*
*.pytest_full*.txt
runs/inventory/*.json                  # research/inventory output -> COS
```

---

## 4. Vectorbt outputs vs source-snapshot / R46 source identity

- **Docs / reports / evidence are already excluded** from the R46 source identity:
  `evidence/current.py` `_NON_SOURCE_OUTPUT_DIRS = {evidence, build, docs/reports,
  docs/generated, archives}` are excluded from the dirty-tree digest
  (`_root_diff_hash`, `_tracked_untracked_source_hash`). This is the "R46
  excludes docs/reports and evidence" behaviour you flagged.
- **However, `vectorbt_qs/examples/outputs/` is NOT excluded** from the platform
  source scan:
  - `evidence/source_snapshot.py` `PLATFORM_PACKAGES` includes `vectorbt_qs`.
    `_package_tree_merkle(vectorbt_qs)` descends the package dir and only skips
    the `_PLATFORM_EXCLUDE_SUFFIXES` (`*.parquet`, `*.pkl`, `*.feather`, …) and
    `EXCLUDE_DIRS` (which lists `evidence`, `docs`, `build`, `data`, `logs`,
    `temp`, `fixtures`, `synthetic`, … — **not** `examples` or `output`).
  - Result: **449 non-parquet tracked files** under
    `vectorbt_qs/examples/outputs` (272 png + 74 csv + 52 html + 51 json,
    ~68 MB) **would be hashed into the `vectorbt_qs` package-tree leaf** of the
    `SourceTreeIdentity`. Every backtest re-run that regenerates any of these
    files changes the platform source identity and marks the tree dirty.
  - The 339 parquet files are excluded by suffix only (safe), but the 
    png/csv/html/json are not.
- **Recommendation**: add `examples` (and/or `output`) to
  `_package_source_exclude_dirs` in `evidence/source_snapshot.py`, or (cleaner)
  move the whole `vectorbt_qs/examples/outputs` tree to COS so nothing under it
  is tracked at all. Also consider adding `outputs` to the source-scan
  `EXCLUDE_DIRS`. **Without one of these, backtest outputs pollute the
  SourceTreeIdentity / dirty-tree detection** on every run.

---

## 5. Honest limitations

- Secret scan is regex-based on tracked file content; it cannot detect secrets
  embedded in compressed/binary blobs (zips, html with inline base64 maps —
  e.g. `vectorbt_qs/vectorbt/docs/…/interactive/…` showed inline mapbox
  `api_key=` tokens from bundled JS, which are public mapbox tokens, not
  credentials) or in git history (history not dumped, per task).
- No live secret scan tooling (gitleaks/trufflehog) run; this is a manual
  pattern audit.
- Size numbers are working-tree byte counts via `wc -c` (git blob sizes can
  differ slightly due to clean filters); adequate for inventory purposes.
- This is a report + plan only; no remediation was applied (non-destructive
  mandate).
