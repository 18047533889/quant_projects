# Wave 0 Local Repository Snapshot

Snapshot time (UTC): `2026-08-13T13:48:17Z`

Scope: read-only inspection of `/home/shw/quant_projects`. The only file written by this audit is this report. No production or legacy source file was modified. No `clean`, `checkout`, `restore`, `stash`, pull, or other repository-changing operation was run.

## 1. Host and repository identity

| Fact | Observed value |
|---|---|
| `pwd` | `/home/shw/quant_projects` |
| `python3 --version` | `Python 3.10.12` |
| Python executable | `/usr/bin/python3` |
| `VIRTUAL_ENV` | empty |
| `CONDA_PREFIX` | empty |
| Git repository | yes, `/home/shw/quant_projects/.git` |
| Current branch | `main` |
| HEAD | `ddb03749b7ff85e63b633770c43dfbbb7562af19` |
| `git log -1 --oneline` | `ddb03749 feat(factor_engine): backend/runtime/modeling remediations and test updates.` |
| Local branches | 43 |
| Branches attached to worktrees | 42 |
| Tracked files | 6,485 |
| Tracked DataAccess files | 405 |
| Tracked FactorEngine files | 2,762 |

`git branch` marks `main` current. It also lists `claude/field-upgrade` and 41 `worktree-agent-*` branches; 42 branches have a non-empty worktree path. Worktree contents were excluded from this audit.

## 2. Working-tree state

Command results:

- `git status --short`: 183 entries, all `??` untracked entries; zero tracked-change entries.
- `git diff --stat`: no output.
- `git diff --stat --cached`: no output; zero staged paths.
- `git ls-files --others --exclude-standard`: 374 individual untracked files. This is larger than 183 because short status collapses untracked directory contents.
- `git ls-files --others --ignored --exclude-standard`: 46,130 ignored file paths.

The 183 short-status entries group by top-level path as follows:

| Top-level path | Status entries |
|---|---:|
| `data/` | 157 |
| `scripts/` | 14 |
| `.github/` | 5 |
| `.claude/` | 1 |
| `.cursor/` | 1 |
| `factor_engine/` | 1 |
| `quant_factor_platform_blueprint/` | 1 |
| `_r38_spool/` | 1 |
| `DataAccess_R32_最新HEAD_全量终审_一次性持续整改任务书_20260812 (1).md` | 1 |
| `QuantProjects_Current_HEAD_Full_Audit_MultiAgent_Master_Remediation_20260813.md` | 1 |

The five untracked workflow files are:

- `.github/workflows/dataaccess-final-closure.yml`
- `.github/workflows/dataaccess-r30-perf-gate.yml`
- `.github/workflows/factorengine-core.yml`
- `.github/workflows/factorengine-production-runtime.yml`
- `.github/workflows/secret-scan.yml`

This report records the untracked tree and does not classify it as disposable.

## 3. Actual directory snapshot

The audit ran `find /home/shw/quant_projects -maxdepth 2 -type d` and excluded `.git`, `.venv`, Python/test/tool caches, and `.claude/worktrees`. The observed top-level project directories are:

```text
ashare_lqtp_kit
AutoFactorEvaluation-RECONSTRUCT
build
.claude
coscli_output
.cursor
data
dataaccess
docs
factor_cold_start
factor_engine
.factor_engine_service
factor_layer
factor-pool-standard
gtja191
lqtp-python-grpc-examples
quant_factor_platform_blueprint
_r38_spill_store
_r38_spool
raw_data_layer
.replace_backup_20260801_190021
.replace_backup_20260802_181017
scripts
.tmp
toolkit
week2_pv_factors
```

Core and legacy second-level directories observed:

```text
dataaccess/{benchmarks,build,clickhouse,config,constraints,contract,core,cos,data_access.egg-info,deploy,dist,docs,ops,quality,r30,read,registry,runtime,scripts,security,service,snapshot,tests,workspace_data,write}
factor_engine/{api,audit,backend,benchmarks,build,cache,.claude,cleaned_operators,data,docs,evidence,examples,execution,expr,factor_engine.egg-info,.factor_engine_service,factor_recipes,fields,ir,market,mining,modeling,operator_contracts,planner,planning,_r38_spill_store,_r38_spool,research_operators,research_tools,runtime,scripts,security,semantic,service,storage,tests,tools,UNKNOWN.egg-info,util}
factor_layer/{factor_admission,factor_agent,factor_evaluation,factor_evaluation_alphapurify,factor_indicators_lysj,factor_pool,README.md}
AutoFactorEvaluation-RECONSTRUCT/{assetization,data_access,docs,evaluation,factor_engine,gateway,initialization,integrations,purification,scripts,tests,utils}
gtja191/{autofactor,candidate_pool,docs,dsl,examples,formulas,lib,scripts,source,tests}
week2_pv_factors/{candidate_pool,formulas,lib,scripts,source}
raw_data_layer/{data_daily_update,raw_data_cleaning,raw_data_fetching}
factor_cold_start/{autofactor,catalogs,reports,scripts,source,tests}
toolkit/{alpha_tools}
quant_factor_platform_blueprint/{AI_GUIDE,.claude,DESIGN,WAVE_0_OUTPUT}
```

The filesystem contains `/home/shw/quant_projects/.venv`, but it was excluded from the tree snapshot as required. The current shell is not using it.

The four planned runtime package directories are all absent at repository root:

- `/home/shw/quant_projects/quant_evaluator`
- `/home/shw/quant_projects/factor_optimizer`
- `/home/shw/quant_projects/factor_assets`
- `/home/shw/quant_projects/factor_preprocess`

## 4. DataAccess ground truth

### Source package

- Path: `/home/shw/quant_projects/dataaccess`
- Packaging file: `/home/shw/quant_projects/dataaccess/pyproject.toml`
- Build backend: `setuptools.build_meta`, requiring `setuptools>=68` and `wheel`.
- Distribution name/version: `data-access`, `0.10.2`.
- Required Python: `>=3.10`.
- Package layout: nonstandard explicit mapping from physical root/subdirectories to `data_access` and `data_access.*` packages.
- Console entrypoints: `data-access-server = data_access.service.__main__:main`; `data-access-quality = data_access.quality.cli:main`.

### Public entrypoint

`/home/shw/quant_projects/dataaccess/__init__.py` is the root public initializer. It declares 47 names in `__all__`, covering:

- Store/engine: `get_store`, `reset_store`, `get_shared_engine`, `reset_shared_engine`, `DataAccessStore`, `DuckDBEngine`.
- Errors: `DataAccessError`, `ValidationError`, `AmbiguousSemanticFieldError`, `DataError`, `EngineError`, `DeadlineExceeded`.
- Aggregation and calendar/session contracts.
- PIT event index and record.
- Physical plan and Contract IR builders.
- Query budget and key policy.
- Snapshot/read results and scan/read/relation handles.
- `DataRequest`, `ReadPlan`, semantic field catalog APIs.
- COS dataset contracts, validation, event-clock resolution, return normalization, and semantic fingerprinting.

The initializer sets `__version__ = full_version(_base_version)` and `__build_sha__ = build_sha()`. Its package-metadata fallback base version is `0.8.0+local` if the distribution is not found.

### Active interpreter state

`python3 -m pip show data-access factor-engine` reports:

- Installed `data-access`: version `0.8.0`, location `/home/shw/.local/lib/python3.10/site-packages`.
- Installed `factor-engine`: not found.

Therefore the active interpreter's installed DataAccess distribution version is not the source `pyproject.toml` version. The audit did not import either source tree or alter `PYTHONPATH`.

## 5. FactorEngine ground truth

### Source package

- Path: `/home/shw/quant_projects/factor_engine`
- Packaging file: `/home/shw/quant_projects/factor_engine/pyproject.toml`
- Build backend: `setuptools.build_meta`, requiring `setuptools>=68` and `wheel`.
- Distribution name/version: `factor-engine`, `0.3.1`.
- Required Python: `>=3.10`.
- Packaging layout: top-level packages such as `api`, `backend`, `cleaned_operators`, `expr`, `ir`, `runtime`, `service`, and `storage`; eight top-level `py-modules` are also declared.
- Console entrypoint: `factor-engine-serve = service.app:main`.
- Optional `data`/`full` extras depend on `data-access>=0.2.0`.

There is no `/home/shw/quant_projects/factor_engine/__init__.py`. Public entrypoints are top-level packages rather than a `factor_engine` import package.

### Public entrypoints inspected

- `/home/shw/quant_projects/factor_engine/api/__init__.py`: DSL expression-building entrypoint; explicitly exports 10 names: `Factor`, `col`, `field`, `delay`, `make_cleaned_call_factory`, `rank`, `ts_mean`, `ts_std`, `ts_std_dev`, `zscore`. It also dynamically exposes registry-allowlisted operators through `__getattr__`.
- `/home/shw/quant_projects/factor_engine/expr/__init__.py`: explicitly exports 8 expression symbols: `CleanedCall`, `ColumnRef`, `Expr`, `FieldRef`, `Literal`, `canonical_expression`, `ensure_expr`, `expression_payload`.
- `/home/shw/quant_projects/factor_engine/service/__init__.py`: exports `create_app` and `main`.

The active interpreter does not have the `factor-engine` distribution installed.

## 6. Packaging and import-path facts

Packaging metadata found within depth three, excluding backups and worktrees:

- `/home/shw/quant_projects/dataaccess/pyproject.toml`
- `/home/shw/quant_projects/factor_engine/pyproject.toml`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/data_access/pyproject.toml`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/pyproject.toml`

No `setup.py` or `setup.cfg` was found by that scan. Backup copies of DA/FE `pyproject.toml` also exist under `.replace_backup_*`; they are not treated as active packages.

The active Python path is:

```text
/usr/lib/python310.zip
/usr/lib/python3.10
/usr/lib/python3.10/lib-dynload
/home/shw/.local/lib/python3.10/site-packages
/usr/local/lib/python3.10/dist-packages
/usr/lib/python3/dist-packages
```

The repository root is not present in this interpreter-reported `sys.path`. A repository-wide scan found many explicit `sys.path.insert`/`append` sites, including legacy runners, examples, scripts, DA/FE scripts, gateway scripts, and tests. One directly relevant legacy example is `/home/shw/quant_projects/factor_layer/factor_admission/run_from_config.py`, which inserts the monorepo root.

## 7. Tests, benchmarks, and CI

Counts exclude `__pycache__` and pytest/tool caches.

| Area | `test_*.py` files | All non-cache files under `tests/` | Files under `benchmarks/` | Python benchmark files |
|---|---:|---:|---:|---:|
| DataAccess | 137 | 144 | 12 | 12 |
| FactorEngine | 956 | 1,006 | 5 | 2 |

DataAccess benchmark files:

```text
benchmark_cos.py
benchmark_factor_batch.py
benchmark_incremental.py
benchmark_join.py
benchmark_local_daily.py
benchmark_local_minute.py
benchmark_session_reuse.py
bench_read.py
fixtures.py
__init__.py
report.py
run_benchmarks.py
```

FactorEngine benchmark files:

```text
backend_cost_baseline.json
backend_operator_bench.py
bench_polars_optimization.py
operator_manifest.json
README.md
```

There are 11 workflow files under `.github/workflows`: 6 tracked and 5 untracked.

Tracked workflows:

```text
autofactor-platform-integration.yml
factor-cold-start-library.yml
factor-engine-contract-hardening.yml
factor-pack-evaluation.yml
gtja191-production-gate.yml
operator-surface-gate.yml
```

The five untracked workflows are listed in section 2. No tests or benchmarks were executed for this read-only snapshot; only files were inventoried.

## 8. Legacy directories confirmed

The following expected legacy/corpus directories exist at these canonical non-worktree paths:

- `/home/shw/quant_projects/factor_layer`
- `/home/shw/quant_projects/factor_layer/factor_agent`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT`
- `/home/shw/quant_projects/toolkit`
- `/home/shw/quant_projects/factor-pool-standard`
- `/home/shw/quant_projects/gtja191`
- `/home/shw/quant_projects/week2_pv_factors`
- `/home/shw/quant_projects/ashare_lqtp_kit`
- `/home/shw/quant_projects/raw_data_layer`
- `/home/shw/quant_projects/factor_cold_start`

Backup trees also exist under `.replace_backup_20260801_190021` and `.replace_backup_20260802_181017`; they are excluded as active migration sources unless a later audit explicitly chooses them as historical comparison material.

## 9. Named migration target files located

Canonical files named by the Wave 0 guides:

| Target | Actual path | Bytes |
|---|---|---:|
| Factor evaluation pipeline | `/home/shw/quant_projects/factor_layer/factor_evaluation/pipeline.py` | 19,001 |
| Batch metrics | `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/evaluation/batch_metrics.py` | 18,443 |
| FactorAnalyzer | `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/FactorAnalyzer.py` | 118,560 |
| Exposures | `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 27,020 |
| AlphaPurifier | `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/AlphaPurifier.py` | 10,798 |
| APr utilities | `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/APr_utils.py` | 150,657 |
| Cross-sectional toolkit | `/home/shw/quant_projects/toolkit/cross_sectional.py` | 5,863 |
| Toolkit registry | `/home/shw/quant_projects/toolkit/registry.py` | 6,006 |
| Alpha-tools registry | `/home/shw/quant_projects/toolkit/alpha_tools/registry.py` | 7,069 |
| Generated alpha library | `/home/shw/quant_projects/toolkit/alpha_tools/generated_library.py` | 12,727 |
| Admission catalog | `/home/shw/quant_projects/factor_layer/factor_admission/catalog.py` | 10,446 |

Archived/example duplicates of `AlphaPurifier.py`, `APr_utils.py`, `Exposures.py`, and `FactorAnalyzer.py` also exist under `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify`; these are distinct from the canonical files above.

Gateway/assetization targets were located at:

- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/models.py` (`Candidate`)
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/complexity.py` (`ComplexityReport`, `ComplexityEvaluator`)
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/deduplicator.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/models.py` (`FactorCandidate`, `FactorAsset`)

Parallel copies of gateway modules also exist under `gateway/gateway/` and at the gateway root. This snapshot records the duplication and does not designate one copy authoritative beyond the explicit symbols located above.

## 10. Wave 0 boundary findings

These are direct observations, not migration decisions:

1. DataAccess and FactorEngine are present as substantial existing source packages with public contracts, tests, benchmarks, and CI coverage.
2. The four planned new package directories do not yet exist at repository root.
3. The active system Python does not resolve the server source tree by default; its installed DataAccess version is older than the checked-out source version, and FactorEngine is not installed.
4. The root working tree has no tracked modifications but has a large untracked surface: 183 short-status entries representing 374 untracked files.
5. All named high-value legacy areas from the guide were found, including duplicate/archive copies and gateway/assetization models.
6. The repository contains explicit local path hacks. New packages cannot treat the current monorepo import behavior as proof of independent installability.

This report is a repository snapshot only. Function-level reuse classifications belong in the separate Wave 0 migration inventories and require source/formula audits plus characterization tests.
