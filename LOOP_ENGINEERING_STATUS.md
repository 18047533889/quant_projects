# Loop Engineering Status (compact)

**Updated:** 2026-08-19
**Current HEAD:** `bce4c0a9` (`bce4c0a98e638734aca32b212c586c3b59c1562f`)
**Full history (do NOT load into agent context):** `docs/R2_HISTORY_ARCHIVE.md`

## Hard rules

- **LOCAL ONLY:** no GitHub/`gh`/fetch/push/remotes.
- No `git checkout|restore|stash|clean|reset`; working tree is truth.
- ≤2 non-overlapping low-memory Writers + 1 read-only Auditor; heavy tests serial with BLAS/OpenMP/MKL/Polars threads=1, no xdist.
- Never claim PASS / production-ready for an unrun gate; broad QE/full repo/root CI stay `NOT_RUN` unless recorded.
- Never load/prompt `docs/R2_HISTORY_ARCHIVE.md`; pass only owned files + ID/evidence pointers.

## Truth matrix

| Surface | Status |
|---|---|
| Q / K | `NOT_PRODUCTION_CERTIFIED` (fail-closed) |
| Polars | `PARTIALLY_VERIFIED` |
| DuckDB | `PARTIALLY_VERIFIED` |
| DataAccess | local focused evidence only; live COS/S3 `NOT_RUN` |
| FactorAssets | `NOT_PRODUCTION_CERTIFIED` |
| Modeling | `NOT_PRODUCTION_CERTIFIED` |
| QuantEvaluator | local focused evidence only; real Redis / broad QE `NOT_RUN` |
| Root CI | `NOT_RUN` |

## Still open / NOT_RUN

- Broad QuantEvaluator suite: previously exit **137** → `NOT_RUN / RESOURCE_BLOCKED`
- Real Redis, root CI, full compile, registry bootstrap, backend parity, PIT poison
- Live COS/S3 DataAccess integration
- Independent review agents: repeatedly fail with **prompt too long** if archive/status is loaded whole
- Residual LQTP-unrunnable RankIC>2% formulas (ADX / RSI_WILDER / pow⅓ / UInt64 mix): 20 left in aug18 materialize

## Latest local evidence (pointers only)

| ID / topic | Result | Evidence |
|---|---|---|
| 8.18 RankIC combo | 197 factors used | `data/cogalpha_lqtp_production/factor_lake_aug18_window/_combo/` |
| Q PlanNode ID collision | fail-closed; commit `19d2ff95` | `factor_engine/evidence/r2/Q2-P0-019-q-plan-node-abi.yaml` |
| Q resident release / batch | focused pass | `Q2-P0-009.yaml`, `Q2-P0-020*`, `Q2-P0-021-022*` |
| Q shared direct namespace | 27 focused tests pass; broader Q gates `NOT_RUN` | `factor_engine/backend/q_backend/test_q_residency.py`, `factor_engine/evidence/r2/Q2-P0-023.yaml` |
| R19 backend/direct-use inventories | current-SHA inventory only; certification `NOT_RUN` | `factor_engine/evidence/r2/BACKEND_GAP_MATRIX.yaml`, `DIRECTUSE_REHABILITATION_MATRIX.yaml` |
| Q `ts_beta` | quarantined (no lowering) | `Q2-P0-008.yaml` |
| POL2 moment/kurt | no defect; oracles added | `POL2-P0-004.yaml` |
| DA semantic catalog identity | latency in correctness key | `dataaccess/DA2_P0_002_MANIFEST.yaml` |
| DA PIT latest revision | focused repair | `dataaccess/R2-P0-039_MANIFEST.yaml` |
| FE multi-region physical plan | local repair | `evidence/r2/R2-P0-059-062-063_manifest.yaml` |
| FA adapter / assembly | focused pass | `FA2-P0-003.yaml`, `FA2-P0-004.yaml` |
| QE cache / streaming / mean_ic | focused pass | `quant_evaluator/tests/test_cache_v2.py` et al. |
| FactorAssets legacy packaging | focused pass; 2 tests | `factor_assets/tests/test_legacy_packaging.py` |
| FactorPreprocess registry admission | fail-closed; 44 tests | `factor_preprocess/tests/registry/test_transforms_registry.py` |
| FactorOptimizer sealed test boundary | focused pass; 160 search tests; production remains fail-closed | `factor_optimizer/tests/search/` |
| FactorPreprocess exposure shape | fail-closed; 741 local pass | `tests/adapters/test_data_access.py` |

## How to append

Add **one** short bullet under “Latest local evidence” (ID · result · path).
Move narrative detail into `docs/R2_HISTORY_ARCHIVE.md` or an evidence YAML. Keep this file under ~120 lines.

## Auto-context (2026-08-19)

Compacted Claude auto-load surfaces: `/home/shw/CLAUDE.md`, `CLAUDE.md`, `MEMORY.md`, campaign/platform/findings pointers, blueprint `START_CLAUDE_CODE.md` / skill.  
Full memory index + 84 detail notes → `~/.claude/projects/-home-shw/memory/archives/` (do not load).  
46 historical root taskbooks → `archives/taskbooks/` with stub pointers at old paths (do not load stubs' targets wholesale).
