# Loop Engineering V2 Session Handoff

Generated: 2026-08-17

## How to continue

Preferred full-context method:

```bash
claude --resume 49d4b856-8ff4-4444-8067-34c6501ea9e0 --fork-session
```

If resuming by handoff file instead, start a new Claude Code window rooted at `/home/shw` or `/home/shw/quant_projects` and send:

> 请读取 /home/shw/SESSION_HANDOFF.md，继承里面的任务背景、已完成工作、约束、当前状态和下一步，然后继续执行。不要重新规划，不要询问是否继续；直接从“下一步操作”开始，遵守所有测试和 Git 约束。

The persistent memory index under `/home/shw/.claude/projects/-home-shw/memory/` is shared by new sessions in the same project, but this handoff and `--resume ... --fork-session` carry the current work state more completely.

## User goal

Continue Loop Engineering V2 hardening under `/home/shw/quant_projects` continuously. Use the server working tree as the sole code truth and follow:

- `/home/shw/quant_projects/quant_factor_platform_blueprint/LOOP_ENGINEERING_6H_INSTITUTIONAL_HARDENING_V2_20260814.md`
- `/home/shw/quant_projects/LOOP_ENGINEERING_STATUS.md`
- `/home/shw/quant_projects/quant_factor_platform_blueprint/AI_GUIDE/00_START_HERE.md`
- `/home/shw/CLAUDE.md`

Required loop: audit -> reproduce/verify -> smallest correction -> focused serial tests -> independent validation -> ledger update -> repeat with another audit dimension.

Primary packages are `quant_evaluator`, `factor_preprocess`, `factor_assets`, and `factor_optimizer`. Touch FactorEngine/modeling/DataAccess/q only for genuine integration or correctness needs. Current user-highlighted scopes also include modeling namespace migration validation and QE `cache_v2` dependency invalidation/absolute TTL repair.

## Mandatory constraints

- Do not enter plan mode unless explicitly requested.
- Do not ask for confirmation or whether to continue; execute directly.
- Never run `git checkout`, `git restore`, `git stash`, `git clean`, or `git reset`.
- Never perform bulk AST rewrites.
- Preserve unrelated and concurrent dirty-tree changes.
- Every pytest/build/wheel/benchmark invocation must be serial, no xdist, with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

- Shared memory ceiling is 15 GiB; target under 12 GiB with reserve. Check memory before fan-out.
- Use at most four low-memory read-only agents and at most two disjoint writers.
- Do not run test/build/benchmark jobs concurrently.
- Maintain truthful capability labels: `PRODUCTION_CANDIDATE`, `SAFE_BY_GATING`, `RESEARCH_ONLY`, `OFFLINE_ONLY`, or `BROKEN`.
- Do not claim production readiness without exact current-tree evidence.
- Update `LOOP_ENGINEERING_STATUS.md` only with verified facts and explicit residual risks.
- Third-party GPL source is reference/corpus, not code-copy material.

## Workspace state

Primary working directory at handoff creation: `/home/shw/quant_projects/factor_assets`.

The package directories are part of one enclosing Git worktree rooted at `/home/shw/quant_projects`. Current branch reported from all inspected package paths:

```text
r2-q-empty-authority-gates
```

Current tracked/untracked status observed:

```text
M  /home/shw/quant_projects/LOOP_ENGINEERING_STATUS.md
M  /home/shw/quant_projects/factor_optimizer/factor_optimizer/adapters/quant_evaluator.py
M  /home/shw/quant_projects/factor_optimizer/tests/adapters/test_quant_evaluator.py
M  /home/shw/quant_projects/quant_evaluator/runtime/streaming_evaluator.py
M  /home/shw/quant_projects/quant_evaluator/tests/test_streaming_evaluator.py
?? /home/shw/quant_projects/data/cogalpha_lqtp_production/candidate_pool_neutral_cache/
```

The untracked data/cache directory was not created or modified by this work. Do not remove it. The tree was already dirty and may contain concurrent changes; inspect before every edit and work with them.

At the last `diff --stat`, the five tracked files above contained 142 insertions and 17 deletions in aggregate. No commit was created.

## Completed modifications

### FactorOptimizer QE adapter fail-closed hardening

Files:

- `/home/shw/quant_projects/factor_optimizer/factor_optimizer/adapters/quant_evaluator.py`
- `/home/shw/quant_projects/factor_optimizer/tests/adapters/test_quant_evaluator.py`

Verified installed QE contract:

```python
Evaluator.evaluate(self, factor_batch, label_bundle, metric_specs, use_chunking=True)
```

The stale FactorOptimizer adapter expects the historical evidence API with `factors`, `labels`, `metrics`, `context`, `DEFAULT_METRICS`, `MetricCatalog.list(tier=...)`, `Evaluator.get_evidence(evaluation_id=...)`, and evidence/result attributes. The installed QE lacks that API.

Changes made:

- `create_qe_adapter()` now rejects unavailable or structurally incompatible QE at construction with `OptionalDependencyMissing`.
- It validates required methods and keyword-callable parameter kinds using `inspect.signature`.
- It validates `get_evidence`, metric catalog list capability, `DEFAULT_METRICS`, and constructor viability.
- Production capability gating still runs before optional dependency probing.
- Mock adapter remains explicitly `RESEARCH_ONLY` and rejects production mode.
- Added regressions for installed incompatibility, missing evidence API, and positional-only signatures.

Independent validation confirmed the exact positional-only hole was closed. Residual limitation: signature-compatible components can still be behaviorally unusable; construction is structural compatibility evidence only. Do not represent this as real FO-to-QE integration.

### QuantEvaluator provenance serialization

Files:

- `/home/shw/quant_projects/quant_evaluator/runtime/streaming_evaluator.py`
- `/home/shw/quant_projects/quant_evaluator/tests/test_streaming_evaluator.py`

`StreamingEvaluationResult.to_dict()` previously returned `MappingProxyType` provenance directly, causing:

```text
TypeError: Object of type mappingproxy is not JSON serializable
```

It now returns `dict(self.provenance)`. Evaluator-owned provenance remains immutable, while serialized output is a distinct JSON-compatible dictionary. Tests now call `json.dumps()` and no longer require identity with the internal mapping proxy.

Independent validation covered both the explicit fixture and `evaluate_large_batch()` output and found no regression for evaluator-generated tuple/scalar provenance. A manually constructed result containing nested mutable objects would still be shallow-copied; that is outside the generated provenance contract.

### QuantEvaluator 2-D label asset validation: in progress

`StreamingEvaluator._validate_chunk()` now rejects 2-D labels whose asset dimension differs from the factor batch asset dimension. A regression currently covers direct `evaluate_stream()` with an undersized label axis.

Independent audit found a remaining confirmed hole:

- `evaluate_large_batch()` silently accepts oversized 2-D labels because `_extract_chunk_labels()` slices labels to the factor asset width before `_validate_chunk()` sees them.
- Example: factor assets 4, label assets 5. Parent dimensions are never validated; labels are silently truncated and updater execution proceeds.
- Undersized large-batch labels reject, direct streams reject both directions, and 1-D label broadcasting remains valid.

This scope is not closed. The next correction must validate the unsplit parent `FactorBatch`/`LabelBundle` before chunk extraction in `evaluate_large_batch()`, then add large-batch oversized and undersized regressions plus a 1-D allowed regression.

## Tests and independent evidence

All listed pytest commands were serial and used the required BLAS thread limits.

FactorOptimizer:

```text
17 passed in 0.12s
23 passed in 0.13s  # adapter + production capability after partial-contract correction
24 passed in 0.14s  # after positional-only correction
412 passed, 4 skipped in 1.41s  # complete FactorOptimizer tests
```

Independent FactorOptimizer validation:

```text
Installed incompatible QE rejection: PASS
Partial contract rejection: PASS
Production gating: PASS
Mock research-only gating: PASS
Positional-only signature rejection: PASS
Focused independent subset: 6 passed in 0.10s
Narrow structural fail-closed verdict: PASS
```

Syntax and formatting:

```text
python3 -m py_compile: PASS
git diff --check: PASS
```

QuantEvaluator streaming:

```text
40 passed in 14.20s  # provenance serialization correction
Independent: 40 passed in 14.62s; JSON probes PASS
41 passed in 14.54s  # after direct-stream label asset regression
```

Independent 2-D label probe:

```text
evaluate_stream undersized/oversized mismatch: rejects before updater
evaluate_large_batch undersized mismatch: rejects
evaluate_large_batch oversized mismatch: BUG, silently truncates
1-D labels: accepted for both entry points
Focused independent subset: 3 passed in 0.03s
```

One command initially used `python`, which is unavailable; rerunning with `python3` passed. One initial pytest command ran from the wrong directory and collected zero tests; absolute paths were then used successfully.

## Status ledger updates already made

`/home/shw/quant_projects/LOOP_ENGINEERING_STATUS.md` now records:

- FactorOptimizer QE adapter structural fail-closed scope as `CLOSED_LOCAL`, with `24` focused and `412 passed, 4 skipped` full-suite evidence, while real integration and production readiness remain open.
- QuantEvaluator provenance serialization as `CLOSED_LOCAL`, with two independent 40-test passes and residual streaming gaps listed.

Do not mark 2-D label asset validation closed yet; the large-batch oversized-label hole remains.

## Key conclusions and decisions

- Keep FactorOptimizer `RESEARCH_ONLY`. Real integration is absent.
- Do not adapt current QuantEvaluator to a stale historical adapter contract merely to make construction succeed.
- Before real FO-to-QE integration, define a typed mapping from QE metric outputs and optimization direction to SearchRunner's scalar `score` and `cost` contract.
- Split-aware evaluation and sealed-test contracts in FactorOptimizer remain deliberate fail-closed stubs; do not casually fill them.
- QuantEvaluator's current `evaluate_large_batch()` is chunked evaluation over a complete in-memory parent batch, not true external bounded-memory streaming.
- Streaming IC retains all per-chunk sufficient-statistic arrays and materializes a full result; constant-memory claims remain unsupported.
- Provenance JSON serialization is locally closed, but broader streaming memory/partition/timing issues remain open.
- FactorAssets SQLite focused tests previously passed, but broad operational production claims remain unsupported despite `PRODUCTION_CAPABLE = True` in the class.

## Remaining issues

### Immediate QuantEvaluator issue

1. Fix parent-level 2-D label asset validation in `evaluate_large_batch()` before any slicing.
2. Add oversized and undersized large-batch mismatch tests.
3. Assert updater does not run on invalid parent input if practical.
4. Preserve valid 1-D label broadcasting.
5. Run the full streaming test file serially.
6. Obtain independent validation and only then append an exact ledger entry.

### Current user-highlighted active scopes

- Modeling namespace migration validation.
- QE `cache_v2` dependency invalidation and absolute TTL repair.

Before modifying either, read the current ledger and local diffs because concurrent sessions may have advanced those scopes. Reproduce current-tree behavior before editing.

### FactorOptimizer

- Real QE integration missing.
- No metric-to-score/cost mapping.
- Split/sealed-test APIs remain stubs.
- Mock results use UUID plus process-randomized `hash()`, so reproducibility claims are misleading.
- `ADAPTERS_COMPLETE.md` and adapter README may overstate completeness and require truthful documentation correction.

### QuantEvaluator

- Streaming IC state scales with the stream and finalization allocates a full result.
- `evaluate_large_batch()` is not external streaming.
- Shape-based partition ambiguity and incomplete memory accounting remain.
- Timing/replay metadata can grow with all observations.
- Full-versus-stream IC parity needs stronger evidence.
- QE `cache_v2` dependency invalidation/absolute TTL is an explicit active scope; inspect current code/tests and ledger first.

### FactorAssets

- Complete adversarial SQLite v1 validation, including concurrent transitions, decision replay, migration drift/newer schema, crash semantics, exports, and factory behavior.
- Keep production claims conservative until backup/restore, outbox, permissions, and operational evidence exist.

### FactorPreprocess

- Close policy/callable parameter schema gaps.
- Validate production preset execution and fail-closed defaults.
- Inspect mismatches such as `limit` vs `max_lag` and `span` vs `halflife`.
- Validate implementation/spec hashes and evidence completeness.

### Packaging/extraction

- Continue wheel payload intersection, clean installation, install-order, and uninstall-survival tests.
- Do not claim extraction closure from dirty-tree evidence alone.

## Next operation

Start with the interrupted verified QuantEvaluator defect:

1. Read the current `evaluate_large_batch()` and `_extract_chunk_labels()` implementations plus nearby tests.
2. Add parent contract validation before generating chunks so oversized 2-D labels cannot be truncated.
3. Add focused regressions for oversized/undersized 2-D parent labels and valid 1-D labels.
4. Run `tests/test_streaming_evaluator.py` serially with BLAS single-thread limits.
5. Send a low-memory read-only agent to adversarially verify both entry points.
6. Update `LOOP_ENGINEERING_STATUS.md` only if independent validation passes.
7. Then inspect current-tree progress for modeling namespace migration and QE `cache_v2` absolute TTL/dependency invalidation, choose a reproduced open defect, and repeat the loop.

Do not restart broad architecture discovery; use the existing taskbook, ledger, persistent memories, and this handoff.
