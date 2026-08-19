# Platform Loop (compact) — NOT FactorEngine / DataAccess

**Updated:** 2026-08-19 · **HEAD:** `60bb18ba` (monorepo)  
**Shared rules:** `CLAUDE.md`, `/home/shw/CLAUDE.md`, `LOOP_ENGINEERING_STATUS.md` (read platform rows only)

## Session results (2026-08-19, platform-loop session)

- `FO-P0-1` → sealed-test boundary fail-closed at API level (EvaluationProtocol-only evaluators, `require_evaluation_protocol=True` default, one-shot seal immutability, checkpoint tamper rejection, disjointness check) → `factor_optimizer/tests/search/` **445 passed, 4 skipped**; `factor_optimizer/factor_optimizer/search/runner.py`
- `FA-P0-3` → similarity REJECT replaced by SHADOW / SHADOWED_COEXIST chain (fail-closed: no novelty signal → REJECTED_SIMILARITY; novelty_score==0 → SHADOW; out-of-range/nonfinite → reject) → `factor_assets/selection/policy.py`; `factor_assets/tests` **610 passed, 43 skipped**
- `FA-P0-2` → assembly ranking now evidence-based (decision recency, factor_id tiebreak; family round-robin for family_robust/diverse; unknown policy fails closed) → `factor_assets/assembly/engine.py`
- `FA-P0-13` → FactorMembership + FactorSetArtifact + assembly_hash/policy_hash/snapshot_ref/split_ref provenance on FactorSet → `factor_assets/contracts/factor_set.py`, `factor_assets/assembly/engine.py`
- `FA-P0-12` → orthogonal ValidationStatus/HealthState enums + FactorAsset fields (+serialization codec) → `factor_assets/contracts/lifecycle.py`, `contracts/asset.py`, `registry/serialization.py`
- Independent scoped review (FA diff) → 9 findings; P0 (exact-duplicate admission at default floor) + P1 (dead family round-robin) + novelty_refs provenance + hash length-prefixing all fixed same session; `audit_reports/fa_selection_assembly_audit_2026-08-19.yaml`
- `QE-P0-5` → cache unified on cache_v2 (CacheProtocol, V2IntermediateCache default, IntermediateCache deprecated, provenance pickling fix) → `quant_evaluator/tests/test_cache_v2.py` 131 pass + main suite green; `quant_evaluator/cache/`
- `QE-Q-P0-002` → quantile binning parity fixed across all 6 implementations (reference/fast/numba-kernel/quantile_numba/polars/cupy): shared `_percentile_boundaries` + side='right' tie policy + 1e-9 integer-position snap; polars boundaries now computed in NumPy and joined (bit-exact); raw quantile_counts below min_assets → `quant_evaluator` main suite **1143 passed, 24 skipped** (`/tmp/qe_main8.log`); scoped review P1 (6th impl `metrics/quantile_numba.py` missed) + P2 dead `factor_id` column fixed same session
- Integration: `integration_tests/` **174 passed** (wheel-boundary test skipped: env lacks `python -m build`)
- `QE-P0-7` → last two EXPERIMENTAL metrics (block_bootstrap_ci, factor_turnover_rate) bound to scalar-per-factor compute_fn adapters; `register_metric` now fail-closed on STABLE-without-compute_fn; registry tests assert contract → 53 focused pass, main suite **1151 passed, 24 skipped** (`/tmp/qe_main9.log`); `quant_evaluator/metrics/registry_adapters.py`, `registry/metrics.py`
- `FP-P0-9/P0-11` → TransformRegistry already fail-closed on defaults (omitted admission ⇒ UNVERIFIED + validate_production rejects; verified by tests); FeatureBundle hardened: ChannelRef type/feature-id/duplicate validation, channels-key⇄channel_name consistency, truthful has_missing/has_freshness flags, wide-layout (T,N,F) shape vs axes+feature-channel → FP suite **710 passed** + adapters **51 passed, 2 skipped** (`/tmp/fp_full_p11b.log`); `factor_preprocess/contracts/feature_bundle.py`, `tests/contracts/test_feature_bundle_hardening.py`
- `FA-P0-14` → residual-IC novelty producer landed (`factor_assets/adapters/residual_novelty.py`): `ResidualICNoveltyProducer` wraps QE `compute_incremental_ic` (optional dep, fail-closed), maps novelty_score = min(1, |residual_ic|/|total_ic|) computed over consistent finite periods (absent total-IC ⇒ **no signal**, never a max score), signals carried on `EvidenceResult.secondary_metrics` with `evidence_result_from_signals`/`novelty_inputs_from_evidence` (corrupt/non-numeric/nonfinite ⇒ absent); scoped review 8 findings (P0 fail-open NaN-denominator, P1 vacuous tests, P2 crash-on-string) all fixed same session → FA suite **637 passed, 43 skipped** (`/tmp/fa_full_residual2.log`); make_decision SHADOW/SHADOWED_COEXIST chain now has a real producer
- `FA-P0-15` → asset-serialization codec bumped to **v2** (FA-P0-12 added validation_status/health_state without bumping — silent format drift): writers emit v2, `_READABLE_CODEC_VERSIONS=(1,2)` keeps v1 DB rows readable with fail-closed UNVALIDATED/ACTIVE defaults, unknown/missing/bool/float versions raise `SchemaVersionError` (wrong kind still `ValueError`); sole consumer sqlite_repository untouched; new `tests/registry/test_serialization_codec.py` (17 tests); scoped review 3×P2 (bool/float version fail-open, stale `build/lib` v1 copy — non-runtime, error-type contract change) → type-check + 2 tests added → FA suite **654 passed, 43 skipped** (`/tmp/fa_full_codec6.log`); note: `factor_assets/build/lib` still carries a stale v1 codec copy — regenerate before any wheel build
- `QE-cache-verify` → cache_v2 dependency-invalidation / absolute-TTL audit: repair **already landed on current tree** — audit of `runtime/cache_v2.py` (2337 lines, full read) confirmed shared `created_at` origin across L1/L2/L3 write-through, `_valid_origin` gates (bool/nonfinite/future-created rejected, parametrized incl. `True`/`False`), fail-closed `CacheMetadata.is_expired`, TTL-exhausted-during-serialization/encoding rejection, transitive dependency-graph invalidation, epoch fences across sibling coordinators/processes, Redis remaining-TTL `ceil`, forged/legacy metadata-key redirection defenses; test file pins all of it (110 tests incl. transitive removal, race-with-promotion, cross-process epoch fencing) → `tests/test_cache_v2.py` **131 passed, 3.72s** (`/tmp/qe_cache_verify.log`); no code change needed — closed as verified, not re-repaired
- `Modeling-ns-verify` → namespace migration validated on current tree: source imports are 100% `modeling_adapters.*` (zero residual `modeling.` old-namespace imports), `pyproject.toml` packages find includes only `modeling_adapters*`; one **pre-existing test failure found and fixed** — `test_contract_hardening.py::test_feature_bundle_values_and_channel_mapping_are_adapted` built a `FeatureBundle` with empty time/asset axes against `(1,1,2)` values, correctly rejected by FP-P0-11's shape hardening; axes now carry 1 timestamp/1 asset → suite **131 passed, 1 skipped** (`/tmp/modeling_full2.log`); wheel clean-install smoke (isolated venv, namespace purity) **1 passed** (`/tmp/modeling_wheel.log`)
- `FA build/lib refresh` → stale git-ignored `factor_assets/build/lib` copy (flagged by FA-P0-15 review as still CODEC_VERSION=1 with old strict-equality reader) regenerated from current source, mirroring the `package-dir = {"factor_assets" = "."}` mapping (all 17 package subdirs + root modules; `__pycache__` pruned) → codec v2 + `_READABLE_CODEC_VERSIONS` confirmed in `build/lib/factor_assets/registry/serialization.py`; FA suite re-run **654 passed, 43 skipped** (`/tmp/fa_full_codec7.log`); **`dist/factor_assets-0.1.0-py3-none-any.whl` still carries the v1 codec** (61 py files, CODEC_VERSION=1, no `_READABLE`) — regenerate the wheel before any distribution
- `FP-P0-12` → transform fail-closed contracts repaired per 2026-08-19 audit: `rolling_zscore` zero lagged-std → NaN not ±inf; `cs_zscore`/`cs_winsor` mask ±inf to NaN before nanmean/nanstd/nanquantile (inf no longer masquerades as 0.0 or NaNs the whole slice); `realized_volatility` zero-variance window → NaN not 0.0; bare `except Exception` in `decomposition/{wavelet,cycle,seasonal}.py` narrowed to `_NUMERICAL_FAILURES` (ValueError/IndexError/LinAlgError/ArithmeticError) so genuine programming bugs propagate; pinned by `tests/transforms/test_fail_closed_contracts.py` (15 tests, incl. exact winsor bounds, -inf, all-inf slice, decomposition-bug-propagation) → FP suite **776 passed, 2 skipped** (`/tmp/fp_full_except_final.log`); scoped review: 3 repairs correct, 6 P2 test-strength findings all addressed same session; changelog note: constant/pegged series now NaN where they previously returned 0.0/±inf
- `FO-P0-16` → sealed-test + score-ingestion hardening (audit P0-1/P1-2/3/4/5): `freeze_for_sealed_test` pins `sealed_test_masks` (train/validation/test bool lists, serialized + validated + replayed via guarded `_sealing` path in `from_dict`; frozen checkpoint lacking them rejected); `consume_sealed_test` rejects a plan whose masks differ from the frozen ones and an EvaluationProtocol bound to different masks (split_id alone no longer suffices); `_sealed_test_disjoint` length-equality check (no zip truncation); boolean evaluation scores rejected at ingestion (True→1.0 previously produced un-deserializable checkpoints); `update_best` rejects bool/NaN/±inf (NaN previously locked the incumbent forever); proposal_fn exceptions recorded as FAILED trials in `duplicate_trials` (budget burn visible, survives checkpoint roundtrip); scoped-review P1-1 (in-place dict mutation of pinned masks bypasses `__setattr__`) fixed same session with `MappingProxyType` + tuple values + `from_dict` deep-copy → pinned by `tests/search/test_sealed_test_hardening.py` (15 tests incl. restored-frozen immutability + in-place mutation block + seal-reset-after-consume) → FO suite **460 passed, 4 skipped** (`/tmp/fo_full_p11.log`)
- `FO-P0-17` → checkpoint roundtrip hardening: `from_dict` now validates (1) `frozen_at` must be null or ISO string (rejects bool/int), (2) `sealed_test_consumed` must be bool, (3) each `sealed_test_results` item requires trial_id/split_id/evaluation_ref/frozen_at/test_metrics keys with non-empty strings + valid ISO frozen_at + dict of finite non-boolean metric values, (4) frozen-session sealed_trial_id/sealed_split_id must be non-empty strings; 26 tests (15 original + 11 corruption); full FO suite **471 passed, 4 skipped** (`/tmp/fo_full_roundtrip.log`)
- `FA-P0-18` → bool guard + isfinite hardening: `plateau.py` `generate_neighbors` now skips bool params (True/False no longer perturbed as 1/0 via `isinstance(param_value, bool)` guard); `MinimumICGate`/`MaximumTurnoverGate`/`MinimumCoverageGate` reject bool/NaN/inf thresholds before range checks (added `math.isfinite` + `isinstance(..., bool)` guards); 15 new tests (1 plateau bool-skip + 4 IC NaN/inf/bool/string + 3 turnover NaN/inf/bool + 3 coverage NaN/inf/bool + edge cases); full FA suite **665 passed, 43 skipped** (`/tmp/fa_full_bool.log`); `factor_assets/optimizer/plateau.py`, `factor_assets/selection/gates.py`
- `QE-P0-1` → bool coercion in evaluator: `runtime/evaluator.py` metric ingestion boundary now rejects `bool` before `float()` call — `True`/`False` no longer silently coerce to 1.0/0.0; focused QE suite **183 passed**
- `QE-P0-22` → bare-except narrowing: 6 bare `except Exception` in QE runtime/backends narrowed to specific exception tuples (programming bugs now propagate, I/O/pickling/JSON failures still caught); focused QE suite **183 passed** (parallel_executor + cache_v2 + kernels)
- `FA-P0-19` → lifecycle observer logging: `registry/lifecycle.py` post-commit observer failures now `logging.warning(...)` with `exc_info=True` before being appended to warnings tuple (previously silent string-only); FA suite **665 passed, 43 skipped**
- `FA-P0-20` → wheel rebuild: `dist/factor_assets-0.1.0-py3-none-any.whl` rebuilt from current source (CODEC_VERSION=2, `_READABLE_CODEC_VERSIONS=(1,2)`, observer logging included); wheel ready for distribution
- `FA-P0-21` → secondary_metrics validation: `EvidenceResult.__post_init__` validates str keys + numeric values (bool raises TypeError, non-numeric silently dropped for corrupt records); FA suite **665 passed, 43 skipped**; `factor_assets/novelty/provider.py`

## Primary focus (this session)

| Package | Role | Status |
|---|---|---|
| `quant_evaluator/` | metrics, cache, streaming, evidence | focused tests pass locally; QE-P0-1 bool coercion fixed; broad QE / Redis `NOT_RUN` |
| `factor_optimizer/` | search, sealed-test boundary | **471 tests pass; sealed-test mask-equality + score-ingestion + checkpoint roundtrip hardening (P0-16/P0-17 done)** |
| `factor_assets/` | assembly, selection, packaging | **665 tests pass; P0-2/3/12/13/14/15/18/19/20 contracts+engine+bool-guard+observer-logging+wheel-rebuild landed; not production-certified** |
| `factor_preprocess/` | transforms, registry admission | **773 pass; zero-variance/±inf fail-closed contracts (P0-12 done)** |
| `modeling/` | pre-model packages / wheel smoke | clean wheel smoke 1 pass |

**Known gaps (next):** QE residual-IC novelty producer (make_decision shadow/coexist chain is write-only until `compute_incremental_ic` → `EvidenceResult.secondary_metrics` wiring lands); QE cache_v2 unification (P0-5); metric namespace alignment (P0-7); FP FeatureBundle (P0-11); CODEC_VERSION bump for FA asset serialization.

**Default:** spend most effort here. Small, blocking fixes in `factor_engine/` or `dataaccess/` are OK when platform work truly needs them — keep diffs narrow and say why.

## De-prioritize (unless blocking)

- Large FE/DA loops (Q residency, Polars parity, DA PIT, operator mining, broad FE regression).
- Do not load full FE/DA archives or old Wave-0 todo dumps into context.
- No autonomous `/loop` or `scheduled_tasks`; max 5 todos.

## Active platform pointers (from LOOP)

| Topic | Evidence / tests |
|---|---|
| QE cache v2 | `quant_evaluator/tests/test_cache_v2.py` (131 pass) |
| FO sealed test | `factor_optimizer/tests/search/` (445 pass) |
| FA assembly/selection | `factor_assets/tests/assembly/test_engine.py` (29 pass incl. provenance/hash) |
| FA selection policy | `factor_assets/tests/selection/test_policy.py` |
| FA legacy packaging | `factor_assets/tests/test_legacy_packaging.py` |
| FP registry | `factor_preprocess/tests/registry/test_transforms_registry.py` |
| Modeling wheel | `modeling/tests/test_wheel_clean_install_smoke.py` |

## Next (platform-only)

1. ~~QE cache_v2 dependency invalidation / absolute TTL repair~~ — verified already landed (see QE-cache-verify above); no code change needed.
2. ~~Modeling namespace migration validation~~ — verified green (see Modeling-ns-verify above).
3. ~~FA stale `build/lib` v1 codec copy~~ — regenerated from current source (see FA build/lib refresh); **`dist/*.whl` still v1 — rebuild wheel before any distribution**.
4. Append one-line results to this file or `LOOP_ENGINEERING_STATUS.md` (platform bullets only).

## Review

Scoped reviewer: diff + one evidence path only. No full test files, no parent session history.
