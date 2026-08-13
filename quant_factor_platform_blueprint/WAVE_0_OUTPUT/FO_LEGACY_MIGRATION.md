# FactorOptimizer Legacy Migration Audit

Scope: read-only mining of legacy code under `/home/shw/quant_projects`; decisions follow `05_FACTOR_OPTIMIZER_SPEC.md` and `21_LEGACY_FUNCTION_MINING_CHECKLIST.md`. Taxonomy: `REUSE_AFTER_TEST`, `REWRITE`, `REFERENCE_ONLY`, `CORPUS_ONLY`, `DISCARD`.

## files found

The requested broad `find` also matched many report HTML files and unrelated registries. The FO-relevant exact Python paths are:

- `/home/shw/quant_projects/toolkit/registry.py`
- `/home/shw/quant_projects/toolkit/alpha_tools/registry.py`
- `/home/shw/quant_projects/toolkit/alpha_tools/generated_library.py`
- `/home/shw/quant_projects/factor_layer/factor_agent/core/agent_loop.py`
- `/home/shw/quant_projects/factor_layer/factor_agent/core/autonomous.py`
- `/home/shw/quant_projects/factor_layer/factor_agent/core/state.py`
- `/home/shw/quant_projects/factor_layer/factor_agent/tools/registry.py`
- `/home/shw/quant_projects/factor_layer/factor_agent/skills/registry.py`
- `/home/shw/quant_projects/factor_layer/factor_agent/verifiers/evaluator.py`
- `/home/shw/quant_projects/factor_layer/factor_agent/verifiers/audit.py`
- `/home/shw/quant_projects/factor_layer/factor_agent/planning/sub_agents/base.py`
- `/home/shw/quant_projects/factor_layer/factor_admission/admission.py`
- `/home/shw/quant_projects/factor_layer/factor_admission/catalog.py`
- `/home/shw/quant_projects/factor_layer/factor_admission/config.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/integrations/quant_platform.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/models.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/registry.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/router.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/compute.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/compute_engine.py`
- `/home/shw/quant_projects/factor_engine/mining/campaign.py`
- `/home/shw/quant_projects/factor_engine/mining/direct_use.py`
- `/home/shw/quant_projects/factor_engine/mining/operator_catalog.py`
- `/home/shw/quant_projects/factor_engine/research_tools/registry.py`
- `/home/shw/quant_projects/factor_engine/research_tools/runtime.py`
- `/home/shw/quant_projects/factor_engine/research_operators/__init__.py`

No `/home/shw/quant_projects/factor_layer/factor_pool/` directory exists. No legacy `MutationSpec` or `CandidateSpec` was found; the only relevant `TransformSpec` is in `toolkit/registry.py`.

### Gateway per-file survey

Canonical implementations live under `gateway/scripts/`; top-level and `gateway/gateway/` files are mostly compatibility re-exports.

| Exact file | Summary | Decision |
|---|---|---|
| `gateway/scripts/complexity.py` | Regex operator extraction, character bracket depth, report DTO; advertised AST parser is unimplemented | REWRITE |
| `gateway/scripts/config.py` | Loads operator whitelist, future blacklist and complexity weights from legacy config | REFERENCE_ONLY |
| `gateway/scripts/deduplicator.py` | Lowercase/whitespace hashes, bounded cache, persistence; semantic/vector paths are stubs | REWRITE |
| `gateway/scripts/future_scanner.py` | Regex blacklist scanner; AST path is not authoritative | DISCARD |
| `gateway/scripts/validator.py` | Candidate schema plus regex operator/field/type validators; AST path incomplete | REWRITE schema only; DISCARD operator validation |
| `gateway/scripts/models.py` | Candidate/gateway/result/manifest DTOs and serialization | REFERENCE_ONLY |
| `gateway/scripts/gateway_core.py` | Seven-step fail-fast pipeline: schema, YAML, dedup, tiny run, LLM future scan, complexity, DQ | REFERENCE_ONLY |
| `gateway/gateway/gateway_core.py` | Older object-oriented composition of validators/scanner/complexity/dedup | REFERENCE_ONLY |
| `gateway/scripts/data_quality.py` | Drift/liquidity/clock checks and direct data loading | DISCARD |
| `gateway/gateway/data_quality/runner.py` | DQ compatibility runner | DISCARD |
| `gateway/gateway/data_quality/checker.py` | Re-export wrapper | DISCARD |
| `gateway/scripts/io_utils.py` | Candidate filesystem IO, routing, hashes, markdown report | DISCARD |
| `gateway/scripts/router.py` | Filesystem scan/move loop | DISCARD |
| `gateway/scripts/kafka_producer.py` | Routing message builder and Kafka producer | DISCARD |
| `gateway/scripts/main.py` | CLI and campaign filesystem loop | REFERENCE_ONLY |
| `gateway/scripts/exceptions.py` | Legacy exception taxonomy | REFERENCE_ONLY |
| `gateway/scripts/logger.py` | Empty | DISCARD |
| `gateway/scripts/tests/test_complexity.py` | Tests current regex behavior and thresholds | CORPUS_ONLY |
| `gateway/scripts/tests/test_deduplicator.py` | Hash/cache/persistence fixtures | CORPUS_ONLY |
| `gateway/scripts/tests/test_future_scanner.py` | Regex future-function cases | CORPUS_ONLY |
| `gateway/scripts/tests/test_gateway_core.py` | Pass/reject/duplicate/batch orchestration cases | CORPUS_ONLY |
| `gateway/scripts/tests/test_validator.py` | Minimal validator cases | CORPUS_ONLY |
| `gateway/scripts/tests/test_io_utils.py`, `test_kafka_producer.py` | IO/Kafka fixtures | DISCARD |
| `gateway/{complexity,config,data_quality,deduplicator,future_scanner,io_utils,kafka_producer,models,validator}.py` | Compatibility imports | DISCARD |
| `gateway/gateway/{complexity,config,deduplicator,future_scanner,io_utils,kafka_producer,validator}.py` | Compatibility imports | DISCARD |
| package `__init__.py` files | Package exports | DISCARD |

## function-level migration matrix

| File | Symbol | Signature | Decision | Target FO module | Reason | Required tests |
|---|---|---|---|---|---|---|
| `toolkit/registry.py` | `TransformSpec` | dataclass `(name, default_parameters, description, when_to_use, leakage_notes, exposed_to_agent)` | REUSE_AFTER_TEST | `factor_optimizer.mutations.spec` | Good frozen metadata and exposure/leakage fields; extend to full MutationSpec, do not import legacy | immutability; schema round-trip; required fields; no kernel ownership |
| same | `get_transform_spec`, `is_allowed_transform` | `(name)`, `(name, *, exposed_only=True)` | REFERENCE_ONLY | `factor_optimizer.mutations.registry` | Useful lookup/exposure semantics; global env-dependent registry is unsuitable | unknown name; exposure tier; deterministic snapshot |
| `toolkit/alpha_tools/registry.py` | `SEED_TOOL_SPECS` | metadata mapping | REUSE_AFTER_TEST | `factor_optimizer.mutations.spec` | Input/return contracts, typed parameters, bounds, defaults, version and source are useful concepts | bound edges; bool-vs-int; defaults; relational bounds |
| same | `AlphaToolsFacade` | `__init__(functions)` | REFERENCE_ONLY | `factor_optimizer.mutations.registry` | Read-only facade idea; FO exposes specs and FE references, not duplicate callables | mutation rejected; stable listing |
| same | `validate_tool_parameter_value` | `(tool_name, parameter_name, value, bound_parameters=None)` | REWRITE | `factor_optimizer.mutations.validation` | Only int/float/min/max/max_ref and repeated global lookup | all kinds; enums; NaN; relational bounds; defaults |
| same | `_with_hashes` | `(specs, functions)` | REWRITE | `factor_optimizer.mutations.identity` | Source hash is useful provenance, but `inspect.getsource`/`repr` and truncation are not canonical identity | formatting/process stability; version changes; unavailable source |
| `toolkit/alpha_tools/generated_library.py` | all 11 functions | e.g. `directional_volatility_ratio(df, window=20, min_periods=5)` | DISCARD | none | Duplicates FE rolling, ATR, correlation, z-score, drawdown and related kernels | boundary audit; FE operator references resolve |
| `gateway/scripts/complexity.py` | `ComplexityReport` | dataclass report fields | REUSE_AFTER_TEST | `factor_optimizer.complexity.models` | Sound report concept; add AST depth, lookback, stateful/CS/nonlinear/interactions/domains/sources/latency | serialization; totals; budget boundary; FE parity |
| same | `ComplexityEvaluator.evaluate` | `(expr) -> (float, bool, ComplexityReport)` | REWRITE | `factor_optimizer.complexity.evaluator` | Regex calls and raw parentheses; AST parser branch is `pass` | nested/malformed AST; aliases; string literals; FE parity |
| same | `_calculate_operator_weights`, `_calculate_max_depth` | `(expr)` | DISCARD | none | Regex and bracket counting cannot be truth | false-positive/negative regression corpus |
| same | `suggest_optimization` | `(expr) -> list[str]` | REFERENCE_ONLY | `factor_optimizer.policy.repair` | Generic heuristics, not diagnosis/evidence-driven | diagnosis mapping; unsupported mutation rejection |
| `gateway/scripts/models.py` | `Candidate` | dataclass schema/ID/expr/config/timestamps/campaign/batch/stages | REFERENCE_ONLY | `factor_optimizer.candidates.models` | Preserve IDs, timestamps, campaign/batch and stage evidence ideas; missing lineage and trial fields | lineage round-trip; parent requirement; immutable origin; versions |
| same | `GatewaySegment`, `GatewayResult`, `Manifest` | DTOs + serialization | REFERENCE_ONLY | `factor_optimizer.search.events` | Auditable stage results useful; fixed metric columns should become EvidenceRefs | lossless serialization; version rejection; evidence refs |
| `gateway/scripts/deduplicator.py` | `ExpressionNormalizer.normalize` | `(expr) -> str` | DISCARD | none | Lowercase/whitespace removal is not DSL canonicalization | poison cases as negative corpus |
| same | `HashDeduplicator` | hash/add/check/load/save/clear/size | REWRITE | `factor_optimizer.seen` | Exact seen-cache concept useful; eviction, silent persistence and raw-string identity are flawed | concurrency; true eviction; atomic persistence; corruption; AST keys |
| same | `DuplicateRecord`, `DeduplicationResult` | dataclasses | REFERENCE_ONLY | `factor_optimizer.seen.models` | Matched ID, report ref, score and match type are useful | exact-vs-structural distinction; serialization |
| same | `SemanticDeduplicator` | `is_duplicate`, `register_factor`, history methods | DISCARD | none | Config equivalence and vector DB are stubs; no valid novelty logic | no semantic-equivalence claim from hash/correlation |
| `gateway/scripts/validator.py` | `SchemaValidator` | `validate(candidate_dict)` | REWRITE | `factor_optimizer.candidates.validation` | Required-field/version pattern useful; timestamps and legacy schema are weak | strict timestamp/version/types/lineage |
| same | `OperatorValidator`, `DataTypeValidator`, `StaticValidator` | expression validators | DISCARD | none | Duplicate FE parser/operator/type/domain authority; regex-heavy | FE adapter fake; invalid operator/domain; boundary audit |
| `gateway/scripts/gateway_core.py` | `run_gateway` | `(manifest, *, campaign_config, init_result, dedup_cache, market_data_root, cache_dir)` | REFERENCE_ONLY | `factor_optimizer.search.runner` | Ordered cheap-to-expensive gates and fail-fast records are useful; implementation couples IO, FE, LLM and DQ | gate order; halt; budget; complete trail |
| same | `_step3_dedup`, `_step6_complexity`, `_calc_nesting_depth` | step helpers | REWRITE | `factor_optimizer.seen`, `.complexity` | Replace strings/brackets with FE canonical AST/IR | FE identity/depth parity |
| same | `_step4_tiny_run` | `(manifest)` | REFERENCE_ONLY | `factor_optimizer.adapters.factor_engine` | L0/L1 idea useful; execution belongs behind protocol | fake protocol; tiny budget; errors |
| `factor_engine/mining/campaign.py` | `_stable_value`, `_expr_payload`, `candidate_semantic_hash` | `(value)`, `(expr)`, `(Factor|Expr)->str` | REUSE_AFTER_TEST via adapter | `factor_optimizer.adapters.factor_engine` | Best AST/parameter-normalized structural identity; alias canonicalization and sorted mappings | aliases; kwarg order; arg order; nonfinite; FE versions |
| same | `compiler_generation` | `() -> str` | REUSE_AFTER_TEST via adapter | same | Binds field/operator catalogs for cache invalidation | catalog changes; deterministic digest |
| same | `NegativeCompileCache`, key/entry DTOs | `get`, `record`, `len` | REUSE_AFTER_TEST | `factor_optimizer.search.cache` | Thread-safe and keyed by candidate plus compiler generation | races; invalidation; deterministic errors only |
| same | `DependencySignature`, `dependency_signature`, `group_source_first` | compiled candidate grouping | REUSE_AFTER_TEST via adapter | `factor_optimizer.search.batch` | Batch-first grouping by sources/fields/history/universe | stable grouping; source/history differences; order independence |
| same | `MiningCampaignSnapshot`, `MiningCampaignSession` | frozen snapshot and batch compile/run/evaluate | REFERENCE_ONLY | `factor_optimizer.search.session` | Snapshot pinning/shared work useful; use protocols, not FE internals | pinned IDs; batch; cancellation/budget; cache |
| `factor_engine/mining/operator_catalog.py` | `MiningOperator`, `AdmissionQuery`, `get_mining_operators` | typed FE catalog | REUSE_AFTER_TEST via adapter | `factor_optimizer.adapters.factor_engine` | Authoritative availability, roles, costs, types and sources; never copy metadata | consistency audit; domain/source rejection; version binding |
| `factor_engine/mining/direct_use.py` | `DirectUseContract`, `InputSlotSpec`, resolver/query helpers | operator metadata | REUSE_AFTER_TEST via adapter | same | Already-migrated eligibility and parameter-role semantics | parameter split; causal/timing gates; no mirror |
| `factor_engine/research_tools/registry.py` | `ResearchToolRegistry` | `register_moved`, `get`, `list_canonical`, `catalog` | REFERENCE_ONLY | `factor_optimizer.adapters.factor_engine` | Good research-vs-DSL separation; not a mutation registry | never production-exposed; backend selection |
| `factor_agent/tools/registry.py` | `register`, `run`, `get_api_tools` | tool metadata/dispatch | REFERENCE_ONLY | `factor_optimizer.llm.tools` | Structured tool schema concept; mutable globals, file tools and string errors are unsafe | allowlist; typed errors; malformed input; no arbitrary code/files |
| `factor_agent/skills/registry.py` | `format_layer1_for_system`, `get_skill_content` | layered prompt content | REFERENCE_ONLY | `factor_optimizer.llm.prompts` | Context layering useful; hard-coded text/imports need versioning | prompt hash; unknown skill; deterministic assembly |
| `factor_agent/core/agent_loop.py` | `agent_loop` | `(query, create_message, *, system, max_rounds, max_tokens, model)` | REFERENCE_ONLY | `factor_optimizer.llm.researcher` | Structured tools, max rounds, verifier stop and audit hooks; reimplement around CandidateMutation | budgets; invalid tool; replay; structured output; no Python code |
| `factor_agent/verifiers/evaluator.py` | `run_eval_script`, `is_passed` | subprocess evaluator/boolean | DISCARD | none | Fixed score threshold and file/subprocess coupling conflict with EvaluatorProtocol/evidence vectors | protocol replacements; timeout/error mapping |
| `factor_admission/admission.py` | `_check_thresholds`, `admit_evaluation_run` | fixed thresholds/persistence | REFERENCE_ONLY | `factor_optimizer.search.admission` | Diagnostics/policy snapshot useful; scalar thresholds duplicate asset admission | snapshot; missing evidence; Pareto/hard gates; no delete |
| `assetization/scripts/registry.py` | `generate_factor_id`, `extract_coordinates` | `(coordinates, seed=None)`, `(config)` | REFERENCE_ONLY | `factor_optimizer.candidates.identity` | Stable seed identity and coordinates useful; final IDs belong to FactorAssets | idempotence; validation; collision |
| `assetization/scripts/models.py` | `FactorCandidate` | candidate wrapper/properties | REFERENCE_ONLY | `factor_optimizer.candidates.models` | Source metadata and gateway-pass concepts only | malformed payload; evidence mapping |
| `integrations/quant_platform.py` | validation/execution/materialization helpers | FE bridge functions | DISCARD for FO runtime | none | Legacy bootstrap, DataSource and materialization duplicate official adapter boundary | package boundary; no legacy runtime import |

## registry patterns worth reusing

1. Frozen metadata, explicit versions, typed parameter bounds/defaults, input/output contracts, provenance and LLM exposure flags from `toolkit`.
2. Read-only facade behavior and deterministic sorted listing, reimplemented without mutable globals or import-time environment state.
3. Callable/source hash only as provenance, not semantic identity; bind it to spec version and use a full canonical digest.
4. FE `MiningOperator`/`DirectUseContract` catalogs through an adapter are authoritative for role, type, source, cost and production eligibility.
5. `ResearchToolRegistry` demonstrates the required separation between research utilities and production DSL operators.

## complexity evaluation legacy

Gateway complexity is definitively regex-based. `OPERATOR_PATTERN` extracts text shaped like calls and `_calculate_max_depth` counts literal parentheses. `use_ast_parser=True` reaches only a commented import and `pass`; the second gateway core repeats bracket depth. The evaluator and weights are not reusable truth.

FO should retain the report/budget concept but obtain canonical facts from `FactorExecutorProtocol.complexity`: AST depth, weighted operator count, lookback, stateful/CS/nonlinear operators, interactions, domains/source count and estimated/observed latency. Complexity remains one Pareto/budget dimension, not a universal scalar utility.

## candidate metadata worth reusing

Useful fields are `candidate_id`, schema version, born timestamp, `campaign_id`, `batch_id`, expression/config, stage result, run ID, checked time, decision reason and historical evidence/report reference. Add mandatory FO fields absent in legacy: `origin`, parent(s), mutation spec/version and parameters, lineage depth, trial ID, hypothesis, expected falsifiable signatures, model ID, prompt version, FE validation ref, QE result ref and budget consumed. Lineage must be immutable and losslessly serializable.

## dedup/seen cache patterns

Keep separate exact candidate identity and seen-record metadata, compiler-generation invalidation, matched candidate/evidence refs, bounded caches and persisted history. Prefer FE `candidate_semantic_hash`, which serializes parsed AST, canonical aliases and sorted kwargs while preserving argument order. Rewrite persistence atomically and fail closed on corruption; make concurrency explicit.

Do not reuse lowercase/whitespace normalization, raw config JSON as semantic identity, the claimed FIFO/LRU implementation, silent `except Exception`, or stub vector similarity. Structural equality is not algebraic equivalence. Novelty must not collapse to `corr > threshold -> delete`.

## agent-loop orchestration patterns

Reference only: bounded rounds, structured tools, assistant/tool-result replay, pre-action hooks, verifier stop, subagent isolation and final audit logging. Reimplement as a clean state machine around `CandidateMutation`, `FactorExecutorProtocol`, `EvaluatorProtocol` and `ResearchLedgerProtocol`. Enforce budgets at every transition, allow only registered MutationSpecs, and never let an LLM submit arbitrary Python or declare success from prose.

Gateway ordered fail-fast stages are also useful as an L0-to-L1 pattern, but old data IO, future scanner, complexity, evaluator subprocess and routing must not be imported.

## LLM integration legacy (if any)

- `factor_agent/config/settings.py` hard-codes model `claude-sonnet-4-20250514`; `agent_loop` passes model/system/messages/tools/max tokens to an injected client.
- `factor_agent` contains system text, layered skill prompts, subagent prompts, context compression and tool schemas. It records messages/final state, but not first-class prompt version, hypothesis, expected signatures or trial ID.
- Gateway `scripts/gateway_core.py` imports `utils.deepseek_client.deepseek_chat` for Step 5 future scanning. Discard this LLM validator; FE validation must be deterministic and authoritative.
- No complete hypothesis ledger was found. FO must write model, prompt version/hash, falsifiable hypothesis, expected signatures, trial ID and tool decisions through `ResearchLedgerProtocol`.

## parameters canonicalization legacy

The strongest implementation is FE `mining/campaign.py`: `_stable_value` recursively canonicalizes scalar/list/mapping values, sorts mapping keys and rejects unsupported values; `_expr_payload` canonicalizes operator aliases; `candidate_semantic_hash` uses compact sorted JSON with `allow_nan=False`. Gateway only strips whitespace, lowercases expressions and sorts config keys. Alpha registry stores defaults/bounds but does not bind omitted defaults into identity.

No reliable sign normalization was found. No commutative operand sorting or algebraic canonicalization exists. FO should delegate syntax/alias/default binding to FE canonical AST/IR, preserve operand order unless FE proves equivalence, normalize parameter types/defaults under the referenced MutationSpec version, and treat sign changes as explicit mutations.

## things to DISCARD (duplicate FE operator logic)

- All implementations in `toolkit/alpha_tools/generated_library.py`: volatility ratios, volume variability, price-volume correlation/agreement, rolling robust z-score, ATR, intraday location/return ratios, range ratio and drawdown depth. They duplicate FE kernels.
- Gateway operator whitelist, regex operator extraction, field classification, datatype validator and future scanner. FE owns legality, semantic types, causality and data availability.
- Gateway regex/bracket complexity as truth.
- Gateway/assetization direct data IO, in-memory source, FE bootstrap, execution, materialization, filesystem router, Kafka routing and DQ. FO uses protocols/adapters.
- Factor-agent subprocess evaluator and fixed score threshold. QE owns metrics/evidence.
- Stub semantic vector dedup/config equivalence and silent persistence recovery.
- Research operators/tools as production mutations; they stay research-only unless promoted through FE governance.
