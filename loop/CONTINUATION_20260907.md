# V3 continuation checkpoint (2026-09-07)

## MOST CURRENT: identity/codec/replay integration pass

Root metadata fix: FrozenMapping storage replacement/deletion blocked; generic
object metadata arrays and structured/object numpy scalars fail closed. AxisRef
object coordinates retain detached public reads and separate scalar-only path.
Codec preserves int/string/tuple map keys and escapes reserved codec-tag-shaped
user maps. Hash canonicalizer distinguishes literal map tags from typed values,
rejects non-string top field keys and structured dtype field-schema loss.
Root files metric_artifacts.py,_ndarray_codec.py,_hashutil.py,factor_batch.py and
test_v3_metadata_ownership.py (15cases). Focused29passed.
Agent final stable fullQE797passed2skipped4warnings112.32sec, source manifests
before/after identical. Log /tmp/v3_optimizer_qe_final_manifested_20260907.log
sha256 e3d59fa9ea32f9cceb76b08e324a7587e405eaa8c52485b412ff995093412690.
Source manifest digest edc1941204cb1e4c0bd59d53fdcca6ae3b45d70cb14ea55050701cd64067cc3c.
Root MUST NOT upload stale QE runtime/registry/etc; optimizer now implementing
one real QE03 dependency/resource admission gap, not just metadata.

QE08 agent added formal dedicated CUDA kernel plus separately registered
daily_quantile_monotonicity_series and daily_quantile_monotonicity_rate.
RealL20 formal longrun1 vs daily0.625 (25up/15down), explicit emptycounts.
u_shape_score still rejected pre-device; no all-shape GPU support claim.

Root FAIdentityProvider string parser bug: ensure_expr('close') made a literal.
Now delegates actual FE parse_expr with explicit surface/dialect/version; full
identity parses once. FE DSL close resolves catalog FieldRef StockDailyBarAdj.close,
intentionally DIFFERENT from raw col('close'); tests reflect canonical field
rather than collapsing dataset semantics. Expr-only path remains supported.
Owned factor_assets/adapters/factor_engine.py, tests/test_adapter_factor_engine.py,
tests/adapters/test_factor_identity_dsl_v3.py. FullFA1326passed43skipped46warnings,
/tmp/quant_v3_fa_dsl_identity_full.log (after prior1317metadatafull).
Old misidentified string artifacts must be reevaluated under new canonical IDs,
never relabel old literal objects or mutate their history. Root ledger FA06 update pending.

MOD07 preprocess real PCR train_model->artifact->prediction->QE job now checks
actual OOS FeatureSchema against model, temporal finalfit/OOS separation, and ID
hash binds actual OOS X/predictions/context/label hashes (root found initial ID
omitted values and copied training schema). Fullmodeling404pass after this.

E2E-G assets added optional model manifest compatibility edge, hardened on root
review: immutable FeatureSetVersion inputs, recompute schema hashes, reject spoof
same refs, explicit intended deployed/replacement model, typed evidence and jobs
resolver checks model->exactfeature binding.34focusedpass. Manual pairwise fixtures
are NOT actual QE evidence: fullE2E-G stillPARTIAL. Own modeling_manifest_bridge.py
and jobs/e2e_g_cluster_model_compatibility.py + tests; root copies stale.
Assets NOW real QE->FA metric grade provenance adapter/tests, not touching root FE adapter.

E2E-E preprocess outer jobs revision composition initially flawed; root required
coordinate normalization/exact coverage, warmup, full-vs-incremental checks,
removal of fabricated cluster refs, and retaining TRUE days_since_update after
fill expiry. Latest fix reads revised raw tail to locate next original observation,
requires DA plan cover age influence (or rejects), transforms warmup..plan end,
patches declared outputs. Long gap finiteage after unusablefill parity passed.
Fullmodeling406passed3warnings; focused6passed. FeatureSchema deliberately research-
incomplete, no fake complete cluster/recipe refs. FullE2E-E still lacks DSL and
all requested cross-domain trace; NOT production certification.

Root OPS08 planner now inventories ALL graph keys/values and orphan evidence even
when artifact_kinds omitted; conflicting same-ID evidence versions rejected.
Owned app/ops08_migration.py and tests/test_ops08_migration.py. Focused running
session84669 (poll if needed). Planner still dry-run; executor/history chain absent.
Temporary PG still STOPPED; last platform439allpass preceded assets bridge tests
and root OPS08 inventory tests, so rerun realPG after platform sources stable.
Root local ledgers/builder need latest797/FA1326/MOD406 summaries and uploads.

## NEWEST: atomic inbox and crash-safe outbox (supersedes below)

Full platform **439 passed,0 skipped**, real isolated PostgreSQL16.15,
`evidence/r2/v3_platform_atomic_inbox_20260907.log`. Temporary PG stopped after run.
Inbox claim/handler/DONE now one transaction, handler failure uses savepoint and
durable FAILED metadata without effects. BaseException rolls back backend tx.
Real subprocess os._exit before/after DB handler write proves no stranded claim.
Actual PostgreSQL duplicate SQL error recovers savepoint; simultaneous consumers
run handler once. Replica no longer writes PROCESSING. Legacy durable PROCESSING
rows deliberately need controlled operator recovery (cannot know old live worker).
Handlers must share connection, not commit/nest transactions; external effects
need outbox/idempotency, not a universal exactly-once promise.
Outbox recover_expired_claims fences old tokens; publish_pending honors stored
backoff and recovers expired claims (default300seconds). Real subprocess death
after delivery but before ack replays through Inbox without repeated DB effect.
Same worker ID cannot steal existing claim. No external exactly-once claim.
Public production_composition now passes dialect facade to migrations (raw
psycopg previously broke) and accepts ordinary SQLite file paths. Shared schema
creation delegates actual PG dialect/sequence path.
Root newly owns composition.py, db/sqlite_backend.py and tests/test_composition.py,
tests/test_outbox_claim.py,test_generation_coordinator.py in addition to prior files.

Stable full QE after root codec and agent calendar/adaptive:780passed2skipped,
`/tmp/v3_optimizer_qe_full_20260907.log`, sha256
a49584489a6597e41bbf1f4fe0154989f29d129fb089df2f3f7d1a97e72f2b7f.
QAT07 session18214 finished exit0:8wheels and combined imports succeed.
ModelPredictionArtifact now truly bytes-backed immutable + strict refs/time axes;
403modeling pass/11focused. Preprocess agent now actual training→QE public chain.
Optimizer now implementing one real GPU shape gap; root MUST NOT change QE runtime.
Assets now assigned E2E-G actual research cluster→deployed model compatibility.
Root ledger reconciled QE18/QE35/MOD07 and PL08; builder PL08 stale override removed,
builder reads limitations as remaining. Upload these local ledger/builder files.
All E2E-A–H and GOLD40 still PARTIAL; component traces are not scenario closure.

## MOST RECENT: real PostgreSQL and SIM05 (read FIRST)

Root started ISOLATED PostgreSQL16.15 in `/tmp/quant-v3-pg-8VsqEo` on server-c.
No TCP listener (listen_addresses=''), UNIX socket same dir, port55439.
Data `/tmp/quant-v3-pg-8VsqEo/data`; server cleanly STOPPED after425pass. Restart
only for additional isolated tests; stop using `/tmp/quant-v3-pg-8VsqEo/root/usr/lib/postgresql/16/bin/pg_ctl -D
/tmp/quant-v3-pg-8VsqEo/data stop -m fast` with
LD_LIBRARY_PATH=/tmp/quant-v3-pg-8VsqEo/root/usr/lib/x86_64-linux-gnu.
DBDSN: `dbname=quant_platform_test host=/tmp/quant-v3-pg-8VsqEo port=55439 user=sunhaiwei`.
Pinned psycopg2-binary2.9.10 isolated target `/tmp/quant-v3-pg-8VsqEo/driver`.
New optional quant-platform[postgres-test] pins that exact driver; default wheel
still no runtime deps. No main .venv install or production DB connection.
Full platform real PostgreSQL now **425 passed,0 skipped**,
`/tmp/quant_v3_platform_live_full.log` (tool91938 exited255 SSH but remote log
shows full completion and no remaining pytest process; verified read-only).

Root changes now OWNED (don't overwrite remote stale mirrors):
- quant_platform/app/db/{schema,postgres_backend,migrations}.py,
  quant_platform/app/outbox.py, storage/generation_coordinator.py;
  pyproject.toml, tests/test_postgres_{live,dialect}.py.
- Real PG first run7fail exposed epoch-number/calendarTIMESTAMP mismatch,
  bool0/1 SQL, missing sequence-backed surrogate IDs/execute_returning facade,
  tuple-vs-SQLite-row shape, duplicate finish true; fixed actual callers.
- Inbox simultaneous first-delivery independent2connections reproduced two
  handlers; rowcount ownership now guards initial/received/retry claims.
- Event outbox/inbox clocks DOUBLE PRECISION Unixseconds (other calendar
  columns unchanged). Actual migration builder requires explicit legacy_zone;
  converts timestamps preserving fractional epoch and restores sequences above
  max IDs. Reuses existing apply_migrations (fixed wrong _conn=>SQLite detection,
  version table binding, custom migration list, swallowed query/commit errors).
  Live test proves migration/replay and old ID42->next>42.
- Live tests isolated per test; cleanup exact TABLE_NAMES (not prefix guesses).
  Old sent-row test incorrectly demanded redelivery; now proves no redelivery
  and independent pending new event delivery. Table count31 stale removed in
  favor of exact schema inventory. No-driver unit test explicitly simulates it.
- Remaining Inbox crash after durable PROCESSING claim before handler remains
  possible; no automatic timeout/fencing recovery implemented. Do not claim
  general exactly-once distributed side effects or complete E2E crash closure.

MOD07 binding moved from newly authored platform adapter into
`jobs/evaluate_model_predictions.py` to respect BOTH independent modeling and
pure-DTO platform boundaries. Test import updated, full platform passes unchanged
boundary guard. Root ledger/older checkpoint path needs correcting accordingly.

SIM05 implemented in factor_assets/clustering/families.py + __init__.py export:
explicit ClusterQualityPolicy(no universal default); actual observed medoid,
signed cycle conflicts (balanced negatives allowed), weak-edge components;
missing star edges INSUFFICIENT (not zero). Production _leiden_finalize requires
PASS every cluster; research keeps explicit quality status. Existing small-size
unstable_clusters semantics preserved (NOT mixed with quality failures).
ClusterArtifact new quality evidence hashed with certification+mode; JSON
roundtrip retains both. Old quality-empty artifact hash retained.
Actual igraph1.0.0 installed ONLY `/tmp/quant-v3-leiden-iQTAlg` via PYTHONPATH.
65cluster tests pass, extended7quality pass, final34quality/certification pass.
FullFA1315pass43skip `/tmp/quant_v3_fa_quality_full.log` with SIM08 25tests; seen
index standalone15pass and full no stall. New source after fullFA only quality
serialization/provenance covered by final34focus; fullFA source snapshot varies.

Agents latest:
- optimizer completed mixedQ CPU+real L20 CUDA adaptive profiles. QE bundle
  factor_artifacts append-only, canonical custom policy JSON request transport,
  CPU planning feasibility explicitly recorded, actual GPU grouped daily/full
  strict kernels,774QEpass2skip. Now approved calendar/rolling CUDA implementation
  (evaluator.py,gpu_executor.py,kernels/gpu/drawdown.py,test_v3_calendar_public.py).
  ALL root QE local mirrors touched by that agent STALE; don't upload them.
- preprocess completed SIM01/SIM08 and V18. SIM01 after root review now actual
  FE compute_block_content_hash binds values/masks/axes/context despite arbitrary
  bundle_id and constant embedding. SIM08 aggregates medoid+topk support, leaves
  aggregateCI unknown with per-support refs/intervals; includes medoid count.
  V18 existing orchestrator/EventEnvelope/modeling manifest full identity trace
  under configured production path,28focus; ledger INTEGRATION_PENDING. Root
  fixed its discovered platform adapter boundary failure via jobs move.
- assets idle/unable further implementation; no source ownership changed.


## LATEST RESUMED CHECKPOINT (supersedes older sections)

Root FA06 real FE->DA->FA taxonomy adapter implemented and verified 49 passed.
`factor_assets/adapters/production_taxonomy.py` requires exact explicit usage-role
bindings, resolves DA canonical IDs and aliases, keeps unknown semantics unknown,
rejects unknown fields. Only alpha domains derive economic mechanism tags.
Taxonomy policy2.0.0 current;1.0.0 retained; mechanism_sources appended after old
dataclass fields to preserve positional callers. Tests cover compatibility.
Preprocess agent owns adapters/__init__.py and agreed export taxonomy wrapper.

Root MOD07 public composition added:
`quant_platform/app/adapters/model_prediction_evaluation.py` binds QE port into
existing `modeling/evaluation.py::evaluate_predictions`. Typed prediction_id,
values and rows must match explicit T/security axes. QE prediction bundle and
independent probe-PnL bundle separated; legacy label spread/member Jaccard marked
research diagnostics, never execution/OOS certified. Independent model package
must NOT import QE directly (packaging test); numerical port sits in platform.
Prediction IC/series use QE; ICIR uses QE summary kernel with min_periods2.
Modeling full395pass (/tmp/quant_v3_modeling_mod07.log), no failures.
MOD07 remains PARTIAL: training callers need typed refs/adapter migration;
importance vs OOS ablation binding and real execution-domain adapter incomplete.

Root ledger adds FA06/MOD07 as entries in v3_qe_ledger_merge_20260907.json.
Unified builder now reads targeted_tests and remaining (previously omitted).
Regenerated177 ledger validation PASS. No commits/push; diff check clean.

Agent optimizer finished SIM02 signed per-window HAC uncertainty and overlap
gates,47focus/1317FA+pairwise pass43skips. Currently adaptive bins QE33/35:
owns adaptive_bins_policy.py, shape_evidence.py, registry/metrics.py,
test_v3_shape_contracts.py (root local copies STALE).45focusedpass, fullQE running.
Brief source SyntaxError during upload fixed; root modeling395pass afterward.
Agent told update own ledger and send root precise new status so root-last ledger
doesn't override improvements with old PARTIAL description.

Agent preprocess currently SIM01 real FP FeatureBundle+FA profile+embedding
fingerprint producer, explicit materialization/profile refs and actual public
incremental wrapper. Do not overwrite adapters/__init__.py, fingerprint files.

Agent assets read-only triage regenerated ledger then FAILED to implement SIM05
(no files changed). Remaining concrete local gaps identified:
- SIM05 Leiden families.py quality report/gates for medoid affinity, signed-cycle
  conflicts, bridge heterogeneity. Existing ClusterArtifact chooses lexicographic
  representative; no cluster quality evidence beyond graph certification.
- SIM08 cluster-level multiple-neighbor support (current pairwise uncertainty
  gate does not aggregate cluster medoid/top-k support).
- QAT04 remaining manual mutant tests, though optimizer already killed3 actual
  source mutations (empty gates, FP lag bridge, real CUDA mask) earlier.
- V18 complete request/trial/definition/recipe/value/evaluation/health/verdict/
  library/feature trace still absent. GOLD40/E2E-A-H all PARTIAL.

Earlier post-W10 root work not in older sections: unknown-gap DD/Calmar CPU/CUDA;
GPU absolute per-cohort held-quote mask prevents opposing net-zero holdings
hiding missing quote; shape inverse-Fisher/LOO stability, mirror asymmetry,
descriptive rank bootstrap, actual disjoint temporal windows and builder params.
QE32 horizon agent completed public CPU/CUDA common mature sample wrapper:
api/horizons.py, api/__init__.py and top-level exports are agent-owned/stale local.
Nearzero tolerance1e-12 consistent CPU/CUDA; fullQE767pass2skip before adaptive.


## NEWEST checkpoint (read before older sections below)

Root QE W8 =724 passed/2skipped; W9 =736 passed/2skipped
(`/tmp/quant_v3_qe_w8.log`, `/tmp/quant_v3_qe_w9.log`). Older W6=708/2.
No commits/push/production mutations. Main source mirror still work/v3-root.

Newly implemented/uploaded root changes:
- Side-specific TradeEligibilityPanel (buy/sell/borrow/cover, availability clocks,
  explicit axes, universe/source refs) applied CPU/GPU entry selection and exit
  fail-closed when permission denied. Public test_v3_trade_eligibility.py.
- Removed stale GPU probe_ls branch consuming labels.
- Public artifacts and bundle carry RESEARCH_PROBE/execution_certified=False,
  permissions hash and side-specific vs unrestricted assumption.
- Explicit Sortino MAR (periodic), downside_denominator negative/all,
  annualization sqrt_frequency/none; Calmar arithmetic/cagr. Registry v2.
- GPU Sortino fixed missing rows with nonzero target (did subtract target on
  padded rows). test_v3_risk_variants.py independent expected and real CUDA.
- GPU prebuilt ProbePortfolioArtifact support via session.stage_context,
  tiled factor-id column projection; real CPU/CUDA test passes.
- Full immutable numeric ownership: _freeze_array bytes-backed arrays prevent
  setflags(write=True); FactorBatch snapshot values/validity+FrozenMapping refs;
  EvaluationRequest freezes nested params/context/metadata. Object dtype axes
  still only conventional readonly; explicit ImmutableBufferRef research escape
  remains. Initial writable memmap is copied ONCE then GPU tiles share snapshot;
  test_public_gpu_tiling expectation updated accordingly, no per-tile copy.
- EvaluationBundle lossless to_dict/from_dict with explicit artifact-kind allow
  list, counts/dtypes/TQF masks preserved; request codec frozen mappings encoded.
- Shared request-local IC series cache for derived siblings and observation counts;
  two methods build twice total (spearman/pearson) test. Base direct IC may still
  recompute. Basic tier/node cost admission, pre-allocation parameter name guards.
- MetricValue.sample_unit and artifact observation_counts/sample_unit recorded;
  rolling counts full valid windows, calendar counts included periods. Legacy
  scalar fallback finite_metric_value still needs per-definition upgrade.
- Agent preprocess new metrics/calendar_returns.py (root must not overwrite stale
  copy): authoritative DA CalendarSnapshot lazy import, worst_calendar_month/
  quarter/year -> ScalarMetricArtifact with period_rows/completeness/counts;
  actual dates require timezone-aware datetimes. Partial policy exclude/include.
  compute_worst_rolling_return now(F,) worst FULL window only; min_periods is
  min valid window COUNT, not permission to use prefix. 8 direct tests.
- Root wired calendar_snapshot runtime field in requests/evaluate, registry three
  calendar and worst_rolling_21d/63d/252d IDs, typed raw ScalarArtifact handling,
  actual ProbePortfolioArtifact clock/factor binding. Public tests2pass. CUDA
  calendar/rolling still unsupported and fails late; must fix admission/support.
- Root main QE ledger now evidence/r2/v3_qe_ledger_merge_20260907.json32entries
  (local same filename under work/v3-root), honest FIXED_LOCAL/PARTIAL only.

CURRENT UNFINISHED TEST:
session35167 /tmp/quant_v3_unknown_risk.log focused risk/coreGPU.
Latest root edits uploaded AFTER W9: maxdrawdown default missing_return_policy
now unknown (legacy explicit zero_fill/fail). Missing gap masks subsequent DD;
scalar NaN unless observed bankruptcy gives known100% loss. Calmar unknown gap
invalid, explicit zero_fill supported CPU/GPU. risk variant missing-input parity
test now explicitly zero_fill. New independent unknown/bankruptcy tests and
effective_definition provenance. Poll/fix then full QE W10.

Agents currently:
- v3_optimizer running remaining23 legacy FE filter_layer failures exposed by
  duplicate test rename. Pairwise/same_clock_lag fixed,108pass23fail last.
  QA tools run_qat04_targeted_mutations.py 3killed real CUDA; run_qat07_import_matrix.py
  8wheels true imports PASS; production FE optional extra added. Duplicate tests
  renamed preserves both bodies; mirrors root tests. QAT06 old GC URI failures
  fixed by preprocess. Do not let helper-only/package count certify GOLD/E2E.
- v3_preprocess NOW DTA08/10 implementation approved: new DA sole missing-reason
  plane after authority search; FP FeatureBundle carries original reason/mask/age,
  modeling manifest refs and train usable coverage. DTA10 extend existing
  DataChangeSet snapshot/watermark ids if missing and test true DA→FE recompute.
  Avoid FE filter_layer/QE runtime files. They finished REAL durable GC authority:
  platform db schema/migrations/root epoch/tombstone/receipt, generation_coordinator
  atomic stage/publish/root protection vs GC and real2connection race/crash tests.
  409platform pass10PGskip, races20repeats pass, no production deletes.
- v3_assets currently completed/idle after failing twice to implement FA09
  (reports execution slice unable, no source changes). Previously finished:
  MOD06 ACTUAL orchestrator bridge public modeling_feature_manifests() carries
  explicit value/recipe/state/cluster refs only; missing=>research partial
  complete=false/model_ready=false, no fabricated fallback. Removed second
  FeatureSetPointer authority. OPS08 dryrun migration uses FA invalidation +
  modeling dependency DAG, unchanged branches retained, no production writes.
  Unified177 ledger builder/validator now includes root+agent sources at
  evidence/r2/v3_177_unified_ledger.json. 3adversarial validator tests. They
  identified FA09 typed QE pairwise status→FA actual consumer as top remaining.
  Agent files code new paths in quant_platform/app/ops08_migration.py and
  modeling_manifest_bridge.py; tests. Root may need take FA09 or delegate to
  optimizer after FE completion.

Important remaining main: QE26 full coherent NAV/risk chain (ongoing); calendar
GPU support/admission; full artifact/builder DAG/context/residual exposures;
statistics agent owns artifact_types.py & registry_adapters.py NEVER overwrite
root stale copies; QE32-35 current behavior audit; metric-version migration
and real historical public chains; strict production ndarray axes ownership;
unified ledger still many open local issues. Continue implementation, no final
all-done claim. User requested whole177, not partial P0. Latest commentary
reported calendar public wiring and model explicit partial readiness.

User still requires the entire 177-issue V3 task, GOLD40/E2E8. Do not stop at
partial package work or call locally implementable missing APIs external blockers.

All edits target server-c `/home/sunhaiwei/quant_projects_v3_remediation`, branch
`codex/v3-remediation-20260906`, base `2db45f4e446309003381a51a5e03071d31859502`.
Original main untouched. SSH control path `/tmp/quant-v3-server-c.sock`.
Python `/home/sunhaiwei/quant_projects/.venv/bin/python`. Use package-isolated
pytest with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=2 and
PYTHONPATH pointing to V3 worktree (FO/FP nested package roots when required).

## Main changes since previous summary

- W1 labels/axes: full QE 648 passed/2 skipped (`/tmp/quant_v3_qe_w1.log`).
- Risk W2: initial NAV shared, bankruptcy zero NAV is absorbing 100% loss,
  negative-capital returns <-1 rejected CPU/GPU. Empty Sharpe proper shape;
  empty drawdown shape/sentinel fixed. Cost turnovers reject negative/NaN/Inf.
  `drawdown_events` one-pass with peak/start/trough/recovery/end/censored/status,
  valuation gap -> INVALID_VALUATION, no recovery guessed across it.
  Underwater no longer drops dates. Recovery only completed events.
  Return skew defaults adjusted Fisher-Pearson, bias variant explicit.
  Registry risk/recovery/skew/rolling IDs version2.0.0. Full QE671/2 skipped.
- W3 facade: EvaluationRequest metric_parameters, portfolio_returns(runtime
  ProbePortfolioArtifact), holding_returns(runtime HoldingReturnPanel),
  portfolio_spec(serializable PortfolioSpec). API parameters truly bind; cached
  repeated evaluator unbinds prior parameter closure before rebinding.
  Metadata parameters included in planner node. Nested FrozenMapping now encoded
  by `_ndarray_codec` Mapping rather than dict-only.
  EvaluationBundle.artifacts retains Scalar/Series/Vector and now DailyQuantile
  artifacts; no implicit time/profile mean. scalar MetricValues grouped by true
  F; registry versions/config_hash genuine, not hardcoded. Columnar
  scalar_metrics/series_metrics/vector_metrics compatibility properties exist.
  GPU returns same EvaluationBundle, scalar observation counts obtained on GPU.
  GPU QF profile now Q=5 time-mean with min-period guard; previous raw TQF under
  `quantile_returns` renamed to canonical `quantile_returns_full` QF.
  GPU IC min assets from metric signature / ICSeries builder default20, min
  periods from signature, zero variance ICIR NaN rather than epsilon infinity.
  GPU effective params accepted via executor.metric_parameters; aliases mapped.
  Full W3 675 pass,1 fail unknown metric KeyError type, fixed to RuntimeError.
- W4 cohort: added `contracts/portfolio_schedule.py`: entry=s+1,
  scheduled_exit=s+1+H, first earned return at s+2. H=post-entry intervals.
  CPU/GPU fixed boundary, two cost events, no forced tail liquidation;
  outputs scheduled_exit/matured. Held missing quote returns NaN not zero.
  Both sides must meet post-tradability bucket minimum (unless weight0).
  Long/short diagnostic kernel uses canonical tie-policy searchsorted disjoint
  sets; shared 2D bool mask works with3D factor; future missing labels cannot
  shift selection thresholds; strict shape/bool/threshold order.
  Added independent price-ledger tests test_v3_cohort_intervals.py.
  W4 full 681pass/3fail/2skip, three stale old semantics assertions in
  tests/metrics/test_probe_cohort_metrics.py updated to post-entry and unknown
  valuation. Prior GPU hand-example similarly updated with explicit rationale.
- Added `contracts/portfolio_inputs.py`: HoldingReturnPanel.from_prices
  (bound explicit time/security axes/source/price basis, immutable) distinct
  from labels; frozen PortfolioSpec RESEARCH_PROBE-only, H/Q/weights/cost.
  CPU facade builds one probe PnL per factor from holding panel; GPU session
  separately stages holding returns and tile executor reuses portfolio_pnl
  once across risk metrics. No forward labels used for these public metrics.
  CPU/GPU supported holding->sharpe/sortino/win/maxdrawdown/calmar public path.
  Registry maxdrawdown/calmar now have compute_fn (previously bare metadata).
  Independent holding/price public parity tests pass; 29 focused holdings/core
  tests passed `/tmp/quant_v3_holdings.log`.
- Daily TQF builder from optimizer agent now registered as
  `quantile_returns_daily`; runtime preserves typed artifact and binds
  config_hash/split_ref using dataclasses.replace. GPU daily still pending.
- Main 177 ledger updated11 QE issues to FIXED_LOCAL with explicit remaining
  work. File local work/v3-root/v3_remediation_ledger.json; remote loop same.
  Do not claim whole issue VERIFIED until public chains/migrations covered.

## Current test/tool sessions

- session56079: focused tests test_v3_public_artifacts,
  test_v3_cohort_intervals,test_daily_quantile_artifact_v3 (poll).
- session71829 is prior W4 full suite completed as above.
- Root has not run full QE after holdings public wiring + daily artifact yet.
- No commits/push. Global diff check reported5 trailing whitespace lines in
  evaluator; removed and uploaded, recheck later.

## Agent ownership / current assignments (all GPT-5.6 Sol)

### v3_optimizer
FO full807 passed. Durable SQLite campaign budgets/leases/test authority,
alias-safe holds, immutable completion replay, real stage trace, finite-grid
TRAIN U-center/tail cutoff freeze, FDR hypothesis ledger, diagnostic split
guards, retention near-zero/sign reversal and horizon safeguards. FO CUDA real
3factor path confirmed typed new QE bundle.
Ledger: evidence/r2/v3_optimizer_ledger_merge_20260907.json (27 entries), YAML
evidence/r2/v3_optimizer_20260906.yaml.
NOW delegated non-overlapping QE stats work, owns:
  quant_evaluator/contracts/artifact_types.py
  quant_evaluator/metrics/registry_adapters.py
  new contracts/statistical_evidence.py, metrics/statistical_evidence.py
  daily/statistical tests.
Main MUST NOT upload local stale copies of these files.
DailyQuantileReturnArtifact already in remote artifact_types.py around229.
Builder `build_daily_quantile_return_artifact(batch,label,n_quantiles=5,
min_assets=10,min_periods=20,*,tie_status_ref=None,tradability_ref=None,
risk_exposure_ref=None,producer_version='1.0.0')` returns trueTQF/count/mask.
Asked agent to fix bool coercion, axes uniqueness, and add ClassVar/property
artifact_kind='daily_quantile'; certification should require resolver not
arbitrary nonempty strings. Agent now implements typed stats by reusing existing
HAC/bootstrap/DSR/PBO/retention/decay/regime kernels, not duplicate math.

### v3_preprocess
FP full288passed/1xfail. Includes auxiliary channels, conservative output
properties, real FE multi-step PlanNode one panel load, production custom
operator guard/domain validation, roundtrip fitted state. DTA adapter trueTN
pivot, symbol dtype preserved, UpdateTime required, duplicate coordinates
fail, schema errors not disguised as optional-dep errors. DA tests74+49passed.
FE authorized files cleaned_bridge.py and tests.
MOD01/05/08 initial implemented, full modeling377passed. Full ordered feature
manifest, artifact/context identity, correct final train+validation preprocess
refit, finite campaign monitoring request with maturity + rollback bindings.
Ledger evidence/r2/v3_preprocess_ledger_20260907.json.
NOW authorized to continue MOD02/03/04 in modeling/diagnostics.py, selection.py,
trainer_governance + tests: integrate existing FO budget authority, consumer
profile noninferiority and production policy requirements. No main overlap.
DTA05 remains main QE: no existing TradeEligibilityPanel/side-specific trade
ExecutionSpec anywhere; need can_buy/sell/borrow/cover typed panel and actual
execution/probe distinction. Existing FE ExecutionContract is not trade domain.

### v3_assets
FA full1279passed/43skip. Strict health/provenance/canonical gates/frozen rules,
missing-required partial, shape-family branches, typed promotion provenance
and metric versions; pure `plan_metric_version_invalidation` planner.
FA taxonomy roles, fingerprint full semantics, ANN identity/near-duplicate
recall and batch index reuse, medoid/final-cluster checks.
Ledgers evidence/r2/v3_assets_177_ledger.json, v3_platform_ledger.json.
PL01-06 local44focused pass: normalization failure identity, success-only
consumed state, typed failure classes, no RankIC shortcut, distinct refs,
cumulative snapshots. PL07 actual bytes hash/size in research store.
NOW explicitly tasked to IMPLEMENT remaining PL07 durable publication and
PL08 transactional outbox/generation publisher in approved quant_platform
orchestrator/contracts/durable/outbox/materializer files; fault injection
fixture tests. Do not stop pending. Then DTA04 dated universe resolution in
factor_assets/adapters/data_access.py+assembly to use real UniverseSnapshot;
also MOD06 immutable FeatureSet cluster-version binding.

## Remaining main priorities / known defects

- GPU old probe_ls branch remains stale and unsafe source (get_metric rejects
  unregistered IDs currently); remove/replace with canonical new holding path,
  never let it consume staged labels.
- GPU turnover vs membership still same kernel; monotonicity old daily-positive
  semantics not CPU long-term shape. Must fix QE07/08 with real goldens.
- GPU parameter validation happens after staging and has no complete normalized
  plan; cost_budget/tier still metadata-only. QE03 fake _FACADE_ARTIFACT_TYPES
  requirements availability still needs removal and real builders.
- CPU public `_to_per_factor_array` now obsolete dead code with old averaging;
  remove after call search. IC counts recompute with defaults and derived
  scalars fallback1: actual min_assets/counts/provenance must match request;
  shared IC builder caching still needed.
- Serialization `EvaluationBundle` own roundtrip remains absent; typed artifacts
  roundtrip works. Strict request types/refs/production axis requirements and
  ndarray setflags ownership still incomplete.
- New dailyTQF CPU registered; GPU needs TQF/counts output and same artifact.
- Portfolio docs/GPU top comments contain stale pre-entry formula; update.
  New RESEARCH_PROBE purpose should be carried on bundle and artifacts, not
  only internal portfolio source. Explicit TradeEligibilityPanel and pricing/
  universe snapshots needed; do not claim executable/capacity certification.
- Sortino/Calmar variants still only legacy defaults; calendar worst-period
  misleading names, downside policy, risk state/artifact chain incomplete.
- Numerical metric versions for cohort/long_short changed need2.0.0 and
  historical invalidation integration; prior root only risk/skew versions done.
- Need full177 ledger merge with agents, GOLD40/E2E8 evidence, QAT mutation,
  independent wheel imports/package matrix, per-issue requirement mapping,
  true capability metadata, immutable history + rollback tests.
- Main read STA4712-5061 and QAT6464-6696 fully in this continuation;
  other unreviewed V/OPS/traceability sections still need reading.

Latest commentary: independent holding tests pass29; price panel notH10 labels;
parallel work advancing model manifest/bounded retrain/retry publication.
