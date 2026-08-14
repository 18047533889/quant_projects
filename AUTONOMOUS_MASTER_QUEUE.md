# Autonomous Master Queue
**Updated:** 2026-08-14 15:09 CST
**Campaign Time:** 0h 00m / 6h 00m

## Queue Statistics
- P0 Tasks: 48 (QUEUED: 48, ACTIVE: 0, REVIEW: 0, CLOSED: 0)
- P1 Tasks: 0 (will be promoted as needed)
- Total Runnable: 48
- Active Workers: 0
- Idle Workers: 8

---

## PRIORITY: P0 - Architecture & I/O Boundary

### ARCH-P0-001: I/O boundary hook not hard-failing
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/core/architecture  
**Risk:** HIGH - violations detected but execution continues  
**Evidence:** Local hook calls checker without strict mode  
**Files:** factor_engine/core/io_boundary_checker.py, hooks/  
**Required:** violations > 0 → non-zero exit; strict enforcement gate  
**Tests:** Inject I/O violation, verify hard fail  
**DoD:** Production gate fails when violations detected  

### ARCH-P0-002: Scanner failure fail-open
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/core/architecture  
**Risk:** HIGH - scan exceptions become "no violations"  
**Evidence:** SyntaxError during scan doesn't fail gate  
**Files:** factor_engine/core/io_boundary_checker.py  
**Required:** scan failure → CHECK_INFRASTRUCTURE_FAILURE  
**Tests:** Inject parse error, verify infrastructure failure  
**DoD:** Gate fails on scanner infrastructure errors  

### ARCH-P0-003: Duplicate I/O boundary authorities
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/core/architecture  
**Risk:** MEDIUM - two independent rule systems  
**Evidence:** Root allowlist + local checker have separate rules  
**Files:** factor_engine/core/io_boundary_checker.py, root policy  
**Required:** Single PhysicalIOAuthorityPolicy with typed exemptions  
**Tests:** Verify single source of truth for all I/O decisions  
**DoD:** All I/O checks consume unified policy  

---

## PRIORITY: P0 - Backend Capability & Contracts

### FE-BE-P0-001: Legacy BackendKind duplicate authority
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends  
**Risk:** HIGH - multiple BackendKind definitions  
**Evidence:** Polars classification module defines own BackendKind  
**Files:** backends/polars/classification.py, backends/contracts/  
**Required:** Single BackendKind/ExecutionKind/CapabilityLevel authority  
**Tests:** Verify no duplicate enums across codebase  
**DoD:** Platform-wide single backend type authority  

### FE-BE-P0-002: BackendCapability vs BackendCapabilityRecord duplication
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends  
**Risk:** MEDIUM - string vs enum execution kind  
**Evidence:** Two capability representations in use  
**Files:** backends/contracts/capability.py  
**Required:** Merge into single typed record  
**Tests:** All consumers use unified type  
**DoD:** Single canonical capability record type  

### FE-BE-P0-003: Production classifier contradicts quality path
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/polars  
**Risk:** HIGH - production=UNSUPPORTED but quality=polars_native_kernel  
**Evidence:** Quality path allows source inspection heuristics  
**Files:** backends/polars/quality_classifier.py, production_router.py  
**Required:** Production uses only explicit PhysicalImplementationSpec  
**Tests:** Verify production never uses heuristic classification  
**DoD:** No heuristic path in production routing  

### FE-BE-P0-004: physical_spec() exception treated as missing
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends  
**Risk:** HIGH - INVALID/ERROR vs MISSING conflated  
**Evidence:** Exception during spec lookup returns None  
**Files:** backends/registry.py, physical_spec.py  
**Required:** Distinguish MISSING / INVALID / INFRASTRUCTURE_ERROR  
**Tests:** Inject spec error, verify typed failure  
**DoD:** Production hard fails on INVALID/INFRASTRUCTURE_ERROR  

### FE-BE-P0-005: is_production_eligible() insufficient validation
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends  
**Risk:** MEDIUM - incomplete production eligibility check  
**Evidence:** Only structural checks, missing contract binding  
**Files:** backends/production_classifier.py  
**Required:** Bind semantic/param/null/implementation/parity evidence  
**Tests:** Verify all eligibility dimensions checked  
**DoD:** Rename if only structural, or complete full validation  

### FE-BE-P0-006: q backend not in main capability authority
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends  
**Risk:** MEDIUM - q capability separate from main system  
**Evidence:** q exists in enum but not in registry/planner/cost  
**Files:** backends/q/, backends/registry.py, planner/  
**Required:** q integrated into unified backend system  
**Tests:** Verify q participates in capability/cost/telemetry  
**DoD:** q treated equivalently to Pandas/Polars/DuckDB  

### FE-BE-P0-007: SQL emitter hash failure returns "unknown"
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/duckdb  
**Risk:** HIGH - hash unavailable → "unknown" → continue  
**Evidence:** SQL hash failure doesn't fail production  
**Files:** backends/duckdb/sql_emitter.py, evidence/  
**Required:** Typed infrastructure error on hash failure  
**Tests:** Inject hash error, verify typed exception  
**DoD:** Production fails on hash unavailability  

---

## PRIORITY: P0 - q Backend

### Q-P0-001: _PHASE1_NATIVE_OPS manual list not capability authority
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q  
**Risk:** HIGH - hardcoded list doesn't reflect reality  
**Evidence:** Manual list can diverge from actual capabilities  
**Files:** backends/q/capability.py  
**Required:** Auto-compute from Declared/Lowering/Compile/Runtime/Parity  
**Tests:** Verify capability derived from actual evidence  
**DoD:** Production capability = intersection of all passes  

### Q-P0-002: Generate Q_NATIVE_WITHOUT_LOWERING difference set
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q  
**Risk:** HIGH - native declaration without lowering implementation  
**Evidence:** No validation that native ops have lowerings  
**Files:** backends/q/capability.py, lowering_registry.py  
**Required:** Q_NATIVE_WITHOUT_LOWERING = 0 gate  
**Tests:** Compute set difference, verify empty  
**DoD:** Hard gate fails if native without lowering  

### Q-P0-003: Generic lambda shadows special lowerings
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q/compiler  
**Risk:** HIGH - special lowerings unreachable  
**Evidence:** ts_beta, wma, clip may hit generic branch first  
**Files:** backends/q/compiler.py, lowering_registry.py  
**Required:** Explicit Q_LOWERING_REGISTRY with priority  
**Tests:** Verify special lowerings actually used  
**DoD:** Each canonical op has exactly one reachable lowering  

### Q-P0-004: ts_quantile declaration vs lowering alignment
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q  
**Risk:** MEDIUM - capability claim without reachable lowering  
**Evidence:** Quantile ops declared but lowering may not exist  
**Files:** backends/q/capability.py, lowering/  
**Required:** Q_CAPABILITY_IMPLIES_REACHABLE_LOWERING gate  
**Tests:** For each capability, verify lowering reachable  
**DoD:** All declared capabilities have proven lowerings  

### Q-P0-005: Rolling corr/cov/beta recertification
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q/rolling  
**Risk:** HIGH - incorrect window semantics  
**Evidence:** Need proof of point-in-time correctness  
**Files:** backends/q/rolling_ops.py  
**Required:** Prove Corr_t(x,y;W), Cov_t, β_t correct per timestep  
**Tests:** Compare against Pandas oracle with explicit windows  
**DoD:** Mathematical parity evidence for rolling stats  

### Q-P0-006: Unified rolling wrapper contract
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q/rolling  
**Risk:** MEDIUM - inconsistent rolling parameters  
**Evidence:** window/min_periods/warmup/ddof/null handling varies  
**Files:** backends/q/rolling_wrapper.py  
**Required:** Freeze contract: window/min_periods/ddof/null/NaN/Inf/ordering/group/session  
**Tests:** Verify all rolling ops respect unified contract  
**DoD:** Single rolling contract enforced across all ops  

### Q-P0-007: Lag/rolling instrument boundary leakage
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q/rolling  
**Risk:** HIGH - cross-instrument data leakage  
**Evidence:** No explicit test for instrument boundaries  
**Files:** backends/q/rolling_ops.py, lag.py  
**Required:** Proof that stock A last row doesn't leak to stock B first row  
**Tests:** Concatenate two instruments, verify boundary isolation  
**DoD:** Parity test proves instrument boundary correctness  

### Q-P0-008: QBackend accepts full logical plan instead of PhysicalBackendRegion
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q  
**Risk:** MEDIUM - bypasses unified planner  
**Evidence:** QBackend takes whole plan, not region  
**Files:** backends/q/backend.py  
**Required:** QBackend only accepts PhysicalBackendRegion  
**Tests:** Verify planner produces regions for q  
**DoD:** QBackend integrated into region execution model  

### Q-P0-009: Missing data returns empty DataFrame instead of typed error
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q  
**Risk:** HIGH - silent failure on data unavailability  
**Evidence:** Base data missing returns empty result  
**Files:** backends/q/data_adapter.py  
**Required:** Raise DataUnavailableError  
**Tests:** Mock missing data, verify typed exception  
**DoD:** Production fails visibly on data unavailability  

### Q-P0-010: Output schema mismatch guesses first column/empty Series
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q  
**Risk:** HIGH - schema mismatch silently produces wrong output  
**Evidence:** Unexpected schema causes fallback guessing  
**Files:** backends/q/output_adapter.py  
**Required:** QOutputContract with keys/value/dtype/row_count/grain/ordering/nullability  
**Tests:** Return wrong schema, verify hard fail  
**DoD:** Schema mismatch fails in production  

### Q-P0-011: Global q singleton has research→production pollution risk
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q  
**Risk:** HIGH - configuration pollution across run modes  
**Evidence:** First call with fallback affects later production calls  
**Files:** backends/q/connection.py  
**Required:** Immutable-config keyed instances or no global singleton  
**Tests:** Research call then production call, verify isolation  
**DoD:** Production cannot be polluted by earlier research config  

### Q-P0-012: object NULL vs empty string/symbol merged
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q  
**Risk:** MEDIUM - NULL conflated with empty string  
**Evidence:** fillna("") used as generic conversion  
**Files:** backends/q/type_conversion.py  
**Required:** Preserve NULL distinct from ""  
**Tests:** NULL vs "" must stay distinct through pipeline  
**DoD:** Production never conflates NULL with empty string  

### Q-P0-013: Short/int/long null sentinel clarity
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q  
**Risk:** MEDIUM - integer null handling unclear  
**Evidence:** Integer types have distinct null sentinels  
**Files:** backends/q/type_system.py  
**Required:** Document and test each integer type's null handling  
**Tests:** Verify short/int/long null round-trip  
**DoD:** Explicit contract for integer null semantics  

### Q-P0-014: Nullable boolean three-state preservation
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q  
**Risk:** MEDIUM - boolean NULL may become False  
**Evidence:** Nullable boolean handling not explicit  
**Files:** backends/q/type_system.py  
**Required:** Preserve True / False / NULL distinctly  
**Tests:** Boolean NULL round-trip through q  
**DoD:** Three-state boolean contract enforced  

### Q-P0-015: Zero-copy fallback catches too broadly
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_engine/backends/q  
**Risk:** MEDIUM - semantic errors masked by fallback  
**Evidence:** Second toq() attempt catches all exceptions  
**Files:** backends/q/zero_copy.py  
**Required:** Only catch explicit zero-copy errors  
**Tests:** Inject semantic error, verify not masked  
**DoD:** Semantic/type errors propagate, not caught by fallback  

---

## PRIORITY: P0 - DataAccess Identity & Layout

### DA-ID-P0-002: Audit hash_cache_key callers for correctness migration
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** data_access/identity  
**Risk:** HIGH - cache key used where correctness identity needed  
**Evidence:** hash_cache_key used in result reuse contexts  
**Files:** data_access/identity/hash.py, callers  
**Required:** Migrate correctness-affecting calls to hash_correctness_identity  
**Tests:** Audit all callers, verify correctness vs performance classification  
**DoD:** No correctness decision uses cache key  

### DA-ID-P0-005: Production uses hash_identity(bits=64) directly
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** data_access/identity  
**Risk:** HIGH - 64-bit hash insufficient for correctness  
**Evidence:** Production code calls hash_identity with 64-bit  
**Files:** data_access/identity/hash.py  
**Required:** Ban production correctness from bits=64  
**Tests:** Scan for hash_identity(bits=64) in production paths  
**DoD:** Production uses full-width hash for correctness  

### DA-LAYOUT-P0-002: Legacy raw bucket API bypasses validated policy
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** data_access/layout  
**Risk:** HIGH - production can bypass validation  
**Evidence:** Raw bucket API callable in production  
**Files:** data_access/layout/bucket.py  
**Required:** Production only uses *_from_policy  
**Tests:** Verify production paths only call validated API  
**DoD:** Raw bucket API inaccessible from production  

---

## PRIORITY: P0 - DataAccess Build & Versioning

### DA-BUILD-P0-001: Artifact version/build revision not frozen at build time
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** data_access/build  
**Risk:** MEDIUM - version determined at runtime  
**Evidence:** Version looked up dynamically  
**Files:** data_access/__init__.py, setup.py  
**Required:** Version frozen during wheel build  
**Tests:** Install wheel, verify version without git  
**DoD:** Installed package has frozen version  

### DA-BUILD-P0-002: Production depends on runtime git rev-parse
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** data_access/build  
**Risk:** HIGH - production requires source tree  
**Evidence:** Code calls git rev-parse at runtime  
**Files:** data_access/version.py  
**Required:** Build identity frozen at wheel creation  
**Tests:** Install wheel outside source tree, verify works  
**DoD:** No git dependency in installed production code  

### DA-BUILD-P0-003: External revision needs strict format validation
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** data_access/build  
**Risk:** MEDIUM - revision injection or malformation  
**Evidence:** External revision accepted without validation  
**Files:** data_access/version.py  
**Required:** Strict regex validation for revision format  
**Tests:** Inject malformed revision, verify rejection  
**DoD:** Only valid git-like revisions accepted  

### DA-BUILD-P0-004: DataSnapshotIdentity mixed with ExecutionBuildIdentity
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** data_access/identity  
**Risk:** MEDIUM - data change conflated with code change  
**Evidence:** Single identity for both snapshot and build  
**Files:** data_access/identity/snapshot.py, build_identity.py  
**Required:** Separate identities, both in provenance  
**Tests:** Code change alone doesn't invalidate data cache  
**DoD:** Data and build identities independent  

---

## PRIORITY: P0 - FactorAssets

### FA-P0-001: DataAccess adapter import names incorrect
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_assets/adapters  
**Risk:** HIGH - import mismatch with published API  
**Evidence:** Adapter uses internal imports not in public contract  
**Files:** factor_assets/adapters/data_access_adapter.py  
**Required:** Use only public DataAccess API imports  
**Tests:** Import from clean environment, verify works  
**DoD:** All imports match published API  

### FA-P0-003: check_factor_availability accepts handle as True
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_assets  
**Risk:** HIGH - availability check doesn't validate  
**Evidence:** Getting handle returns True without materialization  
**Files:** factor_assets/availability.py  
**Required:** Materialize and validate with snapshot/PIT identity  
**Tests:** Mock unavailable data, verify returns False  
**DoD:** Availability proven by actual materialization  

### FA-P0-005: FactorAssets redefines FactorEngine identity
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_assets  
**Risk:** HIGH - duplicate identity authority  
**Evidence:** FactorAssets has own factor identity definition  
**Files:** factor_assets/identity.py  
**Required:** Consume FactorIdentityArtifact from FactorEngine  
**Tests:** Verify single source of factor identity  
**DoD:** FactorAssets uses FactorEngine identity exclusively  

---

## PRIORITY: P0 - Campaign & Contamination

### CAMP-P0-001: max_duration_s not enforced by monotonic clock
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_assets/campaign  
**Risk:** MEDIUM - time limit bypassable  
**Evidence:** Duration check doesn't use monotonic time  
**Files:** factor_assets/campaign/runner.py  
**Required:** time.monotonic() for duration enforcement  
**Tests:** Verify duration limit cannot be bypassed  
**DoD:** Campaign stops at true wall-clock limit  

### LEDGER-P0-001: seal_test_splits doesn't truly enforce
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** factor_assets/ledger  
**Risk:** HIGH - test split contamination not prevented  
**Evidence:** seal writes metadata but doesn't block access  
**Files:** factor_assets/ledger/split_seal.py  
**Required:** Sealed splits inaccessible to adaptive processes  
**Tests:** Try to access sealed split, verify blocked  
**DoD:** Test split truly inaccessible after seal  

---

## PRIORITY: P0 - Modeling Temporal Leakage

### MODEL-P0-001: SplitSpec.gap_days defined but not enforced
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** modeling  
**Risk:** HIGH - temporal leakage via missing gap  
**Evidence:** gap_days field exists but no enforcement  
**Files:** modeling/split.py  
**Required:** Enforce gap_days between train_end and val_start  
**Tests:** gap_days=5, verify 5-day gap enforced  
**DoD:** Gap enforcement proven by test  

### MODEL-P0-002: Label interval purge not implemented
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** modeling  
**Risk:** HIGH - train data overlaps label interval  
**Evidence:** train_end <= val_start insufficient for multi-day labels  
**Files:** modeling/split.py, label.py  
**Required:** Purge train data within label forward window  
**Tests:** 5-day label, verify last 5 days of train purged  
**DoD:** Label interval purge implementation and test  

### MODEL-P0-003: Explicit Embargo contract missing
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** modeling  
**Risk:** HIGH - no standard embargo mechanism  
**Evidence:** No embargo contract found  
**Files:** modeling/  
**Required:** EmbargoSpec with clear semantics  
**Tests:** Embargo period enforced in walk-forward  
**DoD:** Embargo contract implemented and tested  

### MODEL-P0-005: FittedTransform apply_start_time=None OOS bypass
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** modeling  
**Risk:** HIGH - out-of-sample leakage  
**Evidence:** Public transform allows None start time  
**Files:** modeling/transform.py  
**Required:** Public OOS requires ApplicationWindow  
**Tests:** Call with None, verify rejection  
**DoD:** OOS transform requires explicit window  

### MODEL-P0-010: ModelArtifact identity incomplete binding
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** modeling  
**Risk:** HIGH - model not reproducible from identity  
**Evidence:** Identity missing snapshot/universe/semantic/seed  
**Files:** modeling/artifact.py  
**Required:** Bind feature/preprocess/split/label/snapshot/universe/catalog/hyperparams/seed/build/versions  
**Tests:** Reconstruct model from artifact, verify deterministic  
**DoD:** Complete identity binding proven  

---

## PRIORITY: P0 - QuantEvaluator

### QE-Q-P0-001: assign_quantiles method not truly participates in tie policy
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** quant_evaluator  
**Risk:** MEDIUM - tie handling unclear  
**Evidence:** method parameter doesn't clearly affect ties  
**Files:** quant_evaluator/quantiles.py  
**Required:** QuantileTiePolicy contract  
**Tests:** Heavy ties, verify method affects assignment  
**DoD:** Explicit tie policy contract and test  

### QE-Q-P0-002: NumPy vs Numba boundary equality parity uncertain
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** quant_evaluator  
**Risk:** HIGH - tie boundary differences  
**Evidence:** searchsorted(side="right") vs val > boundary may differ  
**Files:** quant_evaluator/quantiles.py, numba_quantiles.py  
**Required:** Exact parity test for boundary ties  
**Tests:** Boundary-equal values, verify identical assignment  
**DoD:** NumPy-Numba tie parity proven  

### QE-Q-P0-004: Numba fastmath=True vs NaN/Inf correctness
**Status:** QUEUED  
**Owner:** (unassigned)  
**Module:** quant_evaluator  
**Risk:** HIGH - fastmath may violate NaN/Inf semantics  
**Evidence:** fastmath=True used without correctness proof  
**Files:** quant_evaluator/numba_kernels.py  
**Required:** Parity proof for NaN/Inf or disable fastmath  
**Tests:** NaN/Inf inputs, verify identical results  
**DoD:** Fastmath proven safe or disabled  

---

## Next Actions

**Immediate (next 5 minutes):**
1. Launch IndependentReviewer to assess prior session commits
2. Launch Worker-Backend for FE-BE-P0-001 through P0-007
3. Launch Worker-Q for Q-P0-001 through Q-P0-003
4. Launch Worker-DA-Architecture for ARCH-P0-001 through P0-003
5. Create AGENT_STATUS.md and FINDINGS_LEDGER.md

**Within 30 minutes:**
6. Launch remaining workers for their assigned P0 tasks
7. Launch first AuditMiner to discover additional issues
8. Update queue with discovered tasks

**Auto-replenish trigger:**
- If runnable < 15: launch AuditMiner
- If runnable < 8: launch second AuditMiner
