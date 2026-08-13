# Modeling Layer Documentation Audit

**Date**: 2026-08-13  
**Scope**: `/home/shw/quant_projects/factor_engine/modeling/`  
**Total Lines**: 8,372  

## Executive Summary

The modeling layer has **strong foundational documentation** with comprehensive module-level docstrings linking to taskbook sections, but **lacks critical operational documentation** for:

1. **Temporal contracts and leakage prevention rules** (scattered across code, not consolidated)
2. **Hard gates documentation** (existence/status not documented at package level)
3. **End-to-end lifecycle examples** (no quickstart/cookbook)
4. **Integration patterns** with factor engine (wiring not documented)

**Overall Grade**: B+ (Strong technical foundation, incomplete operational guidance)

---

## Per-File Assessment

### 1. **contracts.py** — Grade: A-

**Strengths**:
- Excellent module docstring: comprehensive catalog of all contract types with taskbook references
- 12/16 classes documented (75%)
- Rich temporal/contract density: 96 references to temporal concepts
- All major contracts (SampleAdequacyContract, LabelContract, DecisionClock) well-documented

**Gaps**:
- 4 undocumented classes: `SessionPhase`, `TradingTimestamp`, `PredictionOutputContract`, `PredictionBatch`
- Only 4/19 public functions documented (21%)
- Missing: **integration examples** showing how to compose contracts for a real model
- Missing: **decision tree** for choosing between execution classes

**Critical Missing Documentation**:
```python
# NEEDS: Example showing complete contract composition
"""
Example: Building a complete predictive model contract

    from modeling.contracts import (
        ModelExecutionClass, SampleAdequacyContract, 
        LabelContract, DecisionClock, ashare_decision_clock
    )
    
    # 1. Choose execution class
    exec_class = ModelExecutionClass.PREDICTIVE_SUPERVISED
    
    # 2. Define label contract (VWAP-to-VWAP, 1-bar horizon)
    label = LabelContract(
        label_name="ret_1d", 
        horizon_bars=1,
        return_basis="vwap_to_vwap"
    )
    
    # 3. Define decision clock (after-close execution)
    clock = ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP)
    
    # 4. Define sample adequacy
    sample = SampleAdequacyContract(
        min_raw_obs=100, 
        min_effective_obs=80,
        min_unique_dates=20,
        min_unique_stocks=30,
        min_obs_per_parameter=10.0
    )
"""
```

**Recommendations**:
- Add **"Quick Reference"** section to module docstring with contract selection decision tree
- Document all 4 undocumented classes
- Add **"Common Patterns"** section with 3-5 real-world contract compositions
- Add validation examples showing how contracts interact (e.g., clock vs label timing)

---

### 2. **evidence.py** — Grade: A

**Strengths**:
- Excellent module docstring with clear §71 walk-forward reference
- 2/2 classes fully documented (100%)
- 4/6 functions documented (67%)
- **98 temporal references**: highest density, shows deep integration with temporal contracts
- **70 hard gate references**: most comprehensive hard gate documentation
- Dynamic probes documented (e.g., `_probe_no_fit_in_score`, `_probe_purge_label_overlap`)

**Gaps**:
- Missing: **"How to Read Evidence Output"** guide
- Missing: **Hard gate summary table** showing all 18 gates, their status values, and remediation paths
- Missing: **Threshold documentation** for what constitutes "acceptable" OOS degradation

**Critical Missing Documentation**:
```python
# NEEDS: Hard gate reference table
"""
Hard Gates Reference
====================

| Gate | Status | Meaning | Remediation |
|------|--------|---------|-------------|
| MODEL_ZERO_PREDICTIVE_LEARNER_FITTED_INSIDE_SCORE_PATH | PASS/FAIL | Frozen predict never calls fit | Fix: Add TrainOnlyFitGuard |
| MODEL_ALL_PREDICTIVE_LEARNERS_HAVE_VALIDATION | PASS/FAIL | Every learner uses validation-backed selection | Fix: Add validation_bars > 0 to spec |
| MODEL_CURRENT_HEAD_EVIDENCE_FRESH | PASS/FAIL | Evidence matches current git HEAD | Fix: Re-run evidence at HEAD |
| ... | ... | ... | ... |

PASS: Probe verified the invariant holds
FAIL: Probe detected a violation
NOT_RUN: Probe could not execute (missing fixture/module)
"""
```

**Recommendations**:
- Add **"Evidence Interpretation Guide"** section to module docstring
- Document hard gate taxonomy and status semantics
- Add example showing how to debug a FAIL gate

---

### 3. **leakage_guard.py** — Grade: A

**Strengths**:
- Outstanding module docstring: clear three-layer architecture (§13/§58)
- 1/1 class documented (100%)
- 11/12 functions documented (92%)
- **80 temporal references**: strong temporal awareness
- Comprehensive negative control documentation
- Every control documents its **exercised/mutation_effect_verified/status** contract

**Gaps**:
- Missing: **"When to Use Which Guard"** decision tree
- Missing: **Integration examples** showing how to wire guards into custom trainers
- Negative controls lack **expected failure scenarios** documentation

**Critical Missing Documentation**:
```python
# NEEDS: Guard selection guide
"""
Leakage Guard Selection
========================

Use TrainOnlyFitGuard when:
  - Building a custom scoring path
  - Integrating a third-party learner
  - Testing that predict() never calls fit()

Use assert_frozen_preprocessing when:
  - Verifying train-only scaler/PCA
  - Debugging preprocessing state changes
  - Validating new preprocessing steps

Use run_all_negative_controls when:
  - Certifying a new learner for production
  - Validating walk-forward implementation
  - Generating evidence for hard gates
"""
```

**Recommendations**:
- Add **"Leakage Guard Cookbook"** with 3-5 common scenarios
- Document **expected failure patterns** for each negative control
- Add examples of **custom negative controls** for domain-specific leakage

---

### 4. **artifact.py** — Grade: B+

**Strengths**:
- Good module docstring linking to taskbook §21/§22/§31/§52
- 3/3 classes documented (100%)
- **124 temporal references**: highest absolute count, shows rich availability/cutoff handling
- `ModelArtifactManifest` has **excellent field-level documentation**
- Fail-closed ordering invariants documented in `__post_init__`

**Gaps**:
- Only 3/14 functions documented (21%) — many internal helpers undocumented
- Missing: **"Artifact Lifecycle"** diagram (creation → storage → retrieval → scoring)
- Missing: **Cache key composition** explanation (what goes into cache_key()?)
- Missing: **As-of resolution examples** (§50 critical but no examples)
- Missing: **MF-P0-005 field migration** guide (old vs new time fields)

**Critical Missing Documentation**:
```python
# NEEDS: Artifact lifecycle documentation
"""
Artifact Lifecycle
==================

1. **Training** (trainer.py)
   - train_model() fits learner + preprocessing
   - Creates ModelArtifact with manifest
   
2. **Storage** (artifact.save())
   - Atomic write to .json
   - Includes lineage_hash for integrity
   
3. **Retrieval** (dsl_bridge.ArtifactResolver)
   - Resolves by (model_name, asof)
   - Returns latest legal artifact (§50)
   
4. **Scoring** (predictor.py)
   - Loads artifact via resolver
   - Calls artifact.predict() under TrainOnlyFitGuard
   
5. **Caching** (cache_key())
   - Keys: model_name | training_cutoff | available_at | ...
   - Future-trained artifacts never collide
"""
```

**Recommendations**:
- Add **"Artifact Lifecycle"** section to module docstring
- Document cache key composition and collision prevention
- Add **3-5 as-of resolution examples** (early/mid/late, future cutoff)
- Document **MF-P0-005 migration** (old → new time field mapping)

---

### 5. **timing.py** — Grade: B

**Strengths**:
- Good module docstring with clear §8/§11/§62 references
- 1/1 class documented (100%)
- 8/9 functions documented (89%)
- **65 temporal references**: appropriate for timing module
- `maturity_cutoff_fail_closed` has excellent **MF-P0-004** documentation

**Gaps**:
- Missing: **"Decision Clock Scenarios"** comparison table (BEFORE_SAME_DAY_VWAP vs AFTER_CLOSE_TO_NEXT_VWAP)
- Missing: **Visual timeline** showing feature availability within a trading day
- Missing: **Common timing violations** and how to fix them
- No examples of **custom decision clocks** for other markets/frequencies

**Critical Missing Documentation**:
```python
# NEEDS: Decision clock scenario comparison
"""
A-Share Decision Clock Scenarios
=================================

AFTER_CLOSE_TO_NEXT_VWAP (default production):
  Timeline: t_close → decision → score_write(t+1_open) → execution(t+1_VWAP)
  Features: close, vwap, volume (all of day t known)
  Label: VWAP_{t+H} / VWAP_t - 1
  Use: Overnight factor production

BEFORE_SAME_DAY_VWAP (intraday execution):
  Timeline: t_prev_close → decision → score_write(t_open) → execution(t_VWAP)
  Features: open (ok), close/vwap/volume (REJECTED, complete after execution)
  Label: same as above
  Use: Intraday factor production (same-day execution)

Common violations:
  ✗ Using 'close' feature with BEFORE_SAME_DAY_VWAP
  ✗ Label maturity before final_fit_end
  ✗ execution_at before decision_at
"""
```

**Recommendations**:
- Add **scenario comparison table** to module docstring
- Add **visual timeline** showing decision/execution ordering
- Document **how to create custom clocks** for new markets
- Add examples of **clock validation failures** and fixes

---

### 6. **trainer.py** — Grade: B+

**Strengths**:
- Good module docstring linking to §13.2/§13.3/§19
- 2/2 classes documented (100%)
- 2/2 public functions documented (100%)
- **77 temporal references**: strong temporal awareness
- **Fail-closed adequacy** well documented (ValueError on all candidates failing)
- Sample weight policy integration documented

**Gaps**:
- Many internal functions undocumented (14 private helpers)
- Missing: **"Trainer Configuration Cookbook"** (common hyperparam grids, preprocessing specs)
- Missing: **"Training Failure Diagnosis"** guide (adequacy fail, convergence fail, etc.)
- Missing: **Performance guidance** (typical train times, when to use decay weights)
- Missing: **Governance contract** documentation (what is CandidateExposureLedger?)

**Critical Missing Documentation**:
```python
# NEEDS: Training configuration cookbook
"""
Trainer Configuration Cookbook
================================

Pattern 1: Simple PCR (quick exploration)
  preprocessing_spec = PreprocessingSpec(
      steps=("imputer", "winsor", "standardize")
  )
  hyperparam_grid = [
      {"n_components": 2},
      {"n_components": 3},
      {"n_components": 5}
  ]

Pattern 2: Elastic Net (sparse features)
  preprocessing_spec = PreprocessingSpec(
      steps=("imputer", "standardize"),
      use_standardize=True
  )
  hyperparam_grid = [
      {"alpha": 0.01, "l1_ratio": 0.5},
      {"alpha": 0.1, "l1_ratio": 0.7}
  ]

Pattern 3: Regime Model (time-varying)
  # Requires aux_col with regime state
  hyperparam_grid = [
      {"n_regimes": 2, "gating_feature_index": 0},
      {"n_regimes": 3, "gating_feature_index": 0}
  ]

Common Failures:
  ValueError("all N candidates failed adequacy") 
    → Check min_obs_per_parameter vs free parameters
  ValueError("validation_ds required")
    → Add validation_ds or use evaluation_boundary
"""
```

**Recommendations**:
- Add **"Common Configurations"** section with 5-7 recipes
- Add **"Troubleshooting"** section documenting common failures
- Document **governance contracts** (CandidateExposureLedger, parameter validation)
- Add **performance guidance** (expected runtimes, memory usage)

---

### 7. **sample_policy.py** — Grade: A-

**Strengths**:
- Excellent module docstring with clear §5.1/§5.2 chain documentation
- 2/2 classes documented (100%)
- 6/6 functions documented (100%)
- **61 temporal references**: appropriate density
- **§5.1 chain** (raw → finite → mature → purge → regime → effective) clearly documented
- Three-state contract logic well explained

**Gaps**:
- Missing: **"Sample Weight Policy Selection"** guide (when to use which policy)
- Missing: **Visual diagram** of the §5.1 chain narrowing
- Missing: **Adequacy floor recommendations** by model family (how high is high enough?)

**Critical Missing Documentation**:
```python
# NEEDS: Weight policy selection guide
"""
Sample Weight Policy Selection
================================

EQUAL_ROW (default):
  Use: Balanced panel with uniform date coverage
  Effect: Every observation has equal influence
  
EQUAL_DATE:
  Use: Unbalanced panel (varying stocks per date)
  Effect: Every date has equal aggregate influence
  
TIME_DECAY_EQUAL_DATE:
  Use: Non-stationary markets, recent data more relevant
  Effect: Exponential decay on date masses
  Params: half_life_dates (e.g., 60 = 60-day half-life)
  
Example:
  # Recency-weighted training for momentum models
  weights = sample_weights(
      ds, 
      SampleWeightPolicy.TIME_DECAY_EQUAL_DATE,
      half_life_dates=60.0
  )
"""
```

**Recommendations**:
- Add **"Policy Selection Guide"** to module docstring
- Add **§5.1 chain diagram** (ASCII art or reference to external diagram)
- Document **recommended adequacy floors** by model family

---

## Critical Missing Documentation

### 1. **Package-Level Integration Guide** (modeling/__init__.py)

Currently missing:

```python
"""
Modeling Layer — Predictive Model Lifecycle
============================================

This package implements the full predictive modeling lifecycle with
temporal leakage prevention, artifact management, and behavioral certification.

Quick Start
-----------

1. **Define Contracts** (contracts.py):
   - SampleAdequacyContract: min obs/dates/stocks
   - LabelContract: horizon, return basis
   - DecisionClock: feature availability

2. **Train Model** (trainer.py):
   - Loads PanelDataset
   - Fits preprocessing (train-only)
   - Searches hyperparameters on validation
   - Returns frozen ModelArtifact

3. **Generate Evidence** (evidence.py):
   - Runs walk-forward outer loop
   - Produces OOS evaluation
   - Verifies hard gates

4. **Deploy** (predictor.py + dsl_bridge.py):
   - Resolves artifact by (name, asof)
   - Scores under no-fit guard
   - Writes factor values

Temporal Safety
---------------

The modeling layer enforces temporal safety at FOUR boundaries:

1. **Feature availability** (DecisionClock): 
   - Features must be known before execution
   - clock_compliant() rejects future-leaking features

2. **Label maturity** (LabelContract):
   - Training uses only mature labels (anchor + horizon)
   - Immature rows purged fail-closed

3. **Preprocessing** (FrozenPreprocessing):
   - Statistics fit on TRAIN only
   - Transform never recomputes from input

4. **Artifact availability** (ModelArtifact.is_legal_asof):
   - Future-trained artifacts never returned for history
   - available_at >= training_cutoff

Hard Gates (evidence.py)
-------------------------

18 behavioral gates verify temporal safety:
  - MODEL_ZERO_PREDICTIVE_LEARNER_FITTED_INSIDE_SCORE_PATH
  - MODEL_ZERO_RANDOM_TIME_SPLIT
  - MODEL_ZERO_FULL_SAMPLE_SCALER
  - MODEL_ALL_ARTIFACTS_ASOF_RESOLVED
  - ... (see evidence.report_hard_gate_set)

Integration with Factor Engine
-------------------------------

See: docs/modeling_integration.md (TODO)
"""
```

### 2. **Hard Gates Reference Table**

Should be in `evidence.py` or separate `HARD_GATES.md`:

| Gate ID | Status | Probe | Failure Remediation |
|---------|--------|-------|---------------------|
| MODEL_ZERO_PREDICTIVE_LEARNER_FITTED_INSIDE_SCORE_PATH | PASS/FAIL | _probe_no_fit_in_score | Add TrainOnlyFitGuard to predict() |
| MODEL_ALL_PREDICTIVE_LEARNERS_HAVE_VALIDATION | PASS/FAIL | registry validation_bars check | Set validation_bars > 0 in WalkForwardSpec |
| MODEL_ALL_PREDICTIVE_LEARNERS_PURGE_LABEL_OVERLAP | PASS/FAIL | _probe_purge_label_overlap | Fix purge_before_boundary logic |
| MODEL_ZERO_RANDOM_TIME_SPLIT | PASS/FAIL | _probe_zero_random_time_split | Use date-ordered deterministic splits |
| MODEL_ZERO_FULL_SAMPLE_SCALER | PASS/FAIL | _probe_full_sample_preprocessing_leak | Fit preprocessing on train only |
| MODEL_ZERO_FULL_SAMPLE_PCA_PREPROCESS | PASS/FAIL | scaler + no_fit probes | PCR fits PCA inside learner.fit |
| MODEL_ZERO_FULL_SAMPLE_FEATURE_SELECTION | NOT_RUN | static audit | No feature-selection module to probe |
| MODEL_ZERO_TEST_DRIVEN_HYPERPARAM_SELECTION | PASS/FAIL | run_all_negative_controls | Fix hyperparam_poison control |
| MODEL_ALL_ARTIFACTS_ASOF_RESOLVED | PASS/FAIL | _probe_asof_resolution | Fix ArtifactResolver.resolve() logic |
| MODEL_ALL_MODELS_HAVE_SAMPLE_ADEQUACY_CONTRACT | PASS/FAIL | _probe_adequacy_contract_binding | Bind sample_contract_family |
| MODEL_ALL_SEARCHABLE_PARAMS_HAVE_SEARCH_POLICY | PASS/FAIL | _probe_search_policy_rejects | Add ParameterSearchPolicy |
| MODEL_ZERO_NUMERICAL_POLICY_SEARCHABLE | PASS/FAIL | _probe_search_policy_rejects | Set searchable=False for NUMERICAL_POLICY |
| MODEL_ZERO_DATA_POLICY_SEARCHABLE_BY_FACTOR_MINER | PASS/FAIL | _probe_search_policy_rejects | Set searchable=False for DATA_POLICY |
| MODEL_REGIME_ALL_COMPONENTS_HAVE_SUPPORT | PASS/FAIL | _probe_regime_support | Fix min_regime_obs contract |
| MODEL_MOE_ALL_ACTIVE_EXPERTS_HAVE_SUPPORT | PASS/FAIL | _probe_moe_support | Fix min_expert_obs contract |
| MODEL_FINAL_HOLDOUT_NOT_EXPOSED_TO_SEARCH | PASS/FAIL | _probe_final_holdout_not_exposed | Fix fold disjointness |
| MODEL_CURRENT_HEAD_EVIDENCE_FRESH | PASS/FAIL | git SHA comparison | Re-run evidence at current HEAD |
| MODEL_ALL_DIRECT_USE_HAVE_BEHAVIORAL_CERTIFICATION | PASS/FAIL | check_model_direct_use_readiness | Certify timing/lane/params |

### 3. **Decision Clock Visual Timeline**

```
AFTER_CLOSE_TO_NEXT_VWAP Timeline:
====================================

Day t:
  09:30 ───── open ───────────────────────── 15:00 close
                                                   ↓
                                            decision_at
                                                   ↓
                                          [all day-t data known]
                                                   ↓
Day t+1:                                    score written
  09:30 open ← (score available)
           ↓
      VWAP execution ← [factor_t used here]
           ↓
  Label: VWAP_{t+H} / VWAP_t - 1


BEFORE_SAME_DAY_VWAP Timeline:
================================

Day t-1:
  15:00 close
       ↓
  decision_at (prev_close)
       ↓
Day t:
  09:30 open ← score written
       ↓
  10:xx VWAP execution ← [factor_t used here]
       ↓
  Features available: open (✓), close (✗), vwap (✗), volume (✗)
       ↓
  15:00 close (close/vwap/volume complete AFTER execution → rejected)
```

---

## Prioritized Recommendations

### **P0 (Critical — Blocks Production Use)**

1. **Add modeling/__init__.py comprehensive docstring** with:
   - Quick start (4-step lifecycle)
   - Temporal safety boundaries
   - Hard gates summary
   - Integration with factor engine (cross-reference)

2. **Document hard gates in evidence.py**:
   - Add `HARD_GATES_REFERENCE` table (gate → status → remediation)
   - Document status semantics (PASS/FAIL/NOT_RUN)
   - Add "How to Debug a FAIL Gate" section

3. **Add artifact lifecycle documentation to artifact.py**:
   - Creation → storage → retrieval → scoring flow
   - Cache key composition explanation
   - 3-5 as-of resolution examples (§50 critical)

### **P1 (High — Improves Usability)**

4. **Add "Trainer Configuration Cookbook" to trainer.py**:
   - 5-7 common configurations (PCR, ElasticNet, Regime, MoE)
   - Troubleshooting section (adequacy failures, convergence)
   - Performance guidance (runtimes, memory)

5. **Add decision clock comparison to timing.py**:
   - Visual timeline for both scenarios
   - Feature availability table
   - Common violations and fixes

6. **Add integration examples to contracts.py**:
   - Complete contract composition (exec class + label + clock + sample)
   - Decision tree for choosing execution class
   - Validation examples (contracts interact correctly)

### **P2 (Medium — Fills Knowledge Gaps)**

7. **Document internal functions** (across all files):
   - contracts.py: 15 undocumented functions
   - artifact.py: 11 undocumented helpers
   - trainer.py: 14 undocumented helpers

8. **Add "Leakage Guard Cookbook" to leakage_guard.py**:
   - When to use which guard
   - Integration examples (custom trainers)
   - Custom negative controls for domain-specific leakage

9. **Add "Sample Weight Policy Selection" to sample_policy.py**:
   - When to use each policy
   - Recommended adequacy floors by model family
   - §5.1 chain visual diagram

### **P3 (Nice to Have — Polish)**

10. **Add type hints to all public functions** (currently inconsistent)
11. **Cross-reference between modules** (e.g., "See also: timing.py for clock validation")
12. **Performance benchmarks** (typical train times, evidence generation times)

---

## Summary Statistics

| File | Module Doc | Classes | Functions | Temporal Refs | Hard Gates | Grade |
|------|------------|---------|-----------|---------------|------------|-------|
| contracts.py | ✓ | 12/16 (75%) | 4/19 (21%) | 96 | 24 | A- |
| evidence.py | ✓ | 2/2 (100%) | 4/6 (67%) | 98 | 70 | A |
| leakage_guard.py | ✓ | 1/1 (100%) | 11/12 (92%) | 80 | 23 | A |
| artifact.py | ✓ | 3/3 (100%) | 3/14 (21%) | 124 | 4 | B+ |
| timing.py | ✓ | 1/1 (100%) | 8/9 (89%) | 65 | 9 | B |
| trainer.py | ✓ | 2/2 (100%) | 2/2 (100%) | 77 | 16 | B+ |
| sample_policy.py | ✓ | 2/2 (100%) | 6/6 (100%) | 61 | 18 | A- |

**Overall**: 7/7 module docs (100%), 23/27 classes (85%), 38/68 functions (56%)

---

## Conclusion

The modeling layer has **excellent technical documentation** at the module and class level, with comprehensive taskbook cross-references and strong temporal awareness. However, it lacks **operational documentation** needed for:

1. **New users**: No quick start, no integration guide, no configuration cookbook
2. **Debugging**: No hard gate reference, no troubleshooting guide, no failure diagnosis
3. **Production**: No artifact lifecycle guide, no performance guidance, no deployment patterns

**Action**: Implement P0 recommendations (modeling/__init__.py + hard gates + artifact lifecycle) to unblock production use.
