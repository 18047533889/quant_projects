# Package Collaboration Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Quant Factor Platform                            │
│                  Research-to-Production Pipeline                     │
└─────────────────────────────────────────────────────────────────────┘

                         ┌──────────────┐
                         │ Parent Factor│
                         │  Expression  │
                         └──────┬───────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  STAGE 1: Factor Optimizer (FO)               │
        │  ─────────────────────────────────────        │
        │  • Generate mutations (param sweep, op sub)   │
        │  • Track complexity delta                     │
        │  • Manage search budget (L0-L4)               │
        │  • Coordinate with QE for evaluation          │
        └───────────────────┬───────────────────────────┘
                            │
                            │ Candidate Expressions
                            │
                            ▼
        ┌───────────────────────────────────────────────┐
        │  STAGE 2: Quantitative Evaluator (QE)         │
        │  ─────────────────────────────────────        │
        │  • Load factor and return data                │
        │  • Calculate IC (Pearson & Rank)              │
        │  • Quantile analysis + monotonicity           │
        │  • Progressive fidelity (L0→L3)               │
        │  • Return EvidenceBundle                      │
        └───────────────────┬───────────────────────────┘
                            │
                            │ Evidence (IC, IR, Metrics)
                            │
                            ▼
        ┌───────────────────────────────────────────────┐
        │  STAGE 3: Factor Assets (FA)                  │
        │  ─────────────────────────────────────        │
        │  • Generate canonical factor_id (hash)        │
        │  • Register FactorAsset with evidence         │
        │  • Track lifecycle state                      │
        │  • Apply selection gates (IC/IR/Mono)         │
        │  • Build versioned FactorSet                  │
        └───────────────────┬───────────────────────────┘
                            │
                            │ FactorSet (Selected IDs)
                            │
                            ▼
        ┌───────────────────────────────────────────────┐
        │  STAGE 4: Factor Preprocess (FP)              │
        │  ─────────────────────────────────────        │
        │  • Load raw factor values                     │
        │  • Cross-sectional transforms (rank, zscore)  │
        │  • Time-series transforms (rolling, decay)    │
        │  • OLS neutralization (industry, size)        │
        │  • Package as FeatureBundle                   │
        └───────────────────┬───────────────────────────┘
                            │
                            │ FeatureBundle (3D Matrix)
                            │
                            ▼
                ┌───────────────────────┐
                │  Model Training       │
                │  • Linear             │
                │  • Tree (XGBoost)     │
                │  • Neural Net         │
                └───────────────────────┘
```

## Data Flow Detail

```
┌──────────────────────────────────────────────────────────────────┐
│  Example Flow: Momentum Factor Optimization                      │
└──────────────────────────────────────────────────────────────────┘

Step 1: FO - Mutation Generation
─────────────────────────────────
  Parent: ts_rank(close / ts_delay(close, 20), 120)
     │
     ├─► Mutation 1: ts_rank(close / ts_delay(close, 10), 120)  [param sweep]
     ├─► Mutation 2: ts_rank(close / ts_delay(close, 30), 120)  [param sweep]
     ├─► Mutation 3: ts_rank(close / ts_delay(close, 20), 60)   [param sweep]
     └─► Mutation 4: ts_zscore(close / ts_delay(close, 20), 120) [op sub]


Step 2: QE - Progressive Evaluation
────────────────────────────────────
  Mutation 1:
    L0 (10d, 100sym):  IC=0.068, IR=0.43  ✓ Pass → L2
    L2 (120d, 1000sym): IC=0.042, IR=0.29  ✓ Pass → L3
    L3 (250d, 3000sym): IC=0.028, IR=0.17  → Evidence

  Mutation 2:
    L0: IC=-0.024  ✗ Fail → Stop (budget saved)

  Mutation 3:
    L0: IC=0.070  ✓ Pass → L2
    L2: IC=0.072  ✓ Pass → L3
    L3: IC=0.070, IR=0.50  → Evidence ⭐ (Best!)

  Mutation 4:
    L0: IC=0.048  ✓ Pass → L2
    L2: IC=0.073  ✓ Pass → L3
    L3: IC=0.051, IR=0.36  → Evidence


Step 3: FA - Registration & Selection
──────────────────────────────────────
  Register assets with evidence:
    factor_79525: Parent    IC=0.032, IR=0.20
    factor_28133: Mutation1 IC=0.028, IR=0.17
    factor_74768: Mutation3 IC=0.070, IR=0.50 ⭐
    factor_42769: Mutation4 IC=0.051, IR=0.36

  Apply gates (IC≥0.03, IR≥0.9):
    ✗ factor_79525: IC=0.032, IR=0.20  [IR too low]
    ✗ factor_28133: IC=0.028, IR=0.17  [Both too low]
    ✓ factor_74768: IC=0.070, IR=0.50  [PASS if lower threshold]
    ✓ factor_42769: IC=0.051, IR=0.36  [PASS if lower threshold]

  FactorSet: research_alpha_v1
    Selected: [factor_74768, factor_42769]


Step 4: FP - Preprocessing
───────────────────────────
  For each factor in FactorSet:
    Load raw values → (250 dates × 500 symbols)
       ↓
    Winsorize (1%, 99%) → Clip outliers
       ↓
    Cross-sectional Z-score → Standardize per date
       ↓
    Cross-sectional Rank → Percentile [0,1]
       ↓
    Rolling Z-score (60d) → Remove trends per symbol
       ↓
    OLS Neutralization → Remove industry/size exposure
       ↓
    FeatureBundle: (250 dates × 500 symbols × 2 features)


Step 5: Model Training (External)
──────────────────────────────────
  Load FeatureBundle → X: (250, 500, 2)
  Load forward returns → y: (250, 500)
  Train model → Predict returns from features
  Backtest → Calculate Sharpe, drawdown
  Deploy → Production inference
```

## Package Contracts

```
┌─────────────────────────────────────────────────────────────┐
│  Input/Output Contracts                                     │
└─────────────────────────────────────────────────────────────┘

FO Output → QE Input:
  • List of factor expressions (strings)
  • Complexity metric (int)
  
QE Output → FA Input:
  • EvidenceRef:
      - factor_id
      - ic_mean, ic_std, ic_ir
      - rank_ic_mean, rank_ic_std, rank_ic_ir
      - monotonicity_score
      - n_dates, sample_period

FA Output → FP Input:
  • FactorSet:
      - List of factor_ids
      - Selection criteria (thresholds)
      - Evidence bundle references

FP Output → Model Input:
  • FeatureBundle:
      - feature_matrix: ndarray (T × N × F)
      - dates: List[datetime]
      - symbols: List[str]
      - factor_ids: List[str]
      - transforms_applied: List[str]
```

## Execution Flow Matrix

```
┌────────────────────────────────────────────────────────────────┐
│  Who Does What                                                 │
└────────────────────────────────────────────────────────────────┘

                   │  FO  │  QE  │  FA  │  FP  │
───────────────────┼──────┼──────┼──────┼──────┤
Generate mutations │  ✓   │      │      │      │
Evaluate IC/IR     │      │  ✓   │      │      │
Register assets    │      │      │  ✓   │      │
Track lineage      │      │      │  ✓   │      │
Apply gates        │      │      │  ✓   │      │
Transform features │      │      │      │  ✓   │
Neutralize         │      │      │      │  ✓   │
Track provenance   │  ✓   │  ✓   │  ✓   │  ✓   │  (via Research Control)
Store data         │      │      │  ✓   │  ✓   │
Execute factor     │      │  ✓   │      │  ✓   │
```

## Example Mapping

```
┌────────────────────────────────────────────────────────────────┐
│  Which Example Demonstrates What                               │
└────────────────────────────────────────────────────────────────┘

Example 01: Basic Evaluation
  Focus: QE package
  Shows: IC calculation, quantile analysis, monotonicity
  Input: Synthetic factor + return data
  Output: Evaluation metrics

Example 02: Preprocessing Pipeline
  Focus: FP package
  Shows: CS/TS transforms, neutralization, bundling
  Input: Raw factor values
  Output: FeatureBundle (3D matrix)

Example 03: Factor Selection
  Focus: FA package
  Shows: Registration, identity, gates, FactorSet
  Input: Factor expressions + evidence
  Output: FactorSet JSON

Example 04: Optimization Search
  Focus: FO package
  Shows: Mutation, progressive fidelity, Pareto frontier
  Input: Parent factor
  Output: Best mutations with evidence

Example 05: Complete Workflow
  Focus: All packages
  Shows: Full pipeline integration with provenance
  Input: Parent factor
  Output: FeatureBundle + provenance log
```

## Integration Points

```
┌────────────────────────────────────────────────────────────────┐
│  Real-World Integration                                        │
└────────────────────────────────────────────────────────────────┘

1. Factor Computation
   Examples use synthetic data
   Production: FactorEngine computes real factor values
   
2. Data Storage
   Examples use in-memory/temp files
   Production: Database for FA registry, COS for FP bundles
   
3. Evaluation Backend
   Examples use simple IC calculation
   Production: factor_layer/factor_evaluation with full harness
   
4. Model Training
   Examples stop at FeatureBundle
   Production: Hook to XGBoost/LightGBM/Neural pipeline
   
5. Monitoring
   Examples have no monitoring
   Production: Track IC drift, alert on degradation
```
