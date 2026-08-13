# Quant Factor Platform - End-to-End Examples

Complete, runnable examples demonstrating collaboration between the four core packages: Factor Optimizer (FO), Quantitative Evaluator (QE), Factor Assets (FA), and Factor Preprocess (FP).

## Quick Start

All examples are standalone with no installation required:

```bash
cd /home/shw/quant_projects/examples

# Run any example
python3 01_basic_evaluation.py
python3 02_preprocessing_pipeline.py
python3 03_factor_selection.py
python3 04_optimization_search.py
python3 05_complete_workflow.py
```

## Examples

| Example | Package | Focus | Runtime |
|---------|---------|-------|---------|
| **01_basic_evaluation.py** | QE | Factor evaluation with IC metrics | ~5s |
| **02_preprocessing_pipeline.py** | FP | Transform raw factors to model input | ~10s |
| **03_factor_selection.py** | FA | Register and select factors with gates | ~1s |
| **04_optimization_search.py** | FO | Mutation search with progressive fidelity | ~2s |
| **05_complete_workflow.py** | All | Full pipeline: FO → QE → FA → FP | ~3s |

## What You'll Learn

### Example 01: Factor Evaluation (QE)
- Calculate Information Coefficient (IC) between factors and returns
- Compute IC Information Ratio (risk-adjusted IC)
- Run quantile analysis and monotonicity checks
- Interpret evaluation metrics for factor quality

**Key Output**: IC metrics showing predictive power

### Example 02: Preprocessing Pipeline (FP)
- Apply cross-sectional transforms (winsorize, zscore, rank)
- Apply time-series transforms (rolling operations)
- Neutralize against exposures (industry, market cap)
- Package features into standardized FeatureBundle

**Key Output**: 3D feature matrix (dates × symbols × features)

### Example 03: Factor Selection (FA)
- Register factors with canonical identity (expression hash)
- Attach evaluation evidence from QE
- Apply selection gates (IC, IR, monotonicity thresholds)
- Build versioned FactorSet for downstream use

**Key Output**: FactorSet JSON with selected factors

### Example 04: Optimization Search (FO)
- Generate mutations (parameter sweeps, operator substitutions)
- Track multi-fidelity evaluation budget (L0-L4)
- Maintain Pareto frontier (IC vs complexity)
- Guide search with evidence feedback

**Key Output**: Pareto frontier with best factors at each complexity

### Example 05: Complete Workflow (All Packages)
- FO generates candidate factors
- QE evaluates each candidate
- FA registers and selects high-quality factors
- FP transforms to model-ready features
- Research Control tracks full provenance

**Key Output**: Feature bundle + provenance log

## Package Responsibilities

```
FO (Factor Optimizer)
├── Generate candidate factors via mutation
├── Track search budget and complexity
└── Delegate evaluation to QE

QE (Quantitative Evaluator)
├── Evaluate factors at multiple fidelities
├── Produce evidence bundles (IC, IR, monotonicity)
└── No storage—delegate to FA

FA (Factor Assets)
├── Register factors with canonical identity
├── Attach evidence from QE
├── Apply selection gates
└── Build versioned FactorSets

FP (Factor Preprocess)
├── Transform selected factors
├── Apply cross-sectional and time-series operations
├── Neutralize exposures
└── Package as FeatureBundle
```

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Research-to-Production                    │
└─────────────────────────────────────────────────────────────┘

  Parent Factor                                        Model Input
       │                                                     ▲
       ▼                                                     │
  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐    │
  │   FO    │──▶│   QE    │──▶│   FA    │──▶│   FP    │────┘
  │Generate │   │Evaluate │   │ Select  │   │Transform│
  └─────────┘   └─────────┘   └─────────┘   └─────────┘
       │             │             │             │
       ▼             ▼             ▼             ▼
  Mutations    IC/IR/Mono   FactorSet   FeatureBundle
```

## Key Concepts

### Information Coefficient (IC)
Correlation between factor values and forward returns:
- IC > 0.03: Good predictive power
- IC IR > 1.0: Excellent risk-adjusted performance
- Rank IC: More robust to outliers than Pearson IC

### Progressive Fidelity
Evaluate many candidates cheaply, few candidates thoroughly:
- **L0**: Quick filter (10 dates, 100 symbols, ~1s)
- **L2**: Medium check (120 dates, 1000 symbols, ~15s)
- **L3**: Full evaluation (250 dates, 3000 symbols, ~60s)

### Selection Gates
Multi-criteria filtering before production:
- IC threshold: Minimum predictive power
- IR threshold: Minimum risk-adjusted performance
- Monotonicity: Proper factor direction
- Lifecycle: Only production-ready factors

### FeatureBundle
Standardized contract for model input:
- 3D array: (dates × symbols × features)
- Metadata: factor IDs, transforms applied
- Provenance: full lineage from raw to processed

## Understanding the Output

### IC Metrics (Example 01)
```
IC Mean:        0.0450     Good (>0.03)
IC IR:          1.20       Excellent (>1.0)
Rank IC IR:     1.35       Very stable
Monotonicity:   0.85       Strong (>0.7)
```

### Feature Statistics (Example 02)
```
Shape: (250, 500, 3)       250 dates, 500 symbols, 3 features
Missing rate: 11.60%       Due to rolling window warm-up
Mean: 0.0000               Centered after normalization
Std: 0.9867                Standardized
```

### Selection Results (Example 03)
```
Selected: 3/5 factors      3 passed all gates
Gates: IC≥0.03, IR≥0.9, Mono≥0.75
```

### Search Results (Example 04)
```
Pareto frontier: 1 factor  Best at each complexity level
Budget: L0=9/100, L3=3/10  Efficient budget use
```

## Files

- **QUICKSTART.md**: Detailed guide with troubleshooting
- **01_basic_evaluation.py**: QE evaluation workflow
- **02_preprocessing_pipeline.py**: FP transformation pipeline
- **03_factor_selection.py**: FA registration and selection
- **04_optimization_search.py**: FO mutation search
- **05_complete_workflow.py**: Full integration

## Next Steps

### Research
1. Replace synthetic data with real market data
2. Implement actual factor expressions using FactorEngine
3. Run multi-iteration search with larger budgets
4. Build factor library with evidence tracking

### Production
1. Deploy selected FactorSet to production pipeline
2. Set up real-time factor calculation
3. Monitor factor IC drift
4. Automate retraining when quality degrades

### Integration
1. Connect to FactorEngine for factor computation
2. Use real QE from factor_layer/factor_evaluation
3. Persist FA registry to database
4. Hook FP output to model training

## Design Principles

### Separation of Concerns
Each package has a single, focused responsibility.

### Evidence-Driven Selection
Factors must prove predictive power before production.

### Progressive Fidelity
Conserve budget by filtering early.

### Reproducibility
Research Control logs every operation for full provenance.

---

**See QUICKSTART.md for detailed explanations, troubleshooting, and integration guidance.**
