# Quant Factor Platform Examples - Quick Start Guide

## Overview

Five complete, runnable examples demonstrating end-to-end collaboration between the four packages:

- **Factor Optimizer (FO)**: Mutation grammar and search coordination
- **Quantitative Evaluator (QE)**: Factor evaluation and evidence generation  
- **Factor Assets (FA)**: Identity, registry, and lifecycle management
- **Factor Preprocess (FP)**: Model-input preparation

All examples are standalone with inline implementations—no package installation required.

---

## Running the Examples

Each example is self-contained and can be run independently:

```bash
cd /home/shw/quant_projects/examples

# Run individual examples
python3 01_basic_evaluation.py
python3 02_preprocessing_pipeline.py
python3 03_factor_selection.py
python3 04_optimization_search.py
python3 05_complete_workflow.py
```

---

## Example Summaries

### 01_basic_evaluation.py (QE Package)
**Focus**: Factor evaluation fundamentals

**What it demonstrates**:
- Generate synthetic factor and market data
- Winsorize and standardize factor values
- Calculate IC metrics (Pearson and Rank)
- Quantile analysis and monotonicity
- Long-short portfolio backtest

**Key concepts**:
- Information Coefficient (IC): correlation between factor and forward returns
- IC IR: risk-adjusted IC (IC mean / IC std)
- Monotonicity: consistent factor-return relationship across quantiles

**Output**: Evaluation metrics showing factor predictive power

**Runtime**: ~5 seconds

---

### 02_preprocessing_pipeline.py (FP Package)
**Focus**: Transform raw factors to model-ready features

**What it demonstrates**:
- Load multiple raw factors
- Cross-sectional transforms (winsorize, zscore, rank)
- Time-series transforms (rolling zscore, decay)
- OLS neutralization (remove industry/size exposures)
- Package into FeatureBundle

**Key concepts**:
- Cross-sectional normalization: standardize within each date
- Time-series smoothing: remove trends per symbol
- Neutralization: isolate alpha orthogonal to known factors
- FeatureBundle: standardized contract for model input

**Output**: 3D feature matrix (dates × symbols × features)

**Runtime**: ~10 seconds

---

### 03_factor_selection.py (FA Package)
**Focus**: Register factors and apply selection gates

**What it demonstrates**:
- Generate canonical factor IDs from expressions
- Register FactorAssets with evidence
- Apply multi-gate selection (IC, IR, monotonicity)
- Build versioned FactorSet
- Save selection results

**Key concepts**:
- Canonical identity: factor_id from expression hash
- Lifecycle states: DRAFT → EVALUATED → VALIDATED → PRODUCTION
- Selection gates: evidence-based filtering
- FactorSet: versioned collection of selected factors

**Output**: FactorSet JSON with selected factors and criteria

**Runtime**: ~1 second

---

### 04_optimization_search.py (FO Package)
**Focus**: Mutation-based factor search with progressive fidelity

**What it demonstrates**:
- Define parent factor and generate mutations
- Parameter sweeps (window lengths)
- Operator substitutions (rank → zscore)
- Progressive fidelity evaluation (L0 → L3)
- Pareto frontier tracking (IC vs complexity)

**Key concepts**:
- Mutation types: parameter_sweep, operator_substitution, composition
- Multi-fidelity evaluation: cheap L0 filter, expensive L3 validation
- Budget tracking: conserve expensive evaluations
- Pareto frontier: non-dominated solutions

**Output**: Pareto frontier with best factors at each complexity level

**Runtime**: ~2 seconds

---

### 05_complete_workflow.py (All Four Packages)
**Focus**: Full research-to-production pipeline

**What it demonstrates**:
- FO generates candidate factors via mutation
- QE evaluates each candidate with IC/IR metrics
- FA registers assets and applies selection gates
- FP transforms selected factors to FeatureBundle
- Research Control tracks full provenance

**Key concepts**:
- Package collaboration: each package has a focused responsibility
- Provenance tracking: full lineage from mutation to model input
- Selection pipeline: only high-quality factors reach production
- Reproducibility: all operations logged

**Output**: 
- Feature bundle (NPZ file)
- Provenance log (JSON)
- Pipeline statistics

**Runtime**: ~3 seconds

---

## Understanding the Output

### Example 01 Output
```
IC Mean:        0.0450     ← Correlation between factor and returns
IC IR:          1.20       ← Risk-adjusted IC (IC/std)
Top-Bottom:     0.0023     ← Return spread between Q5 and Q1
Monotonicity:   0.85       ← Consistency across quantiles
```

**Interpretation**:
- IC > 0.03 is good
- IC IR > 1.0 is excellent
- Monotonicity > 0.7 indicates proper factor direction

### Example 02 Output
```
Shape: (250, 500, 3)       ← 250 dates, 500 symbols, 3 features
Missing rate: 11.60%       ← Due to rolling window warm-up
Mean: 0.0000               ← Centered after z-score
Std: 0.9867                ← Standardized
```

### Example 03 Output
```
Selected: 3/5 factors      ← 3 passed selection gates
FactorSet ID: research_alpha_v1
```

### Example 04 Output
```
Pareto frontier: 1 factor
  Complexity 3: IC=0.0696  ← Best factor found
Budget consumed: L0=9/100, L3=3/10
```

### Example 05 Output
```
Pipeline statistics:
  Candidates generated: 7
  Evaluated: 3
  Selected: 3
  Features in bundle: 3
```

---

## Key Design Principles

### 1. Separation of Concerns
- **FO**: Generates candidates, no evaluation
- **QE**: Evaluates factors, no storage
- **FA**: Stores and selects, no computation
- **FP**: Transforms features, no evaluation

### 2. Evidence-Driven Selection
Factors must prove predictive power before reaching production:
```
Generate → Evaluate → Register → Select → Preprocess → Model
   FO         QE         FA        FA        FP       External
```

### 3. Progressive Fidelity
Cheap evaluations filter out poor candidates early:
```
L0 (100 candidates) → L2 (20 candidates) → L3 (10 candidates)
    1s/each               15s/each              60s/each
```

### 4. Reproducibility
Research Control logs every operation:
- Which parent was mutated
- What evaluation metrics resulted
- Which factors were selected
- What transforms were applied

---

## Next Steps After Examples

### For Research
1. Replace synthetic data with real market data
2. Implement actual factor expressions using FactorEngine
3. Run multi-iteration search with budget allocation
4. Build factor library with evidence tracking

### For Production
1. Deploy selected FactorSet to production pipeline
2. Set up real-time factor calculation
3. Monitor factor IC drift over time
4. Automate retraining when quality degrades

### For Integration
1. Connect to actual FactorEngine for factor computation
2. Use real QE implementation from factor_layer/factor_evaluation
3. Persist FA registry to database
4. Hook FP output directly to model training

---

## Troubleshooting

### ImportError: No module named 'factor_layer'
**Solution**: Examples are standalone and don't require package installation. Run them directly with `python3`.

### Low IC values in examples
**Expected**: Synthetic data has no real signal. Real factors with actual market data will show higher ICs.

### No factors pass selection in Example 05
**Solution**: Thresholds are calibrated for synthetic data. Real data requires different thresholds (typically IC > 0.03, IR > 1.0).

---

## File Structure

```
/home/shw/quant_projects/examples/
├── README.md                      # This file
├── QUICKSTART.md                  # Quick start guide
├── 01_basic_evaluation.py         # QE: Factor evaluation
├── 02_preprocessing_pipeline.py   # FP: Feature transformation
├── 03_factor_selection.py         # FA: Registration and selection
├── 04_optimization_search.py      # FO: Mutation search
└── 05_complete_workflow.py        # All packages integrated
```

---

## Additional Resources

### Package Documentation
- Factor Optimizer: `/home/shw/quant_projects/factor_optimizer/README.md`
- Quantitative Evaluator: `/home/shw/quant_projects/factor_layer/factor_evaluation/`
- Factor Assets: `/home/shw/quant_projects/factor_assets/README.md`
- Factor Preprocess: `/home/shw/quant_projects/factor_preprocess/README.md`

### Platform Blueprint
- Architecture: `/home/shw/quant_projects/quant_factor_platform_blueprint/`
- Design principles and package contracts

---

## Questions?

These examples demonstrate the conceptual workflow. For production use:
1. Integrate with FactorEngine for real factor computation
2. Use actual evaluation harness from factor_layer
3. Persist registry to database (FA)
4. Connect to model training pipelines

The standalone nature makes them ideal for:
- Understanding package collaboration
- Teaching factor research workflow
- Prototyping new features
- Debugging integration issues
