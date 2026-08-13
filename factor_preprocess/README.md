# Factor Preprocess

**Status:** Initial bounded implementation (Wave 1)

Factor preprocessing and representation layer for quantitative research platform.

## Scope

This package handles model-input preparation **after** factor selection and **before** model training:

- Cross-sectional transforms (rank, zscore, winsor)
- Causal rolling transforms with explicit lag and warmup
- Neutralization (OLS residuals)
- Stateless and fitted transform separation
- Feature bundles with immutable fit metadata

## Out of Scope

- Factor computation (owned by FactorEngine)
- Factor evaluation (owned by QuantEvaluator)
- Factor selection (owned by FactorAssets)
- Model training (out of platform scope)
- Label computation (caller responsibility)

## Installation

```bash
cd /home/shw/quant_projects/factor_preprocess
pip install -e .
```

## Core Contracts

- `PreprocessingPolicy`: Ordered transform specifications
- `FittedState`: Immutable state with fit window metadata
- `FeatureBundle`: Transformed features ready for modeling

## Design Principles

- Batch-first API (single-factor is thin wrapper)
- Explicit future-poison prevention in rolling transforms
- Fold-local fitting (no full-sample fit before split)
- Asset isolation in time-series operations
- Reference and fast implementations with parity tests
