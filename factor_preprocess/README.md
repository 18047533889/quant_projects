# factor_preprocess

Factor preprocessing and representation layer: turns selected factors into
model-ready features — neutralization (industry / market-cap), standardization,
winsorization, rolling/smoothing transforms, regime detection, and typed
`FeatureBundle` outputs with a deep-freeze / causal contract.

**Version:** 0.1.0 ｜ **Repo:** https://github.com/HKUST-QUANT-SOCIETY/factor_preprocess (private)

## Install & first steps

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_preprocess.git
cd factor_preprocess
pip install -e .            # + [fast] for numba/polars/bottleneck
```

```python
from factor_preprocess.registry.policies import get_default_policy_registry
from factor_preprocess.registry.transforms import get_default_registry
from factor_preprocess.contracts.feature_bundle import FeatureBundle

policies = get_default_policy_registry()     # presets: production_full, research_full, cs_only, ...
policy = policies.get("production_full")
# build transform chain → FittedState → FeatureBundle
```

## What it owns

- **Policy presets** (`registry/policies.py`): `production_full`, `research_full`,
  `cs_only`, `minimal`, `returns_preprocessing`, `causal_basic`, ... each a chain
  of `TransformStep` (cs_winsor → cs_rank/cs_zscore → neutralization → smoothing...).
- **Transforms** (`transforms/`): cross-sectional (rank/zscore/demean/winsor/scale),
  rolling (mean/std/zscore), smoothing (ewma/kama/kalman/low-pass/robust_ewma),
  volatility (scale / realized vol / garch-inspired), event_decay, freshness,
  missingness, treatment_variants.
- **Neutralization** (`neutralization/`): OLS / ridge / lasso / elastic-net,
  `NeutralizationSpec` (method, exposures, standardization, PIT identity),
  diagnostics artifact (rank-deficient resolution).
- **Regime** (`regime/`): variance/correlation regime detection, adaptive weights,
  causal detector.
- **Representation** (`representation/`): `linear_ready` / `tree_ready` /
  `neural_ready` / `multichannel` model input layouts.
- **Eligibility** (`eligibility/`): treatment search space + eligibility rules
  (factor family × treatment family).
- **Contracts** (`contracts/`): `FeatureBundle`, `FittedState`, `TreatmentRecipe`,
  `TreatmentLineage`, `FactorProfileArtifact`, `PreprocessingPolicy` (deprecated view).
- **Backends / kernels**: `polars_backend` + selector; numba fast kernels +
  reference bridge (parity-tested).

## Causal / leakage discipline

- `FitBoundary` in `TreatmentRecipe` and `FittedState` deep-freeze; transforms are
  row-wise deterministic / only depend on train-only frozen state
  (complemented by `modeling.leakage_guard.assert_frozen_preprocessing`).
- KAMA and similar smoothings default to strict causal `shift(1)`; the `use_current`
  flag is only enabled when an external clock (`DecisionClock`) has already
  decided usability.

## Related repos

- **data_access** — universe/calendar/values via `adapters/data_access.py`
- **factor_assets** — `adapters/factor_assets.py`
- **modeling** — consumes `FeatureBundle` as model input
- **quant_platform** — feature_set DTOs map onto FP contracts
