# FactorPreprocess Testing Guide

**Version:** 0.1.0  
**Last Updated:** 2026-08-14

## Quick Start

```bash
cd /home/shw/quant_projects/factor_preprocess
pytest tests/
```

## Test Organization

```
tests/
├── contracts/              # Contract validation
│   └── test_contracts.py
├── transforms/            # Transform correctness
│   ├── test_cross_sectional.py
│   ├── test_rolling.py
│   ├── test_volatility.py
│   ├── test_missingness.py
│   └── test_freshness.py
├── neutralization/        # Neutralization quality
│   ├── test_ols.py
│   ├── test_regularized.py
│   └── test_diagnostics.py
├── representation/        # Representation builders
│   ├── test_multichannel.py
│   ├── test_linear_ready.py
│   └── test_tree_ready.py
├── registry/             # Registry tests
│   ├── test_transforms_registry.py
│   └── test_policies_registry.py
├── parity/               # Fast vs reference
│   ├── test_kernels.py
│   └── test_performance.py
├── future_poison/        # Temporal leakage
│   └── test_future_poison.py
└── fold_boundary/        # Train/test safety
    └── test_fold_safety.py
```

## Key Test Patterns

### Cross-Sectional Transform
```python
def test_cs_rank_correctness():
    values = np.array([[1.0, 3.0, 2.0]])  # One time period
    ranked = cs_rank(values, axis=-1)
    
    expected = np.array([[0.0, 1.0, 0.5]])  # Ranks: 0, 2, 1
    np.testing.assert_allclose(ranked, expected)
```

### Causal Rolling
```python
def test_rolling_mean_excludes_current():
    df = pd.DataFrame({
        'date': [0, 1, 2],
        'asset': [0, 0, 0],
        'value': [1.0, 2.0, 3.0],
    })
    
    result = rolling_mean(df, window=2, min_periods=1, ...)
    
    # At date=2, mean should be (1+2)/2 = 1.5, NOT (1+2+3)/3 = 2.0
    assert result.iloc[2] == 1.5
```

### Future Poison Detection
```python
@pytest.mark.future_poison
def test_no_forward_looking():
    """Verify rolling transform doesn't leak future."""
    # Test that rolling_mean at time t only uses data <= t-1
    pass
```

## Pytest Markers

```bash
# Skip future poison tests (slow)
pytest tests/ -m "not future_poison"

# Only parity tests
pytest tests/parity/ -m parity
```

## Coverage Goals
- **Transforms:** 100%
- **Neutralization:** >95%
- **Representation:** >95%
- **Overall:** >90%

---

**Last Updated:** 2026-08-14
