#!/usr/bin/env python3
"""Quick verification that robust statistics operators are properly registered."""
import sys
import numpy as np
import pandas as pd

# Add project root to path
from pathlib import Path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.operator_surface import classify_canonical

print("Loading operator registry...")
ensure_cleaned_loaded()

# Check registration
operators = ["ts_quantile_range", "ts_trimmed_mean", "ts_robust_zscore_inclusive", "ts_robust_zscore_prior"]

print("\n" + "="*70)
print("OPERATOR REGISTRATION CHECK")
print("="*70)

for name in operators:
    op = OperatorRegistry.get(name)
    if op is None:
        print(f"❌ {name}: NOT REGISTERED")
        continue

    surface = classify_canonical(name)
    print(f"✓ {name}: registered, surface={surface}")

    # Check metadata
    if hasattr(op, 'metadata'):
        meta = op.metadata
        tags = meta.tags if hasattr(meta, 'tags') else []
        has_signature = any('signature:' in str(t) for t in tags)
        has_domain = any('domain:' in str(t) for t in tags)
        has_unit = any('unit:' in str(t) for t in tags)
        print(f"  - metadata: signature={has_signature}, domain={has_domain}, unit={has_unit}")

# Check policy
print("\n" + "="*70)
print("OPERATOR POLICY CHECK")
print("="*70)

from factor_engine.cleaned_operators.operator_policy import _EXPLICIT_POLICIES

for name in operators:
    if name in _EXPLICIT_POLICIES:
        policy = _EXPLICIT_POLICIES[name]
        print(f"✓ {name}: {policy}")
    else:
        print(f"❌ {name}: NO EXPLICIT POLICY")

# Check scalar values in audit script
print("\n" + "="*70)
print("AUDIT SCRIPT SCALAR VALUES CHECK")
print("="*70)

sys.path.insert(0, str(project_root / "scripts"))
from audit_all_factor_production import _SCALAR_VALUES

required_scalars = ["q_low", "q_high", "trim_ratio", "center", "scale"]
for scalar in required_scalars:
    if scalar in _SCALAR_VALUES:
        print(f"✓ {scalar}: {_SCALAR_VALUES[scalar]}")
    else:
        print(f"❌ {scalar}: NOT DEFINED")

# Quick functional test
print("\n" + "="*70)
print("FUNCTIONAL TEST")
print("="*70)

idx = pd.date_range("2024-01-01", periods=30, freq="D")
cols = ["A", "B"]
rng = np.random.default_rng(42)
panel = pd.DataFrame(rng.standard_normal((30, 2)), index=idx, columns=cols)

try:
    op = OperatorRegistry.get("ts_quantile_range")
    result = op.calculate(panel, window=20)
    print(f"✓ ts_quantile_range: shape={result.shape}, last_value={result.iloc[-1, 0]:.4f}")
    assert result.shape == panel.shape
except Exception as e:
    print(f"❌ ts_quantile_range: {e}")

try:
    op = OperatorRegistry.get("ts_trimmed_mean")
    result = op.calculate(panel, window=20, trim_ratio=0.1)
    print(f"✓ ts_trimmed_mean: shape={result.shape}, last_value={result.iloc[-1, 0]:.4f}")
    assert result.shape == panel.shape
except Exception as e:
    print(f"❌ ts_trimmed_mean: {e}")

try:
    op = OperatorRegistry.get("ts_robust_zscore_inclusive")
    result = op.calculate(panel, window=20)
    print(f"✓ ts_robust_zscore_inclusive: shape={result.shape}, last_value={result.iloc[-1, 0]:.4f}")
    assert result.shape == panel.shape
except Exception as e:
    print(f"❌ ts_robust_zscore_inclusive: {e}")

try:
    op = OperatorRegistry.get("ts_robust_zscore_prior")
    result = op.calculate(panel, window=20)
    print(f"✓ ts_robust_zscore_prior: shape={result.shape}, last_value={result.iloc[-1, 0]:.4f}")
    assert result.shape == panel.shape
except Exception as e:
    print(f"❌ ts_robust_zscore_prior: {e}")

print("\n" + "="*70)
print("ALL CHECKS COMPLETE")
print("="*70)
