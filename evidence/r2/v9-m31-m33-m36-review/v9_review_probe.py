"""Independent boundary probes for V9-M31, V9-M33, and V9-M36."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.candle_state_space import _matrix_profile_series
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.research_spectral import _residualized_hsic


load_all()
evidence = {}

# M31: independent affine oracle on physical row positions, with both internal
# gaps and a missing query row. The permitted policy is one-step endpoint
# extrapolation, so the answer is the affine value at physical position 19.
op = OperatorRegistry.get(
    "ts_causal_local_linear_smoother", "pandas_numpy", mode="research")
x = pd.DataFrame({"A": np.arange(20.0)})
x.iloc[[5, 6, 7, 19], 0] = np.nan
endpoint = float(op.calculate(x, window=20, min_periods=10).iloc[-1, 0])
assert endpoint == 19.0
evidence["V9-M31"] = {"affine_endpoint": endpoint}

# M33: enumerate the inclusive candidate band independently, then compare its
# exact 1/2/3-candidate boundary with kernel finiteness.
m33 = []
for history in (25, 26, 27):
    length = 20
    candidates = history - length - length // 4 + 1
    values = np.random.default_rng(31).normal(size=(80, 1))
    _, _, frequency, dispersion = _matrix_profile_series(
        values, 60, length, history)
    row = {
        "history": history,
        "candidate_count": candidates,
        "frequency_finite": int(np.isfinite(frequency).sum()),
        "dispersion_finite": int(np.isfinite(dispersion).sum()),
    }
    if history < 27:
        assert row["frequency_finite"] == row["dispersion_finite"] == 0
    else:
        assert candidates == 3
        assert row["frequency_finite"] > 0 and row["dispersion_finite"] > 0
    m33.append(row)
evidence["V9-M33"] = m33

# M36: independently enumerate both retained fold sizes at the last legal
# purge and the immediately illegal purge for even and odd windows.
m36 = []
rng = np.random.default_rng(36)
for window in (24, 25, 30, 31):
    for purge in (window // 2 - 6, window // 2 - 5):
        forward = window - (window // 2 + purge)
        backward = window // 2 - purge
        value = _residualized_hsic(*rng.normal(size=(3, window)), purge)
        expected = min(forward, backward) >= 6
        assert bool(np.isfinite(value)) is expected
        m36.append({
            "window": window,
            "purge_gap": purge,
            "forward_test_rows": forward,
            "backward_test_rows": backward,
            "finite": bool(np.isfinite(value)),
        })
evidence["V9-M36"] = m36

print(json.dumps(evidence, sort_keys=True, indent=2))
