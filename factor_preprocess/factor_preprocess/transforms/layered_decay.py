"""Research-only lagged, normalized decay with observation-origin layer states."""
from __future__ import annotations

import math
import numbers
import numpy as np
from factor_preprocess.transforms.layered_decay_state import LayeredDecayState


def layered_decay(values, layers, half_lives, *, allow_research=False):
    """Filter T x N values using strictly previous-row values and layer labels.

    Each observation retains its entry layer's half-life. Numerator and weight
    states are independent per asset/layer; output divides their sums. Missing
    input or layer -1 resets that asset, preventing stale history resurrection.
    This is adjusted normalized decay, not the ordinary adjust=False EWMA.
    Layers must be signal-only assignments made by the caller, not labels.
    """
    if allow_research is not True:
        raise ValueError("explicit research opt-in required")
    if isinstance(half_lives, (str, bytes)):
        raise ValueError("half_lives must be a numeric sequence")
    half_lives = tuple(half_lives)
    if not 1 <= len(half_lives) <= 20 or any(
        isinstance(h, (bool, np.bool_)) or not isinstance(h, numbers.Real)
        or not math.isfinite(h) or not 1 <= h <= 60 for h in half_lives):
        raise ValueError("1 to 20 finite half_lives in [1,60] are required")
    x, bins = np.asarray(values, dtype=float), np.asarray(layers)
    if x.ndim != 2 or bins.shape != x.shape:
        raise ValueError("matching T x N value and layer panels required")
    if bins.dtype.kind not in "iu" or np.any(bins < -1) or np.any(bins >= len(half_lives)):
        raise ValueError("integer layers must be -1 or a declared layer index")
    state = LayeredDecayState(x.shape[1], half_lives)
    out = np.full(x.shape, np.nan)
    for t in range(1, len(x)):
        out[t] = state.step(x[t-1], bins[t-1])
    return out
