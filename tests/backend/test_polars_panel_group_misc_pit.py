from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl


_MODULE_PATH = (
    Path(__file__).parents[2]
    / "cleaned_operators"
    / "polars_native"
    / "panel_group_misc.py"
)

# Create a unique module name to avoid conflicts
_MODULE_NAME = "panel_group_misc_under_test"
if _MODULE_NAME in sys.modules:
    del sys.modules[_MODULE_NAME]

_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _MODULE
_SPEC.loader.exec_module(_MODULE)
group_tail_lead_score = _MODULE.group_tail_lead_score


def test_group_tail_lead_score_suffix_poison_preserves_prefix() -> None:
    rows = 24
    cut = 16
    base = np.array(
        [
            np.sin(np.arange(rows) * 0.7),
            np.cos(np.arange(rows) * 0.5),
            np.sin(np.arange(rows) * 0.4 + 1.0),
        ]
    ).T
    poisoned = base.copy()
    poisoned[cut:] = np.array([1e9, -1e9, 5e8])
    groups = pl.DataFrame({name: ["sector"] * rows for name in ("a", "b", "c")})

    def calculate(values: np.ndarray) -> np.ndarray:
        frame = pl.DataFrame(
            {name: values[:, i] for i, name in enumerate(("a", "b", "c"))}
        )
        result = group_tail_lead_score()._calculate_series(
            frame,
            groups,
            window=8,
            quantile=0.25,
            side="lower",
            lag=1,
            min_periods=3,
            min_conditioning_events=1,
        )
        return result.to_numpy()

    actual = calculate(base)
    poisoned_actual = calculate(poisoned)

    np.testing.assert_allclose(actual[:cut], poisoned_actual[:cut], equal_nan=True)
    assert np.isfinite(actual[:cut]).any()
