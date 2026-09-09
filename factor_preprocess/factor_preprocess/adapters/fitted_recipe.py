"""Apply-only execution of narrowly supported frozen FP fitted state."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import inspect
import numbers
from typing import Any

import numpy as np
import pandas as pd

from factor_preprocess.contracts.state import FittedState, StateKind


FITTED_STANDARDIZE_NAME = "train_standardize"
FITTED_STANDARDIZE_VERSION = "1.0.0"


def _apply_train_standardize(values: np.ndarray, *, mean: float, scale: float) -> np.ndarray:
    return (values - mean) / scale


def fitted_standardize_implementation_hash() -> str:
    """Content identity of the exact apply-only numeric implementation."""
    return hashlib.sha256(inspect.getsource(_apply_train_standardize).encode("utf-8")).hexdigest()


def apply_frozen_fitted_recipe(
    panel: pd.DataFrame, *, recipe: Any,
    fitted_states: Mapping[str, FittedState], factor_definition_hash: str,
    decision_start: Any,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Apply one frozen train-standardization state; fitting is impossible here."""
    if len(recipe.ordered_steps) != 1:
        raise ValueError("fitted recipe adapter supports exactly one fitted step")
    step = recipe.ordered_steps[0]
    if not step.requires_fit or step.implementation_ref != FITTED_STANDARDIZE_NAME:
        raise ValueError("unsupported fitted recipe implementation")
    if dict(step.parameters):
        raise ValueError("fitted standardization accepts no runtime parameters")
    if not step.state_ref:
        raise ValueError("fitted recipe step has no state_ref")
    state = fitted_states.get(step.state_ref)
    if state is None:
        raise ValueError("missing fitted state for frozen recipe")
    if not isinstance(state, FittedState) or not state.production:
        raise ValueError("frozen recipe requires a production FittedState")
    if state.state_kind is not StateKind.FITTED:
        raise ValueError("frozen recipe state is not FITTED")
    if state.state_id != step.state_ref:
        raise ValueError("recipe state_ref differs from fitted state identity")
    if state.transform_name != FITTED_STANDARDIZE_NAME:
        raise ValueError("fitted state transform identity mismatch")
    if state.transform_version != FITTED_STANDARDIZE_VERSION:
        raise ValueError("unsupported fitted state transform version")
    if state.implementation_hash != fitted_standardize_implementation_hash():
        raise ValueError("fitted state implementation identity mismatch")
    if not factor_definition_hash:
        raise ValueError("factor definition hash is required")
    if not state.is_compatible_with([factor_definition_hash]):
        raise ValueError("fitted state factor-definition contract mismatch")
    start = pd.Timestamp(decision_start)
    fit_end = pd.Timestamp(state.fit_end_time)
    if pd.isna(start) or pd.isna(fit_end):
        raise ValueError("decision and fit cutoff timestamps must not be NaT")
    if (start.tzinfo is None) != (fit_end.tzinfo is None):
        raise ValueError("decision and fit cutoff must use the same timezone policy")
    if start <= fit_end:
        raise ValueError("fitted state cutoff must be strictly before decision segment")
    learned = dict(state.learned_params)
    if set(learned) != {"mean", "scale"}:
        raise ValueError("fitted standardization state requires exact mean/scale parameters")
    mean_raw, scale_raw = learned["mean"], learned["scale"]
    if (isinstance(mean_raw, (bool, np.bool_))
            or not isinstance(mean_raw, numbers.Real)):
        raise ValueError("fitted standardization mean must be a numeric scalar")
    if (isinstance(scale_raw, (bool, np.bool_))
            or not isinstance(scale_raw, numbers.Real)):
        raise ValueError("fitted standardization scale must be a numeric scalar")
    mean = float(mean_raw)
    scale = float(scale_raw)
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("fitted standardization mean/scale are invalid")
    values = panel.to_numpy(dtype=np.float64, copy=True)
    transformed = _apply_train_standardize(values, mean=mean, scale=scale)
    return pd.DataFrame(transformed, index=panel.index, columns=panel.columns), (state.state_id,)


__all__ = [
    "FITTED_STANDARDIZE_NAME", "FITTED_STANDARDIZE_VERSION",
    "apply_frozen_fitted_recipe", "fitted_standardize_implementation_hash",
]
