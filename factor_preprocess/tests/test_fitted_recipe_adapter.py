from datetime import datetime, timezone
import hashlib

import numpy as np
import pandas as pd
import pytest

from factor_preprocess.adapters.fitted_recipe import (
    FITTED_STANDARDIZE_NAME,
    FITTED_STANDARDIZE_VERSION,
    apply_frozen_fitted_recipe,
    fitted_standardize_implementation_hash,
)
from factor_preprocess.contracts.state import FittedState, StateKind
from factor_preprocess.contracts.treatment_recipe import RecipeStep, TreatmentRecipe


DEFINITION = hashlib.sha256(b"factor-definition").hexdigest()


def _digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def _state(**overrides):
    values = dict(
        transform_name=FITTED_STANDARDIZE_NAME,
        transform_version=FITTED_STANDARDIZE_VERSION,
        fit_start_time=datetime(2023, 1, 1), fit_end_time=datetime(2023, 12, 31),
        state_kind=StateKind.FITTED,
        feature_ids=[DEFINITION], feature_order=[DEFINITION],
        learned_params={"mean": 2.0, "scale": 4.0},
        implementation_hash=fitted_standardize_implementation_hash(),
        data_snapshot_ref=_digest("snapshot"), split_ref=_digest("split"),
        universe_ref=_digest("universe"), calendar_ref=_digest("calendar"),
        fit_coordinate_hash=_digest("coordinates"), policy_hash=_digest("policy"),
        production=True,
    )
    values.update(overrides)
    return FittedState(**values)


def _recipe(state_ref):
    return TreatmentRecipe(
        recipe_id="fitted-standardize", source_factor_definition_ref=DEFINITION,
        source_factor_value_ref=_digest("value"),
        ordered_steps=(RecipeStep(
            "standardize", "TRAIN_STANDARDIZE:population-v1",
            FITTED_STANDARDIZE_NAME, "representation", requires_fit=True,
            state_ref=state_ref,
        ),),
    )


def _apply(state, *, recipe=None, definition=DEFINITION, start="2024-01-02"):
    recipe = recipe or _recipe(state.state_id)
    return apply_frozen_fitted_recipe(
        pd.DataFrame([[2.0, 6.0]], index=[pd.Timestamp("2024-01-02")]),
        recipe=recipe, fitted_states={recipe.ordered_steps[0].state_ref: state},
        factor_definition_hash=definition, decision_start=start,
    )


def test_apply_only_fitted_standardize_uses_exact_frozen_state():
    state = _state()
    result, refs = _apply(state)
    assert result.to_numpy().tolist() == [[0.0, 1.0]]
    assert refs == (state.state_id,)


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (_state(implementation_hash=_digest("wrong-implementation")), "implementation identity"),
        (_state(transform_version="2.0.0"), "transform version"),
        (_state(feature_ids=[_digest("other")], feature_order=[_digest("other")]), "factor-definition contract"),
        (_state(feature_ids=[DEFINITION, _digest("second")], feature_order=[_digest("second"), DEFINITION]), "factor-definition contract"),
    ],
)
def test_corrupt_implementation_version_feature_or_order_is_rejected(state, message):
    with pytest.raises(ValueError, match=message):
        _apply(state)


def test_state_ref_object_identity_mismatch_is_rejected():
    state = _state()
    recipe = _recipe("state-ref-that-is-not-content-derived")
    with pytest.raises(ValueError, match="state_ref differs"):
        _apply(state, recipe=recipe)


@pytest.mark.parametrize("start", [pd.NaT, "NaT"])
def test_nat_decision_boundary_is_rejected(start):
    with pytest.raises(ValueError, match="must not be NaT"):
        _apply(_state(), start=start)


def test_nat_fit_cutoff_is_rejected():
    state = _state(fit_end_time=pd.NaT)
    with pytest.raises(ValueError, match="must not be NaT"):
        _apply(state)


def test_mixed_timezone_policy_is_rejected_instead_of_assuming_utc():
    aware = _state(
        fit_start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
        fit_end_time=datetime(2023, 12, 31, tzinfo=timezone.utc),
    )
    with pytest.raises(ValueError, match="same timezone policy"):
        _apply(aware, start="2024-01-02")


def test_fit_cutoff_must_precede_decision_boundary():
    with pytest.raises(ValueError, match="strictly before"):
        _apply(_state(), start="2023-12-31")


@pytest.mark.parametrize("bad", [True, "2.0", [2.0], np.array([2.0])])
@pytest.mark.parametrize("field", ["mean", "scale"])
def test_learned_parameters_require_real_nonboolean_numeric_scalars(field, bad):
    learned = {"mean": 2.0, "scale": 4.0}
    learned[field] = bad
    state = _state(learned_params=learned)
    with pytest.raises(ValueError, match=f"{field} must be a numeric scalar"):
        _apply(state)
