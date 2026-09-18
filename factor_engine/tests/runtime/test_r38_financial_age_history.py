import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.fundamental.expectation_v2 import (
    fin_days_since_expectation_revision,
)
from factor_engine.cleaned_operators.fundamental.transforms_repairs_v2 import (
    fin_days_since_update,
    fin_staleness,
)

from factor_engine.runtime.execution_contract import (
    execution_contract, forward_impact, history_requirement, own_history_requirement,
)
from factor_engine.runtime.incremental_contract import (
    IncrementalMode, classify_incremental_mode, resolve_incremental_contract,
)


def _panel(values):
    return pd.DataFrame({"a": values}, index=pd.RangeIndex(len(values)))


@pytest.mark.parametrize(
    "canonical",
    ["fin_days_since_update", "fin_staleness", "fin_days_since_expectation_revision"],
)
def test_age_contract_is_full_replay_even_when_max_days_is_bound(canonical):
    params = {"max_days": 3}
    contract = execution_contract(canonical, production=True)
    assert contract.state_model == "recursive"
    assert contract.requires_full_history
    assert history_requirement(canonical, params, production=True).is_full_history
    assert own_history_requirement(canonical, params).is_full_history
    assert forward_impact(canonical, params, production=True) is None
    assert classify_incremental_mode(canonical, params) is IncrementalMode.FULL_REPLAY
    resolved = resolve_incremental_contract(canonical, params, production=True)
    assert resolved.incremental_mode is IncrementalMode.FULL_REPLAY
    assert resolved.forward_impact is None


@pytest.mark.parametrize("kernel", [fin_days_since_update, fin_staleness])
def test_update_age_same_suffix_depends_on_prefix_beyond_cap(kernel):
    cap = 3
    # Both runs end in the identical cap-length suffix.  In the first run an
    # observed update in the prefix started the clock; in the second, the same
    # value is only a left-censored anchor.
    started_x = _panel([10.0, 20.0, 20.0, 20.0, 20.0])
    censored_x = _panel([20.0, 20.0, 20.0, 20.0, 20.0])
    period = _panel([1, 1, 1, 1, 1])
    started = kernel(started_x, period, max_days=cap)["a"].to_numpy()
    censored = kernel(censored_x, period, max_days=cap)["a"].to_numpy()
    np.testing.assert_allclose(started[-cap:], [1.0, 2.0, 3.0])
    assert np.isnan(censored[-cap:]).all()
    np.testing.assert_array_equal(started_x["a"].to_numpy()[-cap:], censored_x["a"].to_numpy()[-cap:])


def test_expectation_age_same_suffix_depends_on_prefix_beyond_cap():
    cap = 3
    started = _panel([10.0, 12.0, 12.0, 12.0, 12.0])
    censored = _panel([12.0, 12.0, 12.0, 12.0, 12.0])
    period = _panel([1, 1, 1, 1, 1])
    started_age = fin_days_since_expectation_revision(
        started, period, max_days=cap
    )["a"].to_numpy()
    censored_age = fin_days_since_expectation_revision(
        censored, period, max_days=cap
    )["a"].to_numpy()
    np.testing.assert_allclose(started_age[-cap:], [1.0, 2.0, 3.0])
    assert np.isnan(censored_age[-cap:]).all()


@pytest.mark.parametrize("kernel", [fin_days_since_update, fin_staleness])
def test_update_age_retains_confirmed_state_across_provider_gap(kernel):
    x = _panel([10.0, 20.0, 20.0, np.nan, np.nan, 20.0, 25.0, 25.0])
    period = _panel([1, 1, 1, np.nan, np.nan, 1, 1, 1])
    age = kernel(x, period, max_days=20)["a"].to_numpy()
    np.testing.assert_allclose(
        age, [np.nan, 0.0, 1.0, np.nan, np.nan, 4.0, 0.0, 1.0],
        equal_nan=True,
    )


def test_expectation_age_gap_restarts_censored_until_new_revision():
    expected = _panel([10.0, 12.0, 12.0, np.nan, 12.0, 12.0, 15.0, 15.0])
    period = _panel([1, 1, 1, np.nan, 1, 1, 1, 1])
    age = fin_days_since_expectation_revision(
        expected, period, max_days=20
    )["a"].to_numpy()
    np.testing.assert_allclose(
        age, [np.nan, 0.0, 1.0, np.nan, np.nan, np.nan, 0.0, 1.0],
        equal_nan=True,
    )


def test_period_transition_is_update_but_not_expectation_revision():
    values = _panel([10.0, 12.0, 12.0, 12.0])
    period = _panel([1, 1, 2, 2])
    update_age = fin_days_since_update(values, period, max_days=20)["a"].to_numpy()
    revision_age = fin_days_since_expectation_revision(
        values, period, max_days=20
    )["a"].to_numpy()
    np.testing.assert_allclose(update_age, [np.nan, 0.0, 0.0, 1.0], equal_nan=True)
    np.testing.assert_allclose(revision_age, [np.nan, 0.0, 1.0, 2.0], equal_nan=True)
