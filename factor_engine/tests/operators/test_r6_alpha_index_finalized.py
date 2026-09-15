"""Full-bootstrap numerical acceptance for alpha events and index/listing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()

NAMES = (
    "event_frequency event_cluster_count event_cluster_mean_size report_rolling_mean "
    "report_yoy_lag report_change_breadth report_change_coherence "
    "index_weight_gap_to_free_float index_reconstitution_churn multi_index_entry_intensity "
    "listing_age suspension_frequency suspension_status_coverage index_event_decay"
).split()


def _calls():
    idx = pd.date_range("2020-01-01", periods=120)
    event = pd.DataFrame({"A": ((np.arange(120) % 7) < 2).astype(float)}, index=idx)
    periods = np.repeat(pd.date_range("2018-03-31", periods=12, freq="QE"), 10)
    pid = pd.DataFrame({"A": periods}, index=idx)
    t = np.arange(120, dtype=float)
    f1 = pd.DataFrame({"A": 100 + t + 2 * np.sin(t)}, index=idx)
    f2 = pd.DataFrame({"A": 80 + .8 * t + np.cos(t * .7)}, index=idx)
    f3 = pd.DataFrame({"A": 60 + .5 * t + np.sin(t * .3)}, index=idx)
    member = pd.DataFrame({"A": ((np.arange(120) // 8) % 2).astype(float)}, index=idx)
    weight = pd.DataFrame({"A": .1 + t * .001}, index=idx)
    free = weight * .8
    listing = pd.DataFrame({"A": [pd.Timestamp("2020-01-04")] * 120}, index=idx)
    entry = member.diff().fillna(0.0)
    return {
        "event_frequency": (event, 20, 1),
        "event_cluster_count": (event, 60, 3),
        "event_cluster_mean_size": (event, 60, 3),
        "report_rolling_mean": (f1, pid, 8),
        "report_yoy_lag": (f1, pid, 4),
        "report_change_breadth": (f1, f2, f3, pid, 1, .5),
        "report_change_coherence": (f1, f2, f3, pid, 1),
        "index_weight_gap_to_free_float": (weight, free),
        "index_reconstitution_churn": (member, 20),
        "multi_index_entry_intensity": (member, member, member, 20),
        "listing_age": (listing,),
        "suspension_frequency": (member, 20),
        "suspension_status_coverage": (member, 20),
        "index_event_decay": (entry, 20, .9, "break"),
    }


def test_all_14_finalized_numeric_positional_keyword_and_prefix_causal():
    calls = _calls()
    assert set(calls) == set(NAMES) and len(set(NAMES)) == 14
    for name, args in calls.items():
        op = OperatorRegistry.get(name)
        assert op is not None
        kwargs = dict(zip(op.metadata.param_names, args))
        pos = op.calculate(*args)
        kw = op.calculate(**kwargs)
        pd.testing.assert_frame_equal(pos, kw, check_dtype=False, obj=name)
        assert pos.shape == (120, 1) and np.isfinite(pos.to_numpy()).any(), name
        prefix_args = tuple(a.iloc[:80] if isinstance(a, pd.DataFrame) else a for a in args)
        pd.testing.assert_frame_equal(op.calculate(*prefix_args), pos.iloc[:80], check_dtype=False, obj=name)


def test_independent_event_and_index_oracles():
    calls = _calls()
    freq = OperatorRegistry.get("event_frequency").calculate(*calls["event_frequency"])
    event = calls["event_frequency"][0]
    assert freq.iloc[-1, 0] == pytest.approx(event.iloc[-20:, 0].mean())
    gap = OperatorRegistry.get("index_weight_gap_to_free_float").calculate(*calls["index_weight_gap_to_free_float"])
    assert np.allclose(gap.to_numpy(), .25)
    age = OperatorRegistry.get("listing_age").calculate(*calls["listing_age"])
    assert age.iloc[2, 0] != age.iloc[2, 0]
    assert age.iloc[3, 0] == 0.0 and age.iloc[4, 0] == 1.0
    decay = OperatorRegistry.get("index_event_decay").calculate(*calls["index_event_decay"])
    entry = calls["index_event_decay"][0].iloc[:, 0].to_numpy()
    expected = float(np.dot(entry[-20:][::-1], .9 ** np.arange(20)))
    assert decay.iloc[-1, 0] == pytest.approx(expected)


@pytest.mark.parametrize("name,args", [
    ("event_frequency", (pd.DataFrame({"A": [0.0, 2.0]}), 2, 1)),
    ("event_cluster_count", (pd.DataFrame({"A": [0.0, 1.0]}), 2, -1)),
    ("report_change_breadth", (None, None, None, None, 1, float("nan"))),
    ("index_reconstitution_churn", (pd.DataFrame({"A": [0.0, 2.0]}), 2)),
    ("index_event_decay", (pd.DataFrame({"A": [0.0, 2.0]}), 2, .9, "break")),
])
def test_invalid_contract_values_rejected(name, args):
    with pytest.raises((TypeError, ValueError)):
        OperatorRegistry.get(name).calculate(*args)
