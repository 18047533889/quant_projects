"""R6 contracts for the shareholder churn/network pack."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.shareholder import churn_network
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators import load_all


@pytest.fixture(scope="module", autouse=True)
def _finalized_registry():
    load_all()


def _inputs():
    index = pd.date_range("2024-01-02", periods=12, freq="D")
    columns = ["A", "B"]
    numeric = {}
    for rank in range(1, 11):
        numeric[f"s{rank}"] = pd.DataFrame(0.12 / rank, index=index, columns=columns)
        numeric[f"p{rank}"] = pd.DataFrame(0.11 / rank, index=index, columns=columns)
        current_id = f"H{rank}" if rank < 10 else "NEW"
        previous_id = f"H{rank}" if rank < 10 else "OLD"
        numeric[f"sid{rank}"] = pd.DataFrame(current_id, index=index, columns=columns, dtype=object)
        numeric[f"psid{rank}"] = pd.DataFrame(previous_id, index=index, columns=columns, dtype=object)
    base = pd.DataFrame(
        np.arange(1, 13, dtype=float)[:, None] * np.array([[1.0, 1.2]]),
        index=index, columns=columns,
    )
    snapshot = pd.DataFrame(
        np.asarray(index, dtype="datetime64[ns]")[:, None].repeat(2, axis=1),
        index=index, columns=columns, dtype=object,
    )
    aliases = {
        "pledge_shares": base, "freeze_shares": base, "locked_shares": base,
        "total_capital": pd.DataFrame(100.0, index=index, columns=columns),
        "pledge_ratio": base / 100.0, "top10_concentration": base / 20.0,
        "top10_float_concentration": base / 25.0, "concentration": (base * base) / 100.0,
        "snapshot_date": snapshot, "peer_return": base / 100.0,
        "own_return": base / 200.0, "overlap": pd.DataFrame(0.25, index=index, columns=columns),
        "breadth": base, "degree": base, "total": base + 10.0,
        "shared_holders": base, "total_holders": base + 5.0,
    }
    numeric.update(aliases)
    return numeric


def test_all_30_canonicals_declare_exact_topology_and_execute_finite() -> None:
    assert len(churn_network._CANONICALS) == 30
    values = _inputs()
    scalar_map = {
        "holder_pledge_change": ("lag",),
        "holder_concentration_slope": ("window",),
        "holder_concentration_acceleration": ("window",),
    }
    for canonical in churn_network._CANONICALS:
        operator = OperatorRegistry.get(canonical, backend="pandas_numpy")
        metadata = operator.metadata
        assert metadata.scalar_params == scalar_map.get(canonical, ()), canonical
        assert metadata.panel_arity == len(metadata.panel_params), canonical
        assert set(metadata.panel_params) | set(metadata.scalar_params) == set(metadata.param_names)
        kwargs = {name: values[name] for name in metadata.panel_params}
        result = operator.calculate(**kwargs)
        assert result.index.equals(values[metadata.panel_params[0]].index), canonical
        assert result.columns.equals(values[metadata.panel_params[0]].columns), canonical
        assert np.isfinite(result.to_numpy(dtype=float)).any(), canonical


def test_40_panel_id_family_preserves_object_ids_and_all_call_forms() -> None:
    values = _inputs()
    params = tuple(churn_network._ID_PARAMS)
    assert all(values[name].to_numpy().dtype == object for name in params if "sid" in name)
    args = [values[name] for name in params]
    operator = OperatorRegistry.get("holder_id_matched_churn", backend="pandas_numpy")
    positional = operator.calculate(*args)
    keyword = operator.calculate(**{name: values[name] for name in params})
    mixed = operator.calculate(*args[:20], **{name: values[name] for name in params[20:]})
    pd.testing.assert_frame_equal(positional, keyword)
    pd.testing.assert_frame_equal(positional, mixed)
    expected = 0.5 * (0.01 * sum(1.0 / rank for rank in range(1, 10)) + 0.12 / 10 + 0.11 / 10)
    assert positional.iloc[0, 0] == pytest.approx(expected)

    with pytest.raises((TypeError, ValueError), match="s1|required|missing"):
        operator.calculate(**{name: values[name] for name in params if name != "s1"})
    with pytest.raises((TypeError, ValueError), match="multiple|duplicate|s1"):
        operator.calculate(values["s1"], **{name: values[name] for name in params})
    with pytest.raises((TypeError, ValueError), match="undeclared|unknown|unexpected"):
        operator.calculate(*args, mystery=True)


def test_lag_window_and_snapshot_panel_contracts_are_real() -> None:
    values = _inputs()
    pledge = OperatorRegistry.get("holder_pledge_change", backend="pandas_numpy")
    assert pledge.metadata.panel_params == ("pledge_ratio",)
    assert pledge.metadata.param_specs["lag"].param_role is ParamRole.HORIZON
    pd.testing.assert_frame_equal(
        pledge.calculate(values["pledge_ratio"], 2),
        pledge.calculate(pledge_ratio=values["pledge_ratio"], lag=2),
    )
    with pytest.raises((TypeError, ValueError), match="lag"):
        pledge.calculate(values["pledge_ratio"], lag=0)

    slope = OperatorRegistry.get("holder_concentration_slope", backend="pandas_numpy")
    assert slope.metadata.panel_params == ("concentration", "snapshot_date")
    assert slope.metadata.scalar_params == ("window",)
    positional = slope.calculate(values["concentration"], 5, values["snapshot_date"])
    keyword = slope.calculate(
        concentration=values["concentration"], window=5,
        snapshot_date=values["snapshot_date"],
    )
    mixed = slope.calculate(values["concentration"], snapshot_date=values["snapshot_date"], window=5)
    pd.testing.assert_frame_equal(positional, keyword)
    pd.testing.assert_frame_equal(positional, mixed)
    assert positional.iloc[-1].notna().all()
