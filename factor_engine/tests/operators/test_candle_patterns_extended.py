from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.operator_policy import infer_operator_policy


def _frame(values):
    arr=np.asarray(values,dtype=float)
    if arr.ndim==1: arr=arr[:,None]
    return pd.DataFrame(arr,index=pd.date_range("2024-01-02",periods=arr.shape[0],freq="B"),columns=["A"])


def _op(name):
    load_all(); op=OperatorRegistry.get(name,"pandas_numpy"); assert op is not None; return op


def test_harami_and_star_patterns_use_no_future_bar() -> None:
    open_=_frame([12,9,9.5,10,10.2,10.1])
    close=_frame([9,10,9.6,10.5,10.0,10.3])
    high=_frame([12.5,10.5,10.0,11.0,10.6,10.7])
    low=_frame([8.5,8.8,9.2,9.4,9.8,9.9])
    for name in ("cdl_harami","cdl_harami_cross","cdl_morning_star","cdl_evening_star"):
        op=_op(name)
        full=op.calculate(open_,high,low,close)
        prefix=op.calculate(open_.iloc[:5],high.iloc[:5],low.iloc[:5],close.iloc[:5])
        np.testing.assert_allclose(full.iloc[:5].to_numpy(),prefix.to_numpy(),equal_nan=True)


def test_three_bar_patterns_have_two_bar_policy_lag() -> None:
    for name in ("cdl_morning_star","cdl_evening_star","cdl_three_white_soldiers","cdl_three_black_crows"):
        op=_op(name)
        policy=infer_operator_policy(op,canonical=name)
        assert policy.pit_safe is True
        assert policy.lag >= 2


def test_two_bar_patterns_have_one_bar_policy_lag() -> None:
    for name in ("cdl_harami","cdl_piercing","cdl_dark_cloud_cover","cdl_tweezer_top","cdl_tweezer_bottom"):
        op=_op(name)
        policy=infer_operator_policy(op,canonical=name)
        assert policy.pit_safe is True
        assert policy.lag >= 1


def test_extended_patterns_are_numeric_panels() -> None:
    open_=_frame([10,9,10]); close=_frame([9,11,10.5]); high=_frame([10.5,11.5,11]); low=_frame([8.5,8.5,9.5])
    names=(
        "cdl_dragonfly_doji","cdl_gravestone_doji","cdl_hanging_man","cdl_harami",
        "cdl_piercing","cdl_dark_cloud_cover","cdl_tweezer_top","cdl_tweezer_bottom",
    )
    for name in names:
        out=_op(name).calculate(open_,high,low,close)
        assert isinstance(out,pd.DataFrame)
        assert out.shape==open_.shape
        assert out.to_numpy().dtype.kind in "fi"
