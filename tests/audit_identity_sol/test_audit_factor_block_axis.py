import pandas as pd
import pytest

from factor_engine.runtime.factor_block_ref import axis_key_of, build_factor_block, share_axis_across


def test_axis_key_binds_labels_order_and_row_pairing():
    a=pd.Index(["A","B"],name="instrument")
    assert axis_key_of(a)!=axis_key_of(pd.Index(["X","Y"],name="instrument"))
    assert axis_key_of(a)!=axis_key_of(a[::-1])
    t1,t2=pd.Timestamp("2026-03-02"),pd.Timestamp("2026-03-03")
    x=pd.MultiIndex.from_tuples([(t1,"A"),(t2,"B")],names=["timestamp","instrument"])
    y=pd.MultiIndex.from_tuples([(t1,"B"),(t2,"A")],names=["timestamp","instrument"])
    assert axis_key_of(x)!=axis_key_of(y)


def test_axis_identity_rejects_unsupported_object_labels():
    class Unsupported: pass
    with pytest.raises(Exception,match="Cannot encode"):
        axis_key_of(pd.Index([Unsupported()],dtype=object))


def test_share_axis_requires_verified_index_equality():
    a=pd.Index(["A","B"],name="instrument")
    x=build_factor_block(["f"],{"f":pd.Series([1.,2.],index=a)},a)
    b=pd.Index(["X","Y"],name="instrument")
    y=build_factor_block(["f"],{"f":pd.Series([1.,2.],index=b)},b)
    assert not share_axis_across([x,y])


def test_series_index_mismatch_is_rejected_not_positionally_loaded():
    declared=pd.Index(["A","B"],name="instrument")
    source=pd.Series([20.,10.],index=declared[::-1])
    with pytest.raises(ValueError,match="does not exactly match"):
        build_factor_block(["f"],{"f":source},declared)


def test_equal_distinct_index_objects_are_accepted():
    declared=pd.Index(["A","B"],name="instrument")
    equal=pd.Index(["A","B"],name="instrument")
    block=build_factor_block(["f"],{"f":pd.Series([1.,2.],index=equal)},declared)
    assert block.axis_ref.index.equals(equal)
