"""Named panel/scalar topology must not force every argument positional."""
import pandas as pd
import pytest
from factor_engine.cleaned_operators.base import OperatorMetadata, ParamSpec, SeriesOperator

class Rolling(SeriesOperator):
    metadata = OperatorMetadata(
        name="r6_named_rolling", category="test", param_names=["x", "window"],
        panel_params=("x",), scalar_params=("window",),
        param_specs={"window": ParamSpec(dtype=int, min=1, max=4)},
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 2):
        return x.rolling(window, min_periods=1).mean()

@pytest.mark.parametrize("style", ["default", "positional", "mixed", "keywords"])
def test_valid_named_call_styles(style):
    x = pd.DataFrame({"a": [1., 3., 5.]})
    op = Rolling()
    if style == "default":
        result = op.calculate(x)
    elif style == "positional":
        result = op.calculate(x, 2)
    elif style == "mixed":
        result = op.calculate(x, window=2)
    else:
        result = op.calculate(x=x, window=2)
    pd.testing.assert_frame_equal(result, x.rolling(2, min_periods=1).mean())

@pytest.mark.parametrize("case", ["missing", "duplicate", "unknown", "extra", "bool", "fraction"])
def test_invalid_named_calls_rejected(case):
    x = pd.DataFrame({"a": [1., 3., 5.]})
    args, kwargs = {
        "missing": ((), {"window": 2}),
        "duplicate": ((x,), {"x": x}),
        "unknown": ((x,), {"hidden": 1}),
        "extra": ((x, 2, 3), {}),
        "bool": ((x, True), {}),
        "fraction": ((x,), {"window": 2.5}),
    }[case]
    with pytest.raises((ValueError, TypeError)):
        Rolling().calculate(*args, **kwargs)


@pytest.mark.parametrize("style", ["default", "positional", "mixed", "keywords"])
def test_polars_metadata_constructor_and_named_calls(style):
    import polars as pl
    from factor_engine.cleaned_operators.base_polars import OperatorMetadata as PMeta
    from factor_engine.cleaned_operators.base_polars import SeriesOperator as POperator
    class PolarsRolling(POperator):
        metadata = PMeta(
            name="r6_polars_named", category="test", param_names=["x", "window"],
            panel_params=("x",), scalar_params=("window",), panel_arity=1,
            param_specs={"window": ParamSpec(dtype=int, min=1, max=4)},
        )
        def _calculate_series(self, x, window=2):
            return x.with_columns(pl.all().rolling_mean(window_size=window, min_samples=1))
    x = pl.DataFrame({"a": [1., 3., 5.]})
    op = PolarsRolling()
    if style == "default":
        result = op.calculate(x)
    elif style == "positional":
        result = op.calculate(x, 2)
    elif style == "mixed":
        result = op.calculate(x, window=2)
    else:
        result = op.calculate(x=x, window=2)
    assert result["a"].to_list() == [1., 2., 4.]


@pytest.mark.parametrize("kwargs", [{"k": 2, "window": 3}, {"window": 3, "k": 2}, {"k": 2}])
def test_declared_k_is_not_rewritten_as_legacy_window_alias(kwargs):
    from factor_engine.cleaned_operators.base import bind_operator_call
    class DistinctK(SeriesOperator):
        metadata = OperatorMetadata(
            name="r6_distinct_k", category="test", param_names=["x", "k", "window"],
            panel_params=("x",), scalar_params=("k", "window"),
            param_specs={"k": ParamSpec(dtype=int, min=1, default=1),
                         "window": ParamSpec(dtype=int, min=1, default=20)},
        )
        def _calculate_series(self, x, k=1, window=20):
            return x * k + window
    op = DistinctK()
    x = pd.DataFrame({"A": [1., 3.]})
    expected_window = kwargs.get("window", 20)
    pd.testing.assert_frame_equal(op.calculate(x, **kwargs), x*2+expected_window)
    bound = bind_operator_call(op, (x,), kwargs).bound
    assert bound.normalized_values["k"] == 2
    assert bound.normalized_values["window"] == expected_window
    assert "k" not in bound.canonical_aliases_resolved


def test_bound_kernel_defaults_reach_relational_validation():
    from factor_engine.cleaned_operators.base import RelationalParamSpec, _kernel_param_defaults
    def kernel(x, window=20, bins=5, min_per_bin=3):
        return x * min_per_bin
    class Binned(SeriesOperator):
        metadata=OperatorMetadata(
            name="r6_default_relation", category="test",
            param_names=["x","window","bins","min_per_bin"],
            panel_params=("x",), scalar_params=("window","bins","min_per_bin"),
            param_specs={p:ParamSpec(dtype=int,min=1) for p in ("window","bins","min_per_bin")},
            relational_specs=[RelationalParamSpec(expression="window >= bins * min_per_bin",
                                                 message="insufficient configured samples")],
        )
        _contract_callable=staticmethod(kernel)
        def _calculate_series(self,*args,**kwargs):
            return kernel(*args,**kwargs)
    op=Binned()
    x=pd.DataFrame({"A":[1.,2.]})
    assert _kernel_param_defaults(op)=={"window":20,"bins":5,"min_per_bin":3}
    pd.testing.assert_frame_equal(op.calculate(x,window=20,bins=5),x*3)
    with pytest.raises(ValueError,match="insufficient"):
        op.calculate(x,window=10,bins=5)
