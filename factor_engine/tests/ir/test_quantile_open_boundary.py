import pytest
from factor_engine.cleaned_operators.base import OperatorParameterError
from factor_engine.expr import CleanedCall, ColumnRef
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.sources.datasource import DataSource


class _NoReadSource(DataSource):
    def load_column(self, name):
        raise AssertionError("compile must not read source data")


def _compile(call):
    engine = FactorEngine(PandasBackend(), _NoReadSource(), run_mode="research")
    return engine.compile(Factor(name="tail_boundary", expr=call, surface="compat_research"))


@pytest.mark.parametrize("operator,parameter", [
    ("ts_quantilogram", "quantile"),
    ("ts_cross_quantilogram", "target_q"),
    ("ts_cross_extremogram", "source_q"),
])
def test_zero_tail_probability_rejected_before_execution(operator, parameter):
    args = (ColumnRef("ret"),)
    if operator.startswith("ts_cross_"):
        args += (ColumnRef("ret"),)
    call = CleanedCall(operator, args, ((parameter, 0.0),))
    with pytest.raises(OperatorParameterError, match=parameter):
        _compile(call)


@pytest.mark.parametrize("q", [1e-14, 0.1, 0.5])
def test_positive_tail_probability_keeps_open_interval(q):
    _compile(CleanedCall(
        "ts_quantilogram", (ColumnRef("ret"),), (("quantile", q),),
    ))
