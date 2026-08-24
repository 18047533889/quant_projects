from dataclasses import dataclass

from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.api import rank, ts_mean
from factor_engine.backend.debug_backend import DebugBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.datasource import DataSource


@dataclass
class DummySource(DataSource):
    def load_column(self, name: str):
        raise NotImplementedError


def test_end_to_end_compile_run():
    expr = rank(ts_mean(col("close"), 20) - ts_mean(col("close"), 5))
    factor = Factor(name="mom_20_5_rank", expr=expr)
    engine = FactorEngine(backend=DebugBackend(), data_source=DummySource())
    out = engine.run(factor)

    assert out["factor"].name == "mom_20_5_rank"
    assert out["analysis"].lookback == 20
    assert "sub" in out["result"]
