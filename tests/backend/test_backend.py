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


def test_debug_backend_render():
    factor = Factor(name="demo", expr=rank(ts_mean(col("close"), 5)))
    out = FactorEngine(backend=DebugBackend(), data_source=DummySource()).run(factor)
    text = out["result"]
    assert "rank" in text
    assert "ts_mean" in text
    assert "column" in text
