from dataclasses import dataclass

from api.columns import col
from api.factor import Factor
from api import rank, ts_mean
from backend.debug_backend import DebugBackend
from runtime.engine import FactorEngine
from storage.datasource import DataSource


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
