import pandas as pd
import pytest

from factor_engine.storage.materialize.write_targets import LocalParquetWriteTarget
from factor_engine.runtime import production_policy


@pytest.mark.parametrize("error", [ImportError("authority missing"), RuntimeError("authority broken")])
def test_local_target_authority_error_precedes_frame_access(monkeypatch, error):
    def fail():
        raise error
    monkeypatch.setattr(production_policy, "is_production_mode", fail)
    target = LocalParquetWriteTarget.__new__(LocalParquetWriteTarget)
    with pytest.raises(type(error), match=str(error)):
        target.write_factor_frame("f", None)


def test_local_target_production_rejected_before_frame_access(monkeypatch):
    monkeypatch.setattr(production_policy, "is_production_mode", lambda: True)
    target = LocalParquetWriteTarget.__new__(LocalParquetWriteTarget)
    with pytest.raises(ValueError, match="production 禁止 direct-local"):
        target.write_factor_frame("f", None)


def test_local_target_research_preserves_empty_write(monkeypatch):
    monkeypatch.setattr(production_policy, "is_production_mode", lambda: False)
    target = LocalParquetWriteTarget.__new__(LocalParquetWriteTarget)
    assert target.write_factor_frame("f", pd.DataFrame()) == {
        "factor_id": "f", "rows_written": 0, "target": "local"
    }
