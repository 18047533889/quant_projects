"""Invalid cohort bounds must never initialize a COS-reading engine."""
import importlib.util
from pathlib import Path
import pytest


def _loader():
    path = Path(__file__).parents[1] / 'examples' / 'cos_batch_audit.py'
    spec = importlib.util.spec_from_file_location('cos_admission_test_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kwargs', [
    {'n_assets': 5001}, {'n_assets': True}, {'n_days': 1261},
    {'n_days': True}, {'materialization_budget_bytes': 0},
    {'materialization_budget_bytes': True},
    {'materialization_budget_bytes': 16 * 1024**3 + 1},
])
def test_bad_bounds_rejected_before_cos_engine(monkeypatch, kwargs):
    module = _loader()

    def forbidden_engine(*args, **options):
        pytest.fail('invalid loader request initialized a COS engine')

    monkeypatch.setattr(module, 'DuckDBEngine', forbidden_engine)
    with pytest.raises(ValueError):
        module.load_cos_sample(**kwargs)
