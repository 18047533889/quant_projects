"""COS installation must not downgrade verified timezone-aware schema."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_access.cos_runtime import install_cos_runtime
from data_access.registry.loader import load_registry
from data_access.registry.schema_validation import _is_compatible


@pytest.mark.parametrize('dataset', [
    'ashare_stock_balance', 'ashare_stock_income',
    'ashare_stock_cashflow', 'ashare_stock_indicator',
])
def test_installed_cos_financial_schema_preserves_aware_update_time(dataset):
    root = Path(__file__).resolve().parents[3]
    registry = load_registry(root / 'data_access/config/datasets.yaml')
    store = SimpleNamespace(_registry=registry, _schema_checked={'stale'})
    install_cos_runtime(store)
    ds = store._registry.get(dataset)
    assert ds.schema['UpdateTime'] == 'timestamptz'
    assert _is_compatible(ds.schema['UpdateTime'], 'TIMESTAMP WITH TIME ZONE')
    assert not _is_compatible(ds.schema['UpdateTime'], 'TIMESTAMP')
    assert ds.time_column == 'PubDate'
    assert ds.schema['ReportPeriodEndDate'] == 'date'
    assert store._schema_checked == set()
    fingerprint = store._registry_hash
    assert install_cos_runtime(store) is store
    assert store._registry_hash == fingerprint
