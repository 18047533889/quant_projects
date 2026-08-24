from __future__ import annotations

from data_access.cos_registry_runtime import patch_registry, registry_patch_fingerprint
from data_access.registry import load_registry


def test_verified_registry_axes_and_schemas_are_patched():
    registry = patch_registry(load_registry())
    assert registry.get("ashare_stock_balance").time_column == "PubDate"
    assert "PubDate" in registry.get("ashare_stock_balance").schema
    daily = registry.get("ashare_stock_daily").schema
    assert {"HighLimit", "LowLimit", "Factor", "IsSuspend"} <= set(daily)
    status = registry.get("ashare_stock_status").schema
    assert "PublicStatus" in status
    assert "ListedState" not in status
    assert registry.get("us_stock_valuation_daily").instrument_column == "ticker"
    assert registry_patch_fingerprint()
