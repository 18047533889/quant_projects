from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from factor_engine.backend.operator_capability import _json_contract_value
from factor_engine.cleaned_operators import registry as registry_module
from factor_engine.cleaned_operators.registry import OperatorRegistry, _BOOTSTRAP_TOKEN


def test_freeze_reuses_one_deepfrozen_catalog_without_weakening_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeze one catalog once; public reads and a later thaw stay detached."""
    class ProbeRegistry(OperatorRegistry):
        """Isolated registry state; production singleton remains untouched."""

    source_catalog = {
        "probe": {
            "backends": ["pandas_numpy"],
            "nested": {"values": [1, 2]},
            "array": np.array([3, 4], dtype=np.int64),
        }
    }
    expected_contract = _json_contract_value({
        "backends": source_catalog["probe"]["backends"],
        "nested": source_catalog["probe"]["nested"],
    })

    monkeypatch.setattr(
        ProbeRegistry,
        "_operators",
        {"probe": {"pandas_numpy": object()}},
    )
    monkeypatch.setattr(ProbeRegistry, "_aliases", {})
    monkeypatch.setattr(ProbeRegistry, "_catalog", source_catalog)
    monkeypatch.setattr(ProbeRegistry, "_frozen", None)
    monkeypatch.setattr(
        ProbeRegistry,
        "_lifecycle",
        ProbeRegistry.Lifecycle.FINALIZED,
    )
    monkeypatch.setattr(ProbeRegistry, "_mutation_token", None)
    monkeypatch.setattr(ProbeRegistry, "_version", 17)

    original_deepfreeze = registry_module._deepfreeze_catalog
    freeze_calls = 0

    def counted_deepfreeze(catalog):
        nonlocal freeze_calls
        freeze_calls += 1
        return original_deepfreeze(catalog)

    monkeypatch.setattr(registry_module, "_deepfreeze_catalog", counted_deepfreeze)

    ProbeRegistry.freeze()

    assert freeze_calls == 1
    assert isinstance(ProbeRegistry._catalog, MappingProxyType)
    assert ProbeRegistry._frozen is not None
    assert ProbeRegistry._frozen["catalog"] is ProbeRegistry._catalog
    assert _json_contract_value({
        "backends": ProbeRegistry._catalog["probe"]["backends"],
        "nested": ProbeRegistry._catalog["probe"]["nested"],
    }) == expected_contract

    exported = ProbeRegistry.catalog()
    exported["probe"]["nested"]["values"].append(99)
    exported["probe"]["array"][0] = 99
    assert ProbeRegistry.catalog()["probe"]["nested"]["values"] == [1, 2]
    assert ProbeRegistry.catalog()["probe"]["array"].tolist() == [3, 4]
    with pytest.raises(TypeError):
        ProbeRegistry._catalog["probe"]["nested"]["new"] = "blocked"

    frozen_snapshot = ProbeRegistry._frozen["catalog"]
    ProbeRegistry.thaw_for_bootstrap(_BOOTSTRAP_TOKEN)
    ProbeRegistry._catalog["probe"]["nested"]["values"].append(7)
    ProbeRegistry._catalog["probe"]["array"][0] = 8

    assert frozen_snapshot["probe"]["nested"]["values"] == (1, 2)
    assert frozen_snapshot["probe"]["array"][0] == 3
    assert _json_contract_value({
        "backends": frozen_snapshot["probe"]["backends"],
        "nested": frozen_snapshot["probe"]["nested"],
    }) == expected_contract
