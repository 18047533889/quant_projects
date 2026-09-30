import sys
from types import SimpleNamespace

import numpy as np
import pytest

from factor_preprocess.adapters.factor_assets import (
    FactorAssetsAdapter,
    OptionalDependencyMissing,
    check_factor_assets_available,
    create_adapter,
)


class _Provider:
    def __init__(self, factor_ids):
        self.factor_ids = factor_ids

    def validate_factor_set(self, factor_set):
        return True

    def get_factor_batch(self, **kwargs):
        return {
            "values": np.zeros((1, 1, 2)),
            "dates": ["2026-09-30"],
            "assets": ["A"],
            "factor_ids": self.factor_ids,
            "metadata": {},
        }


def _factor_set():
    return SimpleNamespace(
        set_id="set-1", factor_ids=("f1", "f2"), universe_ref="u1",
        name="test", created_at="2026-09-30", frequency="daily", size=2,
    )


@pytest.mark.parametrize(
    ("factor_ids", "message"),
    [
        ("f1", "sequence of strings"),
        (42, "sequence of strings"),
        (["f1", 2], "only strings"),
        (["f1", ["f2"]], "only strings"),
    ],
)
def test_batch_factor_ids_must_be_string_sequence(factor_ids, message):
    provider = _Provider(factor_ids)
    with pytest.raises(ValueError, match=message):
        FactorAssetsAdapter(provider).load_factor_set(_factor_set())


def test_check_factor_assets_available_only_checks_package_import(monkeypatch):
    assert check_factor_assets_available() is True
    monkeypatch.setitem(sys.modules, "factor_assets", None)
    assert check_factor_assets_available() is False


def test_create_adapter_requires_provider_when_dependency_is_available(monkeypatch):
    monkeypatch.setattr(
        "factor_preprocess.adapters.factor_assets.check_factor_assets_available",
        lambda: True,
    )
    with pytest.raises(ValueError, match="FactorSetProvider must be supplied"):
        create_adapter()


def test_create_adapter_preserves_missing_dependency_error(monkeypatch):
    monkeypatch.setattr(
        "factor_preprocess.adapters.factor_assets.check_factor_assets_available",
        lambda: False,
    )
    with pytest.raises(OptionalDependencyMissing, match="factor_assets"):
        create_adapter()


def test_create_adapter_accepts_injected_provider_without_dependency_probe(monkeypatch):
    monkeypatch.setattr(
        "factor_preprocess.adapters.factor_assets.check_factor_assets_available",
        lambda: pytest.fail("injected provider must not require optional package"),
    )
    provider = _Provider(["f1", "f2"])
    assert create_adapter(provider=provider)._provider is provider
