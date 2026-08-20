from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from cleaned_operators.base import ParamRole


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "cleaned_operators/common/polars_ts_stats.py"
)


def _load_by_path(monkeypatch, *, bridge_threshold: bool = False):
    import cleaned_operators.base as base
    import cleaned_operators.base_polars as base_polars

    if bridge_threshold:
        class _ImportProbeParamRole:
            """Expose only the independently missing legacy threshold alias."""

        for role in ParamRole:
            setattr(_ImportProbeParamRole, role.name, role)
        _ImportProbeParamRole.THRESHOLD = ParamRole.STATE_THRESHOLD
        monkeypatch.setattr(base, "ParamRole", _ImportProbeParamRole)

    monkeypatch.setattr(
        base_polars,
        "register_operator",
        lambda **_kwargs: lambda cls: cls,
    )
    module_name = "_test_common_polars_ts_stats"
    spec = importlib.util.spec_from_file_location(module_name, _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_tuning_parameters_use_authoritative_estimator_role(monkeypatch) -> None:
    module = _load_by_path(monkeypatch, bridge_threshold=True)

    assert (
        module.TSTrimmedMeanNative.metadata.param_specs["trim_pct"].param_role
        is ParamRole.ESTIMATOR_RESOLUTION
    )
    for operator in (
        module.TSLowerPartialMomentNative,
        module.TSUpperPartialMomentNative,
    ):
        assert (
            operator.metadata.param_specs["order"].param_role
            is ParamRole.ESTIMATOR_RESOLUTION
        )


def test_independent_threshold_role_divergence_fails_loudly(monkeypatch) -> None:
    with pytest.raises(AttributeError, match="THRESHOLD"):
        _load_by_path(monkeypatch)
