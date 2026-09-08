from __future__ import annotations

import pytest

from factor_engine.cleaned_operators.base import ParamRole, ParamSpec
from factor_engine.cleaned_operators.base_polars import (
    OperatorMetadata as PolarsMetadata,
    SeriesOperator as PolarsSeriesOperator,
)
from factor_engine.cleaned_operators.registry import _backfill_logical_contract


def _operator_with_role(role: ParamRole | None) -> PolarsSeriesOperator:
    class _Native(PolarsSeriesOperator):
        metadata = PolarsMetadata(
            name="__v7_param_role_oracle",
            category="test",
            param_names=["x", "window"],
            param_specs={
                "window": ParamSpec(dtype=int, min=2, param_role=role),
            },
        )

        def _calculate_series(self, *args, **kwargs):  # pragma: no cover - stub
            return None

    return _Native()


def test_backend_param_spec_inherits_omitted_canonical_role() -> None:
    canonical = {
        "param_specs": {
            "window": ParamSpec(
                dtype=int,
                min=2,
                param_role=ParamRole.HORIZON,
            ),
        },
    }
    backend = _operator_with_role(None)

    _backfill_logical_contract(backend, canonical)

    assert backend.metadata.param_specs["window"].param_role is ParamRole.HORIZON
    assert backend.metadata.param_specs["window"].dtype is int
    assert backend.metadata.param_specs["window"].min == 2


def test_backend_param_spec_preserves_and_reports_explicit_role_conflict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    canonical = {
        "param_specs": {
            "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
        },
    }

    backend = _operator_with_role(ParamRole.THRESHOLD)

    _backfill_logical_contract(backend, canonical)

    assert backend.metadata.param_specs["window"].param_role is ParamRole.THRESHOLD
    assert "logical-contract divergence" in capsys.readouterr().err
