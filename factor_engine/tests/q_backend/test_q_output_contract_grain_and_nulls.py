"""Regression tests for Q output-contract grain derivation and rolling-warmup null allowance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from factor_engine.backend.q_backend.q_errors import QOutputContract, QOutputContractViolation
from factor_engine.backend.q_backend.q_backend import QBackend


@dataclass
class _StubDataSource:
    pass


class _MinimalExecutionContext:
    def __init__(
        self,
        semantic_attrs: Mapping[str, Any] | None = None,
        grain: str | None = None,
    ) -> None:
        self.data_source = _StubDataSource()
        self.semantic_attrs = semantic_attrs or {}
        self.grain = grain


def _clean_output_df(*, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=len(values), freq="D"),
            "instrument": ["AAA"] * len(values),
            "value": values,
        }
    )


def _make_valid_series_backend() -> QBackend:
    return QBackend(fallback_to_pandas=False, production_mode=False)


def test_q_output_contract_defaults_allows_nulls_and_variable_grain() -> None:
    contract = QOutputContract(
        expected_columns=("timestamp", "instrument", "value"),
        expected_dtypes={"timestamp": "datetime64[ns]", "value": "float64"},
    )
    assert contract.allow_nulls is True
    assert contract.grain is None


def test_q_output_contract_derives_grain_from_semantic_attrs() -> None:
    backend = _make_valid_series_backend()
    ctx = _MinimalExecutionContext(semantic_attrs={"grain": "minute"})

    contract = backend._build_output_contract(ctx)

    assert contract.grain == "minute"
    assert contract.allow_nulls is True


def test_q_output_contract_falls_back_to_context_grain_attribute() -> None:
    backend = _make_valid_series_backend()
    ctx = _MinimalExecutionContext(semantic_attrs={}, grain="event")

    contract = backend._build_output_contract(ctx)

    assert contract.grain == "event"


def test_q_output_contract_does_not_reject_rolling_warmup_nulls() -> None:
    backend = _make_valid_series_backend()
    df = _clean_output_df(values=[np.nan, np.nan, 1.0, 2.0])
    ctx = _MinimalExecutionContext(semantic_attrs={"grain": "daily"})

    contract = backend._build_output_contract(ctx)
    backend._validate_output_contract(df, contract)


def test_q_output_contract_rejects_missing_required_column() -> None:
    backend = _make_valid_series_backend()
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="D"),
            "instrument": ["AAA", "AAA"],
        }
    )
    ctx = _MinimalExecutionContext()

    contract = backend._build_output_contract(ctx)

    try:
        backend._validate_output_contract(df, contract)
        raise AssertionError("Expected QOutputContractViolation")
    except QOutputContractViolation as exc:
        assert "Missing required columns" in str(exc)
