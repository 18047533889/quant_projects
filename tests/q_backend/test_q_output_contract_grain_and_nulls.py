"""Regression tests for Q output-contract grain derivation and rolling-warmup null allowance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.q_backend.q_errors import (
    QOutputContract,
    QOutputContractViolation,
    semantic_kind_to_output_dtype,
    semantic_null_policy,
)
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


# ---------------------------------------------------------------------------
# R21-Q-OUTPUT-DTYPE: value dtype + null policy derived from canonical semantic
# kind (Event→bool, State/Group→int64, timestamp-role→datetime, else float64).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("semantic_kind", "expected_dtype", "expected_policy"),
    [
        # Sparse condition/event/mask panels: boolean value, structural nulls
        # allowed (NaN == absent event).
        ("EventBool", "bool", "any_allowed"),
        ("MaskBool", "bool", "any_allowed"),
        ("ConditionBool", "bool", "any_allowed"),
        # State / categorical / group / period codes: integer backing, only
        # rolling-warmup nulls allowed.
        ("StateSigned", "int64", "warmup_ok"),
        ("GroupKey", "int64", "warmup_ok"),
        ("FiscalPeriodId", "int64", "warmup_ok"),
        ("StatusCode", "int64", "warmup_ok"),
        ("CategoryCode", "int64", "warmup_ok"),
        # Timestamp-role outputs: datetime, no nulls permitted.
        ("KnowledgeTimestamp", "datetime64[ns]", "strict"),
        ("EffectiveTimestamp", "datetime64[ns]", "strict"),
        ("RevisionTimestamp", "datetime64[ns]", "strict"),
        # Numeric-derived / generic: float64, rolling-warmup nulls only.
        (None, "float64", "warmup_ok"),
        ("PriceContinuous", "float64", "warmup_ok"),
        ("ReturnDecimal", "float64", "warmup_ok"),
    ],
)
def test_semantic_kind_derives_dtype_and_null_policy(
    semantic_kind: str | None,
    expected_dtype: str,
    expected_policy: str,
) -> None:
    assert semantic_kind_to_output_dtype(semantic_kind) == expected_dtype
    warmup, structural, policy = semantic_null_policy(semantic_kind)
    assert policy == expected_policy
    if expected_policy == "any_allowed":
        assert warmup is True and structural is True
    elif expected_policy == "strict":
        assert warmup is False and structural is False
    else:
        assert warmup is True and structural is False


def test_build_output_contract_derives_value_dtype_and_null_policy() -> None:
    backend = _make_valid_series_backend()

    event_contract = backend._build_output_contract(
        _MinimalExecutionContext(semantic_attrs={"semantic_kind": "EventBool"})
    )
    assert event_contract.expected_dtypes["value"] == "bool"
    assert event_contract.warmup_nulls_allowed is True
    assert event_contract.structural_nulls_allowed is True
    assert event_contract.output_null_policy == "any_allowed"
    assert event_contract.allow_nulls is True

    state_contract = backend._build_output_contract(
        _MinimalExecutionContext(semantic_attrs={"semantic_kind": "StateSigned"})
    )
    assert state_contract.expected_dtypes["value"] == "int64"
    assert state_contract.warmup_nulls_allowed is True
    assert state_contract.structural_nulls_allowed is False
    assert state_contract.output_null_policy == "warmup_ok"
    assert state_contract.allow_nulls is True

    ts_contract = backend._build_output_contract(
        _MinimalExecutionContext(
            semantic_attrs={"semantic_kind": "EffectiveTimestamp"}
        )
    )
    assert ts_contract.expected_dtypes["value"] == "datetime64[ns]"
    assert ts_contract.warmup_nulls_allowed is False
    assert ts_contract.structural_nulls_allowed is False
    assert ts_contract.output_null_policy == "strict"
    assert ts_contract.allow_nulls is False


def test_build_output_contract_defaults_to_float64_when_no_semantic_kind() -> None:
    backend = _make_valid_series_backend()
    contract = backend._build_output_contract(_MinimalExecutionContext(semantic_attrs={}))
    assert contract.expected_dtypes["value"] == "float64"
    assert contract.output_null_policy == "warmup_ok"
    assert contract.warmup_nulls_allowed is True
    assert contract.structural_nulls_allowed is False
    assert contract.allow_nulls is True
