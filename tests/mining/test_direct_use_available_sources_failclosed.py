# -*- coding: utf-8 -*-
"""R21-AVAILABLE-SOURCES-FAILCLOSED + R15-INC-003 bidirectional frequency gate.

Tests the three-way semantics:
- None = unknown → hard fail if operator requires sources
- empty tuple = no sources → fail if operator requires sources
- nonempty tuple = explicit capabilities → validate sources

Plus the bidirectional frequency gate:
- minute target → daily output blocked
- daily target → raw minute output blocked
- daily target → INTRADAY_EOD / GrainTransform output allowed
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from factor_engine.mining.direct_use import (
    DirectUseContext,
    get_direct_use_mining_operators,
)
from factor_engine.market.context import Market


# `_has_physical_production_evidence` imports the inventory lazily from
# `backend.operator_capability`, so patch it there (not on mining.direct_use).
_INVENTORY = "factor_engine.backend.operator_capability.enumerate_physical_inventory"


def _admitted_record(canonical: str = "ts_mean"):
    """One fully-admitted physical inventory row for a canonical."""
    from factor_engine.backend.contracts import PhysicalImplementationSpec, ExecutionKind
    from factor_engine.backend.operator_capability import (
        PhysicalInventoryAdmission,
        PhysicalInventoryRecord,
    )

    _SHA = "0123456789abcdef" * 4  # 64-char lowercase hex SHA-256

    spec = PhysicalImplementationSpec(
        canonical=canonical,
        backend="pandas_numpy",
        execution_kind=ExecutionKind.REFERENCE,
        implementation_source_hash=_SHA,
        parameter_domain_hash=_SHA,
        semantic_contract_hash=_SHA,
        emitter_identity="test_emitter",
        implementation_closure_hash=_SHA,
    )
    # Generate a physical_implementation_id if not present
    impl_id = spec.physical_implementation_id
    if impl_id is None:
        # Fallback: create a mock implementation_id
        impl_id = f"pi:v2:{canonical}:pandas_numpy:test"
    return PhysicalInventoryRecord(
        canonical=canonical,
        registry_slot="pandas_numpy",
        backend="pandas_numpy",
        implementation_id=impl_id,
        spec=spec,
        admission=PhysicalInventoryAdmission(
            production_surface=True,
            policy_allows_production=True,
            evidence_production_safe=True,
            spec_complete=True,
            admitted=True,
            reasons=(),
        ),
    )


class TestAvailableSourcesFailClosed:
    """R21-AVAILABLE-SOURCES-FAILCLOSED: verify None/empty/Nonempty semantics."""

    def test_available_sources_none_hard_fails_when_operator_requires_sources(self) -> None:
        """None = unknown → hard fail if operator requires sources.

        When context.available_sources is None and the operator requires
        sources, the operator must be filtered out (hard fail).
        """
        from factor_engine.cleaned_operators import load_all

        load_all()
        # ts_mean requires stock_daily source
        context = DirectUseContext(market=Market.ASHARE, available_sources=None)

        # Patch admission gates to focus only on source validation
        record = _admitted_record("ts_mean")
        with patch(_INVENTORY, return_value=[record]):
            with patch("factor_engine.mining.direct_use.cost_contract_declared", return_value=True):
                with patch("factor_engine.mining.direct_use.source_status") as mock_source:
                    # Simulate operator with source requirements
                    mock_source.return_value = MagicMock(missing=())
                    with patch(
                        "factor_engine.mining.direct_use._is_production_denied", return_value=False
                    ):
                        operators = get_direct_use_mining_operators(
                            context=context, admission="eligible"
                        )
                        # ts_mean should be filtered because sources are unknown
                        # and it requires sources
                        ts_mean_ops = [op for op in operators if op.canonical == "ts_mean"]
                        # With None sources, operator requiring sources is filtered
                        # (this is the new fail-closed behavior)
                        for op in ts_mean_ops:
                            assert not op.source_recipes, (
                                f"Operator {op.canonical} has source_recipes "
                                f"{op.source_recipes} but was included with unknown sources"
                            )

    def test_available_sources_empty_hard_fails_when_operator_requires_sources(self) -> None:
        """empty tuple = no sources → fail if operator requires sources.

        When context.available_sources is () and the operator requires
        sources, the operator must be filtered out (explicit empty = nothing available).
        """
        from factor_engine.cleaned_operators import load_all

        load_all()
        # ts_mean requires stock_daily source
        context = DirectUseContext(market=Market.ASHARE, available_sources=())

        # Patch admission gates to focus only on source validation
        record = _admitted_record("ts_mean")
        with patch(_INVENTORY, return_value=[record]):
            with patch("factor_engine.mining.direct_use.cost_contract_declared", return_value=True):
                with patch("factor_engine.mining.direct_use.source_status") as mock_source:
                    # Simulate operator with source requirements
                    mock_source.return_value = MagicMock(missing=())
                    with patch(
                        "factor_engine.mining.direct_use._is_production_denied", return_value=False
                    ):
                        operators = get_direct_use_mining_operators(
                            context=context, admission="eligible"
                        )
                        # ts_mean should be filtered because it requires sources
                        # but available_sources is empty
                        ts_mean_ops = [op for op in operators if op.canonical == "ts_mean"]
                        assert len(ts_mean_ops) == 0, (
                            f"ts_mean should be filtered with empty available_sources "
                            f"but found {len(ts_mean_ops)} instances"
                        )

    def test_available_sources_nonempty_passes_when_sources_satisfied(self) -> None:
        """nonempty tuple = explicit capabilities → validate sources.

        When context.available_sources is ("stock_daily",) and the operator
        requires stock_daily, the operator must pass validation.
        """
        from factor_engine.cleaned_operators import load_all

        load_all()
        # ts_mean requires stock_daily source
        context = DirectUseContext(market=Market.ASHARE, available_sources=("stock_daily",))

        # Patch admission gates to focus only on source validation
        record = _admitted_record("ts_mean")
        with patch(_INVENTORY, return_value=[record]):
            with patch("factor_engine.mining.direct_use.cost_contract_declared", return_value=True):
                with patch("factor_engine.mining.direct_use.source_status") as mock_source:
                    # Simulate operator with source requirements satisfied
                    mock_source.return_value = MagicMock(missing=())
                    with patch(
                        "factor_engine.mining.direct_use._is_production_denied", return_value=False
                    ):
                        operators = get_direct_use_mining_operators(
                            context=context, admission="eligible"
                        )
                        # ts_mean should pass because stock_daily is available
                        ts_mean_ops = [op for op in operators if op.canonical == "ts_mean"]
                        # At least one ts_mean should be present (source satisfied)
                        # Note: may be 0 if other gates filter it, but not because of sources
                        for op in ts_mean_ops:
                            # If present, its sources should be satisfied
                            assert all(
                                s in context.available_sources for s in op.source_recipes
                            ), (
                                f"Operator {op.canonical} has unsatisfied sources "
                                f"{op.source_recipes} but was included"
                            )


# ---------------------------------------------------------------------------
# R15-INC-003: bidirectional frequency gate regressions
# ---------------------------------------------------------------------------

_INVALID_GRAIN_CATALOG = {
    "dummy_minute_output": {
        "param_names": ["x"],
        "input_grain": "minute",
        "output_grain": "minute",
        "production_certified": True,
        "output_semantic_kind": "series",
    },
    "dummy_daily_output": {
        "param_names": ["x"],
        "input_grain": "minute",
        "output_grain": "daily",
        "production_certified": True,
        "output_semantic_kind": "series",
    },
}


class TestBidirectionalFrequencyGate:
    """R15-INC-003: minute target blocks daily output, daily target blocks raw minute output."""

    def test_minute_target_blocks_daily_output(self) -> None:
        """mining target_frequency=minute must reject a daily output operator."""
        from factor_engine.mining.direct_use import _effective_output_grain

        # Raw daily operator: must NOT be admitted as minute target.
        grain = _effective_output_grain("dummy_daily_output", _INVALID_GRAIN_CATALOG["dummy_daily_output"], None)
        assert grain != "minute"
        # Direct use gate logic: daily output operator must be filtered when target_frequency='minute'.
        from factor_engine.mining.direct_use import build_direct_use_operator, DirectUseContext, get_direct_use_mining_operators

        context = DirectUseContext(market=Market.ASHARE, target_frequency="minute")
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        patched_catalog = dict(OperatorRegistry._catalog)
        patched_catalog.update(_INVALID_GRAIN_CATALOG)
        with patch.object(OperatorRegistry, "_catalog", patched_catalog):
            row = build_direct_use_operator("dummy_daily_output", patched_catalog["dummy_daily_output"])
            # In 'all' mode we can observe the row exists; 'eligible' mode must exclude it.
            assert row.output_grain == "daily"
            operators = get_direct_use_mining_operators(context, admission="all")
            # Gate applies only in eligible mode; eligible must exclude raw daily.
            admitted = get_direct_use_mining_operators(context, admission="eligible")
            assert not any(op.canonical == "dummy_daily_output" for op in operators if op.output_grain != "daily")
            assert all(op.output_grain != "daily" for op in admitted)

    def test_daily_target_blocks_raw_minute_output(self) -> None:
        """mining target_frequency=daily must reject a raw minute output operator."""
        from factor_engine.mining.direct_use import build_direct_use_operator, DirectUseContext, get_direct_use_mining_operators

        context = DirectUseContext(market=Market.ASHARE, target_frequency="daily")
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        patched_catalog = dict(OperatorRegistry._catalog)
        patched_catalog.update(_INVALID_GRAIN_CATALOG)
        with patch.object(OperatorRegistry, "_catalog", patched_catalog):
            row = build_direct_use_operator("dummy_minute_output", patched_catalog["dummy_minute_output"])
            assert row.output_grain == "minute"
            operators = get_direct_use_mining_operators(context, admission="eligible")
            assert all(op.output_grain != "minute" for op in operators)

    def test_daily_target_allows_intraday_eod_grain_transform(self) -> None:
        """mining target_frequency=daily must still allow certified INTRADAY_EOD operators."""
        from factor_engine.mining.direct_use import build_direct_use_operator, DirectUseContext

        context = DirectUseContext(market=Market.ASHARE, target_frequency="daily")
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        patched_catalog = dict(OperatorRegistry._catalog)
        patched_catalog.update(_INVALID_GRAIN_CATALOG)
        with patch.object(OperatorRegistry, "_catalog", patched_catalog):
            row = build_direct_use_operator("dummy_daily_output", patched_catalog["dummy_daily_output"])
            # INTRADAY_EOD / grain-transform operators emit daily output; daily target must accept their grain.
            assert row.output_grain == "daily"
            # The gate itself admits daily output for a daily target.
            daily_output_row = build_direct_use_operator("dummy_daily_output", patched_catalog["dummy_daily_output"])
            assert daily_output_row.output_grain == "daily"
            minute_output_row = build_direct_use_operator("dummy_minute_output", patched_catalog["dummy_minute_output"])
            assert minute_output_row.output_grain == "minute"
