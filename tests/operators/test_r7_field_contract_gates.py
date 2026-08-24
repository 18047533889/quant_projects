"""Round-7 WS-E field/source contract gates (findings #279-#293, #315-#317).

Covers:
    #279 mining_allowed production hard gate
    #280 current_snapshot_only cannot backfill a historical window
    #282 four-layer PIT eligibility (field ∧ table ∧ dataset ∧ operator)
    #283 FieldRegistry.replace cleans old aliases
    #284 role must be a validated Enum vocabulary (catalog load fails on typo)
    #289 SourceRef scalar params reject NaN / Inf
    #290 minute→daily exact alignment (no asof carry on missing days)
    #291 Intermediate/DerivedField default join_policy=exact
    #292 all-missing minute volume/amount → NaN (not 0)
    #293 timestamp convention read from the DatasetContract, not a 09:30 heuristic
    #315 HistoricalCoverageContract coverage gating
    #316 MissingSemantic enum + field→semantic mapping
    #317 source_dependency_hash in the source materialization identity
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _anchor(days=("2024-01-02", "2024-01-03", "2024-01-04"), instruments=("A",)):
    return pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), inst) for d in days for inst in instruments],
        names=["timestamp", "instrument"],
    )


# ---------------------------------------------------------------------------
# #279 mining_allowed production hard gate
# ---------------------------------------------------------------------------
def test_mining_gate_rejects_mining_allowed_false_field():
    from factor_engine.api.columns import field

    # index_weight is mining_allowed=False (one_to_many, must be aggregated).
    assert field("index_weight").table == "IndexConstituent"
    with pytest.raises(ValueError, match="mining_allowed=False"):
        field("index_weight", for_mining=True)


def test_mining_gate_source_level_raw_read_rejection():
    from factor_engine.storage.sources.data_access_source import (
        DataAccessSource,
        FieldMiningGateError,
    )
    from factor_engine.storage.sources.field_plan import NormalizedFieldPlan

    src = DataAccessSource(dataset="ashare_stock_daily", enforce_mining_gate=True)
    plan = NormalizedFieldPlan(logical_concept="idx", mining_allowed=False)
    with pytest.raises(FieldMiningGateError, match="mining_allowed=False"):
        src._enforce_field_contract_gates({"idx": plan})
    # Gate disabled (default) keeps the raw read working.
    src2 = DataAccessSource(dataset="ashare_stock_daily")
    src2._enforce_field_contract_gates({"idx": plan})  # no raise


# ---------------------------------------------------------------------------
# #280 current_snapshot_only cannot backfill a historical window
# ---------------------------------------------------------------------------
def test_current_snapshot_only_historical_backfill_rejected():
    from factor_engine.storage.sources.data_access_source import (
        DataAccessSource,
        HistoricalSnapshotBackfillError,
    )
    from factor_engine.storage.sources.field_plan import NormalizedFieldPlan

    src = DataAccessSource(
        dataset="ashare_stock_daily",
        start_date="2020-01-01",
        enforce_mining_gate=True,
    )
    plan = NormalizedFieldPlan(
        logical_concept="snap", current_snapshot_only=True, coverage="current_snapshot"
    )
    with pytest.raises(HistoricalSnapshotBackfillError, match="current_snapshot_only"):
        src._enforce_field_contract_gates({"snap": plan})


def test_current_snapshot_only_allowed_with_snapshot_now_only():
    from factor_engine.storage.sources.data_access_source import DataAccessSource
    from factor_engine.storage.sources.field_plan import NormalizedFieldPlan

    src = DataAccessSource(
        dataset="ashare_stock_daily",
        start_date="2020-01-01",
        enforce_mining_gate=True,
        snapshot_now_only=True,
    )
    plan = NormalizedFieldPlan(
        logical_concept="snap", current_snapshot_only=True, coverage="current_snapshot"
    )
    src._enforce_field_contract_gates({"snap": plan})  # no raise


def test_current_snapshot_only_now_window_allowed():
    from factor_engine.storage.sources.data_access_source import DataAccessSource
    from factor_engine.storage.sources.field_plan import NormalizedFieldPlan

    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    src = DataAccessSource(
        dataset="ashare_stock_daily", start_date=today, enforce_mining_gate=True
    )
    plan = NormalizedFieldPlan(
        logical_concept="snap", current_snapshot_only=True, coverage="current_snapshot"
    )
    src._enforce_field_contract_gates({"snap": plan})  # not historical → no raise


# ---------------------------------------------------------------------------
# #282 four-layer PIT eligibility
# ---------------------------------------------------------------------------
def test_four_layer_pit_combined_eligibility():
    from pit_contract import PITLayerVerdict, four_layer_pit_allowed

    ok, layers = four_layer_pit_allowed(
        field_pit_allowed=True,
        table_pit_allowed=True,
        dataset_pit_allowed=True,
        operator_pit_allowed=True,
    )
    assert ok is True
    assert all(v == PITLayerVerdict.PROVEN_TRUE for v in layers.values())

    ok, layers = four_layer_pit_allowed(field_pit_allowed=False)
    assert ok is False and layers["field_pit_allowed"] == PITLayerVerdict.PROVEN_FALSE

    ok, layers = four_layer_pit_allowed(
        field_pit_allowed=True, operator_pit_allowed=False
    )
    assert ok is False and layers["operator_pit_allowed"] == PITLayerVerdict.PROVEN_FALSE

    # R24-049..051: an OMITTED layer is UNKNOWN — never treated as True.
    ok, layers = four_layer_pit_allowed(field_pit_allowed=True)
    assert ok is False
    assert layers["table_pit_allowed"] == PITLayerVerdict.UNKNOWN


def test_four_layer_pit_source_preflight_rejects_field_layer():
    from factor_engine.storage.sources.data_access_source import (
        DataAccessSource,
        FourLayerPITError,
    )
    from factor_engine.storage.sources.field_plan import NormalizedFieldPlan

    src = DataAccessSource(dataset="ashare_stock_daily")
    plan = NormalizedFieldPlan(logical_concept="div", strict_pit_allowed=False)
    with pytest.raises(FourLayerPITError, match="field_pit_allowed"):
        src.assert_four_layer_pit({"div": plan})


# ---------------------------------------------------------------------------
# #283 FieldRegistry.replace must clean old aliases
# ---------------------------------------------------------------------------
def test_replace_cleans_old_field_aliases():
    from factor_engine.fields import FieldRegistry, FieldSpec, TableSpec

    reg = FieldRegistry(tables=[TableSpec("T", "d")])
    reg.register(FieldSpec("f", "T", "OldName", aliases=("old_alias",)))
    assert reg.get("old_alias") is not None
    assert reg.get("OldName", table="T") is not None

    reg.register(FieldSpec("f", "T", "NewName", aliases=("new_alias",)), replace=True)

    # Old aliases owned by the replaced identity are gone.
    assert reg.get("old_alias") is None
    assert reg.get("OldName", table="T") is None
    # New aliases resolve.
    assert reg.get("new_alias") is not None
    assert reg.get("NewName", table="T") is not None


def test_replace_table_cleans_old_aliases():
    from factor_engine.fields import FieldRegistry, TableSpec

    reg = FieldRegistry()
    reg.register_table(TableSpec("T", "old_dataset", aliases=("OldT",)))
    assert reg.resolve_table("OldT") is not None
    reg.register_table(TableSpec("T", "new_dataset", aliases=("NewT",)), replace=True)
    assert reg.resolve_table("OldT") is None
    assert reg.resolve_table("NewT") is not None


# ---------------------------------------------------------------------------
# #284 role vocabulary validation (catalog load fails on unknown values)
# ---------------------------------------------------------------------------
def test_role_typo_fails_catalog_validation():
    from factor_engine.fields.catalog import FieldRole, validate_field_role

    assert FieldRole.FEATURE.value == "feature"
    validate_field_role("feature")  # known role is accepted
    validate_field_role("knowledge_time")
    with pytest.raises(ValueError, match="unknown field role"):
        validate_field_role("knowledge_tiem")


def test_role_typo_fails_field_builder():
    from factor_engine.fields.catalog import _f  # private builder used by ASHARE_FIELD_SPECS

    # _f validates role before constructing a FieldSpec; a typo must raise.
    with pytest.raises(ValueError, match="unknown field role"):
        _f("bad", "StockDailyBar", "Open", role="knowledge_tiem")


# ---------------------------------------------------------------------------
# #289 SourceRef scalar params reject NaN / Inf
# ---------------------------------------------------------------------------
def test_source_ref_rejects_non_finite_scalars():
    from factor_engine.api.source_ref import make_source_ref

    spec = make_source_ref("StockDailyBar", "Close", params={"lookback": 5})
    assert spec.params_dict()["lookback"] == 5
    make_source_ref("t", "f", params={"x": 1.5})  # finite float OK
    with pytest.raises(ValueError, match="NaN/Inf"):
        make_source_ref("t", "f", params={"x": float("nan")})
    with pytest.raises(ValueError, match="NaN/Inf"):
        make_source_ref("t", "f", params={"x": float("inf")})
    with pytest.raises(ValueError, match="NaN/Inf"):
        make_source_ref("t", "f", params={"x": float("-inf")})


# ---------------------------------------------------------------------------
# #290 minute→daily exact alignment — no asof carry on missing days
# ---------------------------------------------------------------------------
def test_minute_daily_exact_alignment_no_asof_carry():
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

    anchor = _anchor()
    # minute-aggregated daily is missing 2024-01-03 for instrument A.
    daily = pd.Series(
        [1.0, 3.0],
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "A"),
                (pd.Timestamp("2024-01-04"), "A"),
            ],
            names=["timestamp", "instrument"],
        ),
        name="v",
    )
    exact = LQTPLogicalDataSource._align_exact_by_instrument(anchor, daily)
    missing = exact.loc[(pd.Timestamp("2024-01-03"), "A")]
    assert pd.isna(missing), "missing minute day must be NaN, not a carry"
    assert exact.loc[(pd.Timestamp("2024-01-02"), "A")] == 1.0
    assert exact.loc[(pd.Timestamp("2024-01-04"), "A")] == 3.0

    # For contrast: the legacy asof alignment carries 1.0 forward — the exact
    # gate exists precisely to prevent that.
    asof = LQTPLogicalDataSource._align_by_instrument(anchor, daily)
    assert asof.loc[(pd.Timestamp("2024-01-03"), "A")] == 1.0


# ---------------------------------------------------------------------------
# #291 Intermediate/DerivedField default join_policy=exact
# ---------------------------------------------------------------------------
def test_intermediate_derived_default_join_policy_exact():
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

    anchor = _anchor()
    series = pd.Series(
        [1.0, 3.0],
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "A"),
                (pd.Timestamp("2024-01-04"), "A"),
            ],
            names=["timestamp", "instrument"],
        ),
        name="v",
    )
    exact = LQTPLogicalDataSource._align_for_join_policy(
        anchor, series, "exact", context="intermediate"
    )
    assert pd.isna(exact.loc[(pd.Timestamp("2024-01-03"), "A")])
    # Explicit state_asof retains backward carry semantics for callers that opt in.
    asof = LQTPLogicalDataSource._align_for_join_policy(
        anchor, series, "state_asof", context="intermediate"
    )
    assert asof.loc[(pd.Timestamp("2024-01-03"), "A")] == 1.0
    with pytest.raises(Exception):
        LQTPLogicalDataSource._align_for_join_policy(
            anchor, series, "bogus", context="intermediate"
        )


# ---------------------------------------------------------------------------
# #292 all-missing minute volume/amount → NaN, not 0
# ---------------------------------------------------------------------------
def test_minute_all_missing_volume_is_nan_not_zero():
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

    grp = pd.DataFrame({"value": [np.nan, np.nan]})
    assert np.isnan(LQTPLogicalDataSource._minute_semantic_aggregate_frame(grp, "volume"))
    assert np.isnan(LQTPLogicalDataSource._minute_semantic_aggregate_frame(grp, "amount"))
    mixed = pd.DataFrame({"value": [np.nan, 2.0]})
    assert LQTPLogicalDataSource._minute_semantic_aggregate_frame(mixed, "volume") == 2.0


# ---------------------------------------------------------------------------
# #293 timestamp convention from the DatasetContract, not a 09:30 heuristic
# ---------------------------------------------------------------------------
def test_session_timestamp_convention_reads_contract():
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

    assert LQTPLogicalDataSource._session_timestamp_convention() == "bar_end"


def test_session_slots_use_declared_convention():
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime([
            "2024-01-02 09:31:00", "2024-01-02 11:30:00",
            "2024-01-02 13:01:00", "2024-01-02 15:00:00",
        ])}
    )
    out = LQTPLogicalDataSource._session_slots(frame)
    # bar_end: 09:31 → am slot 0, 11:30 → am slot 119, 13:01 → pm slot 0.
    assert list(out["session"]) == ["am", "am", "pm", "pm"]
    assert list(out["session_slot"]) == [0, 119, 0, 119]


# ---------------------------------------------------------------------------
# #315 HistoricalCoverageContract
# ---------------------------------------------------------------------------
def test_historical_coverage_contract_violations():
    from factor_engine.storage.sources.data_access_source import (
        HistoricalCoverageContract,
        assert_historical_coverage,
        HistoricalCoverageError,
    )

    good = HistoricalCoverageContract(
        field="x", coverage_ratio=0.9, coverage_by_year={2020: 0.95, 2021: 0.9},
    )
    assert_historical_coverage(good, start="2020-01-01", end="2021-12-31", threshold=0.7)

    partial = HistoricalCoverageContract(
        field="x", coverage_ratio=0.4, first_valid_date="2021-06-01",
        coverage_by_year={2020: 0.2, 2021: 0.8},
    )
    with pytest.raises(HistoricalCoverageError, match="overall coverage"):
        assert_historical_coverage(partial, start="2020-01-01", end="2021-12-31", threshold=0.7)
    with pytest.raises(HistoricalCoverageError, match="first_valid_date"):
        assert_historical_coverage(partial, start="2020-01-01", end="2020-06-01", threshold=0.1)


def test_coverage_gate_source_level():
    from factor_engine.storage.sources.data_access_source import (
        DataAccessSource,
        HistoricalCoverageContract,
        HistoricalCoverageError,
        register_coverage_contract,
    )
    from factor_engine.storage.sources.field_plan import NormalizedFieldPlan

    field_name = "test_cov_field_315"
    register_coverage_contract(
        HistoricalCoverageContract(field=field_name, coverage_ratio=0.4)
    )
    src = DataAccessSource(
        dataset="ashare_stock_daily",
        start_date="2020-01-01",
        mining_coverage_threshold=0.7,
        strict_unknown_fields=True,
    )
    plan = NormalizedFieldPlan(logical_concept=field_name, coverage="partial_history")
    with pytest.raises(HistoricalCoverageError):
        src._gate_coverage(plan, field_name)


# ---------------------------------------------------------------------------
# #316 MissingSemantic enum + field→semantic mapping
# ---------------------------------------------------------------------------
def test_missing_semantic_mapping():
    from factor_engine.storage.sources.field_plan import (
        MissingSemantic,
        NormalizedFieldPlan,
        missing_semantic_for_plan,
    )

    sparse = NormalizedFieldPlan(logical_concept="ann", coverage="sparse_event")
    assert missing_semantic_for_plan(sparse) is MissingSemantic.NO_EVENT

    # R17-004: a partial-history FINANCIAL field's missing value is UNKNOWN
    # (expected but not yet observed), never reinterpreted as "no event".
    financial = NormalizedFieldPlan(logical_concept="rev", coverage="partial_history")
    assert missing_semantic_for_plan(financial) is MissingSemantic.UNKNOWN

    # ``coverage=partial_history`` IS the financial-pit plan: a missing numeric
    # financial value is UNKNOWN, never NO_EVENT (R17-004).
    financial_pit = NormalizedFieldPlan(
        logical_concept="roe", coverage="partial_history", semantic_kind="roe"
    )
    assert missing_semantic_for_plan(financial_pit) is MissingSemantic.UNKNOWN

    # R17-004 + R24-065..067: a current-snapshot table not covering a
    # historical date is OUT_OF_COVERAGE (the source has no history) — NOT
    # "no event" and NOT "economically not applicable".
    current_only = NormalizedFieldPlan(logical_concept="mc", coverage="current_snapshot")
    assert missing_semantic_for_plan(current_only) is MissingSemantic.OUT_OF_COVERAGE

    dense = NormalizedFieldPlan(logical_concept="close", coverage="historical_panel")
    assert missing_semantic_for_plan(dense) is MissingSemantic.UNKNOWN

    zero = NormalizedFieldPlan(logical_concept="z", null_policy="zero_fill")
    assert missing_semantic_for_plan(zero) is MissingSemantic.STRUCTURAL_ZERO

    role = NormalizedFieldPlan(logical_concept="d", role="knowledge_time")
    assert missing_semantic_for_plan(role) is MissingSemantic.NOT_APPLICABLE

    assert {item.name for item in MissingSemantic} == {
        "UNKNOWN", "NO_EVENT", "NOT_APPLICABLE", "NOT_TRADING", "STRUCTURAL_ZERO",
        # R24-065: OUT_OF_COVERAGE = the source has no history for this
        # time/instrument.
        "OUT_OF_COVERAGE",
    }


# ---------------------------------------------------------------------------
# #317 source_dependency_hash in the source materialization identity
# ---------------------------------------------------------------------------
def test_source_dependency_hash_stable_and_identity_bound():
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    src = DataAccessSource(dataset="ashare_stock_daily")
    src._ensure_field_plans(["ret", "close"])
    h1 = src.source_dependency_hash()
    assert isinstance(h1, str) and len(h1) == 64
    # Deterministic for the same plan + snapshot state.
    assert h1 == src.source_dependency_hash()

    # A different plan set (different source identity) changes the hash.
    src2 = DataAccessSource(dataset="ashare_stock_daily")
    src2._ensure_field_plans(["close"])
    assert src2.source_dependency_hash() != h1
