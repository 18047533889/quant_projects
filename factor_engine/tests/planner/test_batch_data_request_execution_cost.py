from types import SimpleNamespace as NS

import pytest

from factor_engine.planner.batch_data_request import (
    BatchDataRequest,
    CompositeExecutionScanCostUnavailable,
    ScanCostUnavailable,
    SourceScanGroup,
)
from factor_engine.planner.physical_factor_dag import SourceScopeId


def _cost(*, rows, selected, projection, files=1, remote=False, basis="physical_scope"):
    return NS(
        estimated_rows=rows,
        selected_bytes=selected,
        projection_bytes=projection,
        file_count=files,
        selected_files=files,
        total_bytes=selected,
        projected_columns=1,
        remote=remote,
        cost_basis=basis,
    )


def _group(dataset, cost, *, params="", unavailable=None):
    return SourceScanGroup(
        group_id=0,
        dataset=dataset,
        source_scope=SourceScopeId(dataset=dataset, market="ashare", params_digest=params),
        snapshot_id="snap",
        fields=("value",),
        scan_cost=cost,
        scan_cost_unavailable=unavailable,
    )


def test_proven_filtered_empty_scope_keeps_snapshot_files_but_no_row_payload():
    anchor = _group("daily", _cost(rows=40, selected=400, projection=320))
    empty = _cost(rows=0, selected=0, projection=0, files=20)
    empty.total_bytes = 100000
    empty.selected_files = 0
    empty.selected_rowgroups = 0
    empty.empty_result_proven = True
    empty.rowgroup_pruning_basis = "parquet_footer_min_max_closed_predicates"
    event = _group("events", empty)
    got = BatchDataRequest(anchor_source_scope=anchor.source_scope, groups=[anchor, event]).execution_scan_cost_map()[anchor.source_scope_key]
    assert got.file_count == 21
    assert got.selected_files == 1
    assert got.total_bytes == 100400
    assert got.projection_bytes == 320
    assert got.estimated_rows == 40


@pytest.mark.parametrize("proof", [None, False, -1, 1, 0])
def test_nonempty_physical_scope_without_valid_empty_proof_is_rejected(proof):
    anchor = _group("daily", _cost(rows=40, selected=400, projection=320))
    empty = _cost(rows=0, selected=0, projection=0, files=20)
    empty.selected_files = 0
    empty.selected_rowgroups = proof
    event = _group("events", empty)
    with pytest.raises(CompositeExecutionScanCostUnavailable):
        BatchDataRequest(anchor_source_scope=anchor.source_scope, groups=[anchor, event]).execution_scan_cost_map()


def test_two_independent_sources_sum_raw_execution_cost_under_exact_anchor_key():
    anchor = _group("daily", _cost(rows=100, selected=1_000, projection=800))
    event = _group(
        "events", _cost(rows=25, selected=400, projection=240, files=2, remote=True)
    )
    request = BatchDataRequest(anchor_source_scope=anchor.source_scope, groups=[anchor, event])

    got = request.execution_scan_cost_map()

    assert set(got) == {anchor.source_scope_key}
    composite = got[anchor.source_scope_key]
    assert composite.estimated_rows == 125  # raw scan rows, not aligned output rows
    assert composite.selected_bytes == 1_400
    assert composite.projection_bytes == 1_040
    assert composite.file_count == composite.selected_files == 3
    assert composite.remote is True
    assert composite.component_scope_keys == (
        anchor.source_scope_key, event.source_scope_key
    )
    # Per-scope planner facts remain unchanged.
    assert request.scan_cost_map == {
        anchor.source_scope_key: anchor.scan_cost,
        event.source_scope_key: event.scan_cost,
    }


def test_rank_scopes_are_each_charged_even_when_dataset_and_field_match():
    anchor = _group("daily", _cost(rows=40, selected=400, projection=320))
    rank1 = _group(
        "topten", _cost(rows=1_000, selected=8_000, projection=6_000),
        params="p:ShareholderRank=1",
    )
    rank2 = _group(
        "topten", _cost(rows=900, selected=7_000, projection=5_000),
        params="p:ShareholderRank=2",
    )
    request = BatchDataRequest(
        anchor_source_scope=anchor.source_scope, groups=[anchor, rank1, rank2]
    )

    composite = request.execution_scan_cost_map()[
        anchor.source_scope_key
    ]

    assert composite.estimated_rows == 1_940
    assert composite.projection_bytes == 11_320
    assert composite.component_scope_keys[-2:] == (
        rank1.source_scope_key, rank2.source_scope_key
    )


@pytest.mark.parametrize("missing_kind", ["none", "unavailable", "unknown_basis"])
def test_any_unknown_component_fails_closed(missing_kind):
    anchor = _group("daily", _cost(rows=40, selected=400, projection=320))
    if missing_kind == "none":
        secondary = _group("events", None)
    elif missing_kind == "unavailable":
        secondary = _group(
            "events", None,
            unavailable=ScanCostUnavailable(dataset="events", reason="footer timeout"),
        )
    else:
        secondary = _group(
            "events", _cost(rows=10, selected=100, projection=80, basis="unknown")
        )
    request = BatchDataRequest(
        anchor_source_scope=anchor.source_scope, groups=[anchor, secondary]
    )

    with pytest.raises(CompositeExecutionScanCostUnavailable):
        request.execution_scan_cost_map()


def test_single_source_preserves_existing_cost_object_and_scope_contract():
    anchor = _group("daily", _cost(rows=40, selected=400, projection=320))
    request = BatchDataRequest(anchor_source_scope=anchor.source_scope, groups=[anchor])

    got = request.execution_scan_cost_map()

    assert got == request.scan_cost_map
    assert got[anchor.source_scope_key] is anchor.scan_cost
    wrong = BatchDataRequest(
        anchor_source_scope=SourceScopeId(dataset="not-the-anchor"), groups=[anchor]
    )
    with pytest.raises(CompositeExecutionScanCostUnavailable):
        wrong.execution_scan_cost_map()


def test_multi_source_rejects_anchor_key_not_present_in_typed_groups():
    first = _group("daily", _cost(rows=40, selected=400, projection=320))
    second = _group("events", _cost(rows=10, selected=100, projection=80))
    request = BatchDataRequest(
        anchor_source_scope=SourceScopeId(dataset="invented"),
        groups=[first, second],
    )

    with pytest.raises(CompositeExecutionScanCostUnavailable):
        request.execution_scan_cost_map()


@pytest.mark.parametrize("bad", [1.5, "3"])
def test_fractional_or_text_byte_cost_is_not_silently_coerced(bad):
    anchor = _group("daily", _cost(rows=40, selected=400, projection=320))
    secondary_cost = _cost(rows=10, selected=100, projection=80)
    secondary_cost.projection_bytes = bad
    secondary = _group("events", secondary_cost)
    request = BatchDataRequest(
        anchor_source_scope=anchor.source_scope, groups=[anchor, secondary]
    )

    with pytest.raises(CompositeExecutionScanCostUnavailable):
        request.execution_scan_cost_map()
