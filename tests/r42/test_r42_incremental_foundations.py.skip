from __future__ import annotations

import pandas as pd

from planner.logical_plan import PlanNode
from runtime.change_impact import (
    ColumnIdentity,
    DataChangeSet,
    compute_data_change_impact,
)
from storage.delta_store import (
    DeltaManifest,
    DeltaReadSelection,
    read_delta_partition,
    select_delta_fragments,
    write_delta_fragment,
)


def _fragment(rows):
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime([row[0] for row in rows]),
            "asset": [row[1] for row in rows],
            "value": [row[2] for row in rows],
            "is_valid": [1] * len(rows),
        }
    )


def test_r42_072_delta_index_prunes_date_and_asset_fragments(tmp_path, monkeypatch):
    manifest = DeltaManifest(base="missing.parquet")
    write_delta_fragment(
        tmp_path,
        _fragment([("2024-01-02", "A", 1.0)]),
        manifest=manifest,
    )
    write_delta_fragment(
        tmp_path,
        _fragment([("2024-02-02", "B", 2.0)]),
        manifest=manifest,
    )

    selection = DeltaReadSelection.build(
        time_range=("2024-02-01", "2024-02-29"), assets=["B"]
    )
    selected = select_delta_fragments(manifest, selection)
    assert [entry["seq"] for entry in selected] == [1]

    reads = []
    real_read = pd.read_parquet

    def recording_read(path, *args, **kwargs):
        reads.append(str(path))
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read)
    result = read_delta_partition(
        tmp_path,
        time_range=("2024-02-01", "2024-02-29"),
        assets=["B"],
        columns=["value"],
    )
    assert result.to_dict("records") == [
        {"datetime": pd.Timestamp("2024-02-02"), "asset": "B", "value": 2.0}
    ]
    assert len(reads) == 1
    assert "gen_00001" in reads[0]


def test_r42_072_base_generation_never_overrides_first_delta(tmp_path):
    base = _fragment([("2024-01-02", "A", 1.0)])
    base.to_parquet(tmp_path / "base.parquet", index=False)
    manifest = DeltaManifest(base="base.parquet")
    write_delta_fragment(
        tmp_path,
        _fragment([("2024-01-02", "A", 9.0)]),
        manifest=manifest,
    )

    result = read_delta_partition(tmp_path)
    assert result["value"].tolist() == [9.0]


def test_r42_085_typed_change_set_preserves_time_series_instruments(monkeypatch):
    from ir.types import AxisEffectKind, register_axis_effect

    register_axis_effect("ts_mean", AxisEffectKind.TIME_SERIES)
    plan = PlanNode(
        "ts_mean",
        [PlanNode("column", attrs={"name": "close", "source_table": "daily"})],
        attrs={"window": 5},
    )
    change = DataChangeSet.build(
        column=ColumnIdentity(dataset="daily", field="close"),
        changed_start="2024-01-10",
        instruments=["A", "B"],
    )
    impact = compute_data_change_impact(plan, change)
    assert impact.root is not None
    assert impact.root.end == "2024-01-16"
    assert impact.instruments == frozenset({"A", "B"})


def test_r42_085_cross_section_expands_instrument_scope():
    from ir.types import AxisEffectKind, register_axis_effect

    register_axis_effect("rank", AxisEffectKind.CROSS_SECTION)
    plan = PlanNode(
        "rank",
        [PlanNode("column", attrs={"name": "close", "source_table": "daily"})],
    )
    change = DataChangeSet.build(
        column=ColumnIdentity(dataset="daily", field="close"),
        changed_start="2024-01-10",
        instruments=["A"],
    )
    impact = compute_data_change_impact(plan, change)
    assert impact.root is not None
    assert impact.instruments is None
