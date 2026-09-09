from datetime import date

import numpy as np
import pandas as pd
import pytest

from data_access.r30.change_impact_types import AxisEffect, OperatorDependencyTraits, TimeAxisKind
from data_access.r30.data_change import DataChangeSet
from factor_preprocess.transforms.freshness import days_since_update
from factor_preprocess.transforms.missingness import forward_fill, missing_indicator
from jobs.e2e_e_revision_preprocess_modeling import replay_announcement_revision


class _Sessions:
    def __init__(self, days):
        self.days = tuple(date.fromisoformat(day) for day in days)

    def offset(self, base, amount, clamp=True):
        index = self.days.index(base)
        target = index + amount
        if clamp:
            target = max(0, min(target, len(self.days) - 1))
        return self.days[target]

    def sessions_between(self, start, end):
        lower, upper = date.fromisoformat(str(start)[:10]), date.fromisoformat(str(end)[:10])
        return tuple(day for day in self.days if lower <= day <= upper)


class _SyntheticPITProvider:
    def __init__(self, snapshots, *, shuffle=False, omit=None):
        self.snapshots = snapshots
        self.calls = []
        self.shuffle = shuffle
        self.omit = omit

    def load(self, snapshot_ref, *, start, end, instruments):
        self.calls.append((snapshot_ref, start, end, instruments))
        frame = self.snapshots[snapshot_ref]
        mask = (
            frame["asset_id"].isin(instruments)
            & frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        )
        result = frame.loc[mask].copy()
        if self.omit is not None and snapshot_ref == "snapshot:revised":
            result = result.loc[result["date"] != pd.Timestamp(self.omit)]
        if self.shuffle:
            result = result.sample(frac=1.0, random_state=17)
        return result.reset_index(drop=True)


def test_announcement_revision_replays_bounded_fp_channels_and_changes_model_manifest():
    days = ("2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07",
            "2025-01-08", "2025-01-09", "2025-01-10", "2025-01-13",
            "2025-01-14", "2025-01-15")
    dates = pd.to_datetime(days)
    old = pd.DataFrame({
        "date": dates, "asset_id": ["A"] * len(days),
        "value": [1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0],
    })
    revised = old.copy()
    revised.loc[revised["date"] == pd.Timestamp("2025-01-08"), "value"] = 10.0
    provider = _SyntheticPITProvider(
        {"snapshot:old": old, "snapshot:revised": revised}, shuffle=True
    )
    change = DataChangeSet(
        dataset="announcements", change_kind="revision",
        changed_time_range=("2025-01-08", "2025-01-08"),
        event_time_range=("2024-12-31", "2024-12-31"),
        knowledge_time_range=("2025-01-13T08:00:00Z", "2025-01-13T08:00:00Z"),
        snapshot_before="snapshot:old", snapshot_after="snapshot:revised",
        data_watermark_before="2025-01-10T08:00:00Z",
        data_watermark_after="2025-01-13T08:00:00Z",
        changed_instruments=("A",), changed_columns=("value",),
    )
    demand = {
        "factor_id": "announcement", "datasets": ("announcements",),
        "instruments": ("A",), "axis_effect": "time_series",
        "operator_traits": OperatorDependencyTraits(
            backward_input_horizon=2, forward_output_horizon=4,
            axis_effect=AxisEffect.TIME_SERIES, time_axis=TimeAxisKind.TRADING_BARS,
        ),
    }
    short_demand = dict(demand)
    short_demand["operator_traits"] = OperatorDependencyTraits(
        backward_input_horizon=2, forward_output_horizon=2,
        axis_effect=AxisEffect.TIME_SERIES, time_axis=TimeAxisKind.TRADING_BARS,
    )
    with pytest.raises(ValueError, match="downstream age influence"):
        replay_announcement_revision(
            change, short_demand,
            _SyntheticPITProvider({"snapshot:old": old, "snapshot:revised": revised}),
            calendar=_Sessions(days), full_start=days[0], full_end=days[-1], max_lag=2,
        )
    evidence = replay_announcement_revision(
        change, demand, provider, calendar=_Sessions(days),
        full_start=days[0], full_end=days[-1], max_lag=2,
    )

    assert (evidence.replay_range.affected_start, evidence.replay_range.affected_end) == (
        "2025-01-06", "2025-01-14"
    )
    assert provider.calls == [
        ("snapshot:old", "2025-01-02", "2025-01-15", ("A",)),
        ("snapshot:revised", "2025-01-02", "2025-01-15", ("A",)),
    ]
    old_primary = evidence.old_bundle.get_primary_values().reshape(-1)
    revised_primary = evidence.revised_bundle.get_primary_values().reshape(-1)
    np.testing.assert_allclose(old_primary[:4], revised_primary[:4], equal_nan=True)
    np.testing.assert_allclose(old_primary[9:], revised_primary[9:], equal_nan=True)
    np.testing.assert_allclose(revised_primary[4:7], [10.0, 10.0, 10.0])

    old_plane = evidence.old_bundle.get_missing_reason_plane()
    revised_plane = evidence.revised_bundle.get_missing_reason_plane()
    assert old_plane.original_missing.reshape(-1)[4:7].tolist() == [True, True, True]
    assert revised_plane.original_missing.reshape(-1)[4:7].tolist() == [False, True, True]
    assert revised_plane.filled.reshape(-1)[4:7].tolist() == [False, True, True]
    np.testing.assert_allclose(revised_plane.age.reshape(-1)[4:7], [0.0, 1.0, 2.0])
    assert np.isnan(revised_primary[7]) and np.isnan(revised_primary[8])
    np.testing.assert_allclose(revised_plane.age.reshape(-1)[7:9], [5.0, 6.0])
    assert not np.array_equal(
        old_plane.age.reshape(-1)[7:9], revised_plane.age.reshape(-1)[7:9]
    )
    assert not evidence.old_manifest.is_complete and not evidence.revised_manifest.is_complete
    assert evidence.old_feature_identity != evidence.revised_feature_identity
    assert evidence.old_bundle.bundle_id != evidence.revised_bundle.bundle_id
    # The retained artifact remains immutable and still describes pre-revision output.
    assert np.isnan(old_primary[6])

    full_revised = revised.sort_values(["date", "asset_id"]).reset_index(drop=True)
    full_values = forward_fill(full_revised, max_lag=2).to_numpy()
    full_missing = missing_indicator(full_revised).to_numpy(dtype=bool)
    full_age = days_since_update(full_revised).to_numpy()
    np.testing.assert_allclose(revised_primary, full_values, equal_nan=True)
    assert revised_plane.original_missing.reshape(-1).tolist() == full_missing.tolist()
    np.testing.assert_allclose(revised_plane.age.reshape(-1), full_age, equal_nan=True)


def test_revision_provider_must_return_every_declared_coordinate():
    days = ("2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07",
            "2025-01-08", "2025-01-09", "2025-01-10")
    frame = pd.DataFrame({
        "date": pd.to_datetime(days), "asset_id": ["A"] * len(days),
        "value": [1.0, np.nan, np.nan, 4.0, 10.0, np.nan, np.nan],
    })
    provider = _SyntheticPITProvider(
        {"snapshot:old": frame, "snapshot:revised": frame}, omit="2025-01-06"
    )
    change = DataChangeSet(
        dataset="announcements", change_kind="revision",
        changed_time_range=("2025-01-08", "2025-01-08"),
        snapshot_before="snapshot:old", snapshot_after="snapshot:revised",
        data_watermark_before="w:old", data_watermark_after="w:new",
        changed_instruments=("A",), changed_columns=("value",),
    )
    demand = {
        "factor_id": "announcement", "datasets": ("announcements",),
        "instruments": ("A",), "axis_effect": "time_series",
        "operator_traits": OperatorDependencyTraits(
            backward_input_horizon=2, forward_output_horizon=2,
            axis_effect=AxisEffect.TIME_SERIES, time_axis=TimeAxisKind.TRADING_BARS,
        ),
    }
    with pytest.raises(ValueError, match="provider coordinates differ"):
        replay_announcement_revision(
            change, demand, provider, calendar=_Sessions(days),
            full_start=days[0], full_end=days[-1], max_lag=2,
        )
