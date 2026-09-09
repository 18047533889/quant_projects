"""Outer E2E-E composition for bounded DA revision replay into modeling."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from data_access.core.missingness import MissingReason, MissingReasonPlane
from data_access.r30.change_impact import AffectedFactor, plan_minimal_recompute
from data_access.r30.data_change import DataChangeSet
from factor_preprocess.contracts.feature_bundle import AxisRef, FeatureBundle
from factor_preprocess.transforms.freshness import days_since_update
from factor_preprocess.transforms.missingness import forward_fill, missing_indicator
from modeling.dataset import FeatureSchema
from modeling.ledger import stable_hash


class RevisionSnapshotProvider(Protocol):
    """Read-only PIT provider; implementations supply data, never factor math."""

    def load(self, snapshot_ref: str, *, start: str, end: str,
             instruments: tuple[str, ...]) -> pd.DataFrame: ...


@dataclass(frozen=True)
class RevisionReplayEvidence:
    change: DataChangeSet
    replay_range: AffectedFactor
    old_bundle: FeatureBundle
    revised_bundle: FeatureBundle
    old_manifest: FeatureSchema
    revised_manifest: FeatureSchema
    old_feature_identity: str
    revised_feature_identity: str


def _sessions(calendar: Any, start: str, end: str) -> tuple[pd.Timestamp, ...]:
    if not hasattr(calendar, "sessions_between"):
        raise ValueError("calendar must expose sessions_between for coordinate validation")
    return tuple(pd.Timestamp(value) for value in calendar.sessions_between(start, end))


def _normalize_provider_frame(frame: pd.DataFrame, *, calendar: Any, start: str, end: str,
                              instruments: tuple[str, ...]) -> pd.DataFrame:
    required = {"date", "asset_id", "value"}
    if set(frame.columns) != required:
        raise ValueError("provider frame must contain exactly date/asset_id/value")
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    if normalized.duplicated(["date", "asset_id"]).any():
        raise ValueError("provider frame must have unique date/asset_id/value rows")
    expected = {(day, asset) for day in _sessions(calendar, start, end) for asset in instruments}
    actual = set(zip(normalized["date"], normalized["asset_id"]))
    if actual != expected:
        missing, unexpected = expected - actual, actual - expected
        raise ValueError(
            f"provider coordinates differ from declared range: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    ordered = normalized.sort_values(["date", "asset_id"], kind="stable").reset_index(drop=True)
    return ordered


def _preprocess(ordered: pd.DataFrame, *, max_lag: int) -> tuple[np.ndarray, MissingReasonPlane]:
    if not ordered.groupby("asset_id", sort=False)["date"].is_monotonic_increasing.all():
        raise ValueError("provider rows must form ordered asset time series")
    original_missing = missing_indicator(ordered).to_numpy(dtype=bool)
    filled_values = forward_fill(ordered, max_lag=max_lag).to_numpy(dtype=np.float64)
    age = days_since_update(ordered).to_numpy(dtype=np.float64)
    filled = original_missing & np.isfinite(filled_values)
    usable = np.isfinite(filled_values)
    reasons = np.where(original_missing, MissingReason.RAW_MISSING.value,
                       MissingReason.OBSERVED.value)
    shape = (len(ordered), 1)
    return filled_values, MissingReasonPlane(
        reasons.reshape(shape), original_missing.reshape(shape), filled.reshape(shape),
        usable.reshape(shape), age.reshape(shape),
    )


def _artifact(frame: pd.DataFrame, values: np.ndarray, plane: MissingReasonPlane, *,
              snapshot_ref: str, data_watermark: str, max_lag: int
              ) -> tuple[FeatureBundle, FeatureSchema, str]:
    ordered = frame.sort_values(["date", "asset_id"], kind="stable").reset_index(drop=True)
    # The composed acceptance path is deliberately single-instrument; this
    # reshape makes the time/asset axes explicit rather than inventing a panel.
    assets = tuple(ordered["asset_id"].drop_duplicates())
    times = tuple(ordered["date"].drop_duplicates())
    if len(assets) != 1 or len(ordered) != len(times):
        raise ValueError("E2E-E bounded replay currently requires one complete instrument series")
    primary = np.asarray(values, dtype=np.float64).reshape(len(times), 1, 1)
    plane3 = MissingReasonPlane(
        plane.reasons.reshape(primary.shape), plane.original_missing.reshape(primary.shape),
        plane.filled.reshape(primary.shape), plane.usable.reshape(primary.shape),
        plane.age.reshape(primary.shape),
    )
    identity = stable_hash({
        "snapshot_ref": snapshot_ref, "data_watermark": data_watermark,
        "times": [str(value) for value in times], "assets": list(assets),
        "values": primary.tolist(), "original_missing": plane3.original_missing.tolist(),
        "filled": plane3.filled.tolist(), "age": plane3.age.tolist(), "max_lag": max_lag,
    })
    bundle = FeatureBundle.from_primary_with_auxiliary(
        bundle_id="e2e-e:" + identity, primary_values=primary, feature_ids=("announcement",),
        time_axis=AxisRef("time", times, "datetime64[ns]"),
        asset_axis=AxisRef("asset", assets, "str"),
        original_validity_mask=~plane3.original_missing,
        missing_reason_plane=plane3, freshness_values=plane3.age,
        fitted_state_refs=("snapshot:" + snapshot_ref, "watermark:" + data_watermark),
        policy_id="bounded-forward-fill", config_hash=identity,
    )
    # No cluster/version authority is in this edge's scope, so do not fabricate
    # the refs required for a complete production FeatureSchema. The modeling
    # identity remains explicit but research-only until E2E-G supplies them.
    schema = FeatureSchema(columns=("announcement",))
    feature_identity = stable_hash({
        "modeling_schema": schema.to_dict(), "feature_bundle_id": bundle.bundle_id,
        "snapshot_ref": snapshot_ref, "data_watermark": data_watermark,
        "preprocess_config_hash": identity,
    })
    return bundle, schema, feature_identity


def replay_announcement_revision(
    change: DataChangeSet, demand: Mapping[str, Any], provider: RevisionSnapshotProvider,
    *, calendar: Any, full_start: str, full_end: str, max_lag: int,
) -> RevisionReplayEvidence:
    """Execute DA range planning, bounded FP replay, and manifest derivation."""
    if not change.snapshot_before or not change.snapshot_after:
        raise ValueError("revision replay requires before/after snapshot identities")
    if not change.data_watermark_before or not change.data_watermark_after:
        raise ValueError("revision replay requires before/after data watermarks")
    plans = plan_minimal_recompute(change, [demand], calendar=calendar)
    if len(plans) != 1 or plans[0].affected_start is None or plans[0].affected_end is None:
        raise ValueError("revision must resolve to one bounded factor replay range")
    plan = plans[0]
    traits = demand.get("operator_traits")
    if traits is None or traits.backward_input_horizon < max_lag:
        raise ValueError("operator traits must cover bounded-fill warmup")
    instruments = tuple(plan.affected_instruments or change.changed_instruments)
    if not instruments:
        raise ValueError("bounded replay requires explicit affected instruments")

    old_frame = _normalize_provider_frame(
        provider.load(change.snapshot_before, start=full_start, end=full_end,
                      instruments=instruments),
        calendar=calendar, start=full_start, end=full_end, instruments=instruments,
    )
    old_values, old_plane = _preprocess(old_frame, max_lag=max_lag)
    old_bundle, old_manifest, old_feature_identity = _artifact(
        old_frame, old_values, old_plane, snapshot_ref=change.snapshot_before,
        data_watermark=change.data_watermark_before, max_lag=max_lag,
    )

    replay_input_start = str(calendar.offset(
        date.fromisoformat(str(plan.affected_start)[:10]), -max_lag, clamp=True
    ))
    # Age remains meaningful after bounded fill expires. Read the requested
    # PIT tail to locate the next original observation, then require DA's
    # declared output range to cover every age value that the revision changes.
    replay_tail = _normalize_provider_frame(
        provider.load(change.snapshot_after, start=replay_input_start,
                      end=full_end, instruments=instruments),
        calendar=calendar, start=replay_input_start, end=full_end,
        instruments=instruments,
    )
    changed_end = pd.Timestamp(change.changed_time_range[1])
    required_age_end = changed_end
    tail_sessions = _sessions(calendar, replay_input_start, full_end)
    for instrument in instruments:
        observed_after = replay_tail.loc[
            (replay_tail["asset_id"] == instrument)
            & (replay_tail["date"] > changed_end)
            & replay_tail["value"].notna(), "date"
        ]
        next_observation = observed_after.min() if not observed_after.empty else None
        impacted = [day for day in tail_sessions
                    if day > changed_end and (next_observation is None or day < next_observation)]
        if impacted:
            required_age_end = max(required_age_end, impacted[-1])
    if pd.Timestamp(plan.affected_end) < required_age_end:
        raise ValueError(
            "declared replay range does not cover downstream age influence; "
            f"requires through {required_age_end.date()}"
        )
    replay_frame = replay_tail.loc[
        replay_tail["date"] <= pd.Timestamp(plan.affected_end)
    ].reset_index(drop=True)
    replay_values, replay_plane = _preprocess(replay_frame, max_lag=max_lag)
    revised_frame = old_frame.copy()
    revised_values = np.array(old_values, copy=True)
    revised_channels = {
        "reasons": np.array(old_plane.reasons, copy=True),
        "original_missing": np.array(old_plane.original_missing, copy=True),
        "filled": np.array(old_plane.filled, copy=True),
        "usable": np.array(old_plane.usable, copy=True),
        "age": np.array(old_plane.age, copy=True),
    }
    keys = list(zip(pd.to_datetime(revised_frame["date"]), revised_frame["asset_id"]))
    positions = {key: idx for idx, key in enumerate(keys)}
    in_affected = replay_frame["date"].between(
        pd.Timestamp(plan.affected_start), pd.Timestamp(plan.affected_end)
    )
    patch_frame = replay_frame.loc[in_affected].reset_index(drop=True)
    replay_positions = np.flatnonzero(in_affected.to_numpy())
    patch_indices = [positions[(pd.Timestamp(row.date), row.asset_id)]
                     for row in patch_frame.itertuples()]
    revised_frame.loc[patch_indices, "value"] = patch_frame["value"].to_numpy()
    for target, source in (
        (revised_values, replay_values[replay_positions]),
        (revised_channels["reasons"], replay_plane.reasons.ravel()[replay_positions]),
        (revised_channels["original_missing"], replay_plane.original_missing.ravel()[replay_positions]),
        (revised_channels["filled"], replay_plane.filled.ravel()[replay_positions]),
        (revised_channels["usable"], replay_plane.usable.ravel()[replay_positions]),
        (revised_channels["age"], replay_plane.age.ravel()[replay_positions]),
    ):
        target.reshape(-1)[patch_indices] = np.asarray(source).reshape(-1)
    revised_reason = MissingReasonPlane(**revised_channels)
    revised_bundle, revised_manifest, revised_feature_identity = _artifact(
        revised_frame, revised_values, revised_reason,
        snapshot_ref=change.snapshot_after, data_watermark=change.data_watermark_after,
        max_lag=max_lag,
    )
    return RevisionReplayEvidence(
        change, plan, old_bundle, revised_bundle, old_manifest, revised_manifest,
        old_feature_identity, revised_feature_identity,
    )


__all__ = ["RevisionReplayEvidence", "RevisionSnapshotProvider", "replay_announcement_revision"]
