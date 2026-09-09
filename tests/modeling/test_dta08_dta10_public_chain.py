from datetime import date

import numpy as np
import pandas as pd
import pytest

from data_access.core.missingness import MissingReason, MissingReasonPlane
from data_access.r30.change_impact import plan_minimal_recompute
from data_access.r30.change_impact_types import AxisEffect, OperatorDependencyTraits, TimeAxisKind
from data_access.r30.data_change import DataChangeSet
from factor_engine.runtime.incremental_contract import IncrementalMode, resolve_incremental_contract
from factor_engine.runtime.stateful_incremental import compute_state_node_identity
from factor_preprocess.contracts.feature_bundle import AxisRef, FeatureBundle
from modeling.dataset import FeatureField, FeatureSchema, PanelDataset, panel_telemetry
from modeling.contracts import LabelContract
from modeling.trainer import PreprocessingSpec, _extract_matrices, fit_preprocessing
from modeling.walk_forward import purge_before_boundary


def _reason_plane():
    reasons = np.array([
        [[MissingReason.SOURCE_OUTAGE.value], [MissingReason.NOT_LISTED.value]],
        [[MissingReason.OBSERVED.value], [MissingReason.OBSERVED.value]],
    ])
    original = np.array([[[True], [True]], [[False], [False]]])
    filled = np.array([[[True], [False]], [[False], [False]]])
    usable = np.array([[[True], [False]], [[True], [True]]])
    age = np.array([[[1.0], [np.nan]], [[0.0], [0.0]]])
    return MissingReasonPlane(reasons, original, filled, usable, age)


def test_explicit_missing_reasons_survive_da_fp_modeling_and_drive_semantic_coverage():
    plane = _reason_plane()
    primary = np.array([[[10.0], [999.0]], [[30.0], [40.0]]])
    bundle = FeatureBundle.from_primary_with_auxiliary(
        bundle_id="dta08",
        primary_values=primary,
        feature_ids=("alpha",),
        time_axis=AxisRef("time", (1, 2), "int64"),
        asset_axis=AxisRef("asset", ("A", "B"), "str"),
        original_validity_mask=~plane.original_missing,
        missing_reason_plane=plane,
    )
    carried = bundle.get_missing_reason_plane()
    assert carried.reasons[0, 0, 0] == MissingReason.SOURCE_OUTAGE.value
    assert carried.reasons[0, 1, 0] == MissingReason.NOT_LISTED.value
    assert carried.original_missing[0, 0, 0] and carried.filled[0, 0, 0]
    assert carried.original_missing[0, 1, 0] and not carried.filled[0, 1, 0]

    field = FeatureField(
        "alpha", "value:alpha", "recipe:fill-outage-only", "state:v1", "float64",
        mask_ref="mask:alpha", missing_reason_ref="reason:alpha",
        missing_age_ref="age:alpha", cluster_version_ref="cluster:v1",
    )
    schema = FeatureSchema(
        columns=("alpha",), fields=(field,), consumer_profile="linear-v1",
        feature_set_version_ref="features:v1",
    )
    flat_plane = MissingReasonPlane(
        carried.reasons.reshape(4, 1), carried.original_missing.reshape(4, 1),
        carried.filled.reshape(4, 1), carried.usable.reshape(4, 1), carried.age.reshape(4, 1),
    )
    frame = pd.DataFrame({
        "date": [1, 1, 2, 2], "stock": ["A", "B", "A", "B"],
        "alpha": primary.reshape(4), "label": [0.1, 0.2, 0.3, 0.4],
    })
    ds = PanelDataset(
        frame, feature_schema=schema, label_col="label", missing_reason_plane=flat_plane,
        label_missing_reasons=np.array([
            "observed", "observed", "observed", MissingReason.LABEL_NOT_YET_MATURE.value,
        ]),
    )
    X, _, _, _, usable_rows = ds.as_matrix()
    assert np.isnan(X[1, 0])  # numeric 999 cannot override NOT_LISTED semantics
    assert usable_rows.tolist() == [True, False, True, False]
    telemetry = panel_telemetry(ds)
    assert telemetry["missingness_coverage"] == {
        "total": 4, "observed": 2, "usable": 3, "filled": 1, "original_missing": 2,
    }
    assert telemetry["label_not_yet_mature_count"] == 1
    assert schema.to_dict()["fields"][0]["missing_reason_ref"] == "reason:alpha"

    preprocessing = fit_preprocessing(ds, PreprocessingSpec(steps=("imputer",)))
    fit_X, fit_y, _, _ = _extract_matrices(ds, preprocessing, None)
    assert fit_X.shape == (2, 1)
    np.testing.assert_allclose(fit_y, [0.1, 0.3])
    purged = purge_before_boundary(ds, 3, LabelContract("next_bar"))
    assert purged.missing_reason_plane is not None
    assert purged.missing_reason_plane.reasons.shape == (2, 1)
    assert purged.label_missing_reasons.shape == (2,)


def test_reason_plane_never_infers_two_nan_causes_and_rejects_fabricated_fill():
    plane = _reason_plane()
    assert plane.reasons[0, 0, 0] != plane.reasons[0, 1, 0]
    with pytest.raises(ValueError, match="subset"):
        MissingReasonPlane(
            [[MissingReason.OBSERVED.value]], [[False]], [[True]], [[True]], [[0.0]],
        )


class _Sessions:
    def __init__(self, days):
        self.days = tuple(date.fromisoformat(day) for day in days)

    def offset(self, base, amount, clamp=True):
        idx = self.days.index(base)
        target = idx + amount
        if clamp:
            target = max(0, min(target, len(self.days) - 1))
        return self.days[target]


def _rolling_mean(values, window):
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        out[i] = np.mean(values[i - window + 1:i + 1])
    return out


def test_late_revision_has_explicit_watermarks_bounded_window_and_full_recompute_parity():
    days = ("2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07",
            "2025-01-08", "2025-01-09", "2025-01-10", "2025-01-13")
    change = DataChangeSet(
        dataset="prices", change_kind="revision", changed_time_range=("2025-01-08", "2025-01-08"),
        event_time_range=("2025-01-08", "2025-01-08"),
        knowledge_time_range=("2025-01-13T08:00:00Z", "2025-01-13T08:00:00Z"),
        snapshot_before="snapshot:old", snapshot_after="snapshot:revised",
        data_watermark_before="2025-01-10T08:00:00Z",
        data_watermark_after="2025-01-13T08:00:00Z",
        changed_instruments=("A",), changed_columns=("close",),
    )
    traits = OperatorDependencyTraits(
        backward_input_horizon=2, forward_output_horizon=2,
        axis_effect=AxisEffect.TIME_SERIES, time_axis=TimeAxisKind.TRADING_BARS,
    )
    plan = plan_minimal_recompute(change, [{
        "factor_id": "ma3", "datasets": ("prices",), "instruments": ("A",),
        "operator_traits": traits, "axis_effect": "time_series",
    }], calendar=_Sessions(days))
    assert len(plan) == 1
    assert (plan[0].affected_start, plan[0].affected_end) == ("2025-01-06", "2025-01-10")
    payload = change.to_dict()
    assert payload["snapshot_after"] == "snapshot:revised"
    assert payload["data_watermark_after"] == "2025-01-13T08:00:00Z"
    assert payload["event_time_range"] != payload["knowledge_time_range"]

    revised = np.arange(len(days), dtype=float)
    revised[4] = 100.0
    full = _rolling_mean(revised, 3)
    context = revised[2:7]  # affected_start..affected_end, includes required history
    local = _rolling_mean(context, 3)
    np.testing.assert_allclose(local[2:], full[4:7])


def test_snapshot_revision_changes_checkpoint_identity_and_ema_policy_is_not_polluted_resume():
    old = compute_state_node_identity(
        canonical="ts_ema", params={"span": 20}, source_scope="snapshot:old",
        input_columns=("close",), missing_support_policy="reason-aware-v1",
    )
    revised = compute_state_node_identity(
        canonical="ts_ema", params={"span": 20}, source_scope="snapshot:revised",
        input_columns=("close",), missing_support_policy="reason-aware-v1",
    )
    assert old.stable_key != revised.stable_key
    contract = resolve_incremental_contract("ts_ema", {"span": 20})
    assert contract.incremental_mode in {IncrementalMode.CHECKPOINTED_STATE, IncrementalMode.FULL_REPLAY}
    assert contract.revision_policy in {"affected_domain", "full_replay"}
