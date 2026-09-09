from dataclasses import dataclass

import numpy as np
import pytest

from factor_assets.adapters.qe_joint_evidence import adapt_qe_distributions


@dataclass(frozen=True)
class Artifact:
    metric_id: str
    stat_names: tuple[str, ...]
    samples: np.ndarray
    provenance: dict


def provenance(**updates):
    value = {"resampling_plan_ref": "plan", "replicate_ids": ("r1", "r2"),
             "clock_ref": "clock", "time_ids": ("t1", "t2"),
             "factor_ids": ("factor", "other"), "recipe_hash":"recipe",
             "fitted_state_hash":"state", "data_snapshot_hash":"data",
             "universe_hash":"universe", "label_hash":"label",
             "value_artifact_hash":"values",
             "resampling_plan_content_hash":"plan-content",
             "sample_identity_hash":"samples","time_identity_hash":"times",
             "common_mask_hash":"mask"}
    value.update(updates)
    return value


def artifact(metric, values, **updates):
    return Artifact(metric, ("factor", "other"), np.asarray(values, dtype=float),
                    provenance(**updates))


def adapt(artifacts):
    return adapt_qe_distributions(
        evidence_id="qe", comparison_context_hash="ctx", factor_id="factor",
        metric_ids=("ic", "turnover"), metric_units=("ratio", "fraction"),
        window_ids=("w",), scenario_ids=("s",), artifacts=artifacts,
        qualification_scope="research", source_tree_hash="tree",
        implementation_hash="impl", route="cpu", backend="numpy",
        parameter_domain_hash="domain", metric_instance_hash="instance")


def valid_grid():
    return {("w", "s", "ic"): artifact("ic", [[.1, 9], [.2, 8]]),
            ("w", "s", "turnover"): artifact("turnover", [[.3, 7], [.4, 6]])}


def test_adapter_preserves_named_metric_plan_replicate_and_factor_axes():
    result = adapt(valid_grid())
    assert result.metric_ids == ("ic", "turnover")
    assert result.metric_units == ("ratio", "fraction")
    assert result.resampling_plan_ref == "plan"
    assert result.replicate_ids == ("r1", "r2")
    assert result.samples == ((((.1, .3),),), (((.2, .4),),))


@pytest.mark.parametrize("mutation", ("alias", "plan", "replicates", "nonfinite"))
def test_adapter_rejects_identity_axis_and_finiteness_mutations(mutation):
    grid = valid_grid()
    if mutation == "alias":
        grid[("w", "s", "ic")] = artifact("rank_ic", [[.1, 9], [.2, 8]])
    elif mutation == "plan":
        grid[("w", "s", "ic")] = artifact("ic", [[.1, 9], [.2, 8]], resampling_plan_ref="other")
    elif mutation == "replicates":
        grid[("w", "s", "ic")] = artifact("ic", [[.1, 9], [.2, 8]], replicate_ids=("r2", "r1"))
    else:
        grid[("w", "s", "ic")] = artifact("ic", [[np.inf, 9], [.2, 8]])
    with pytest.raises(ValueError):
        adapt(grid)
