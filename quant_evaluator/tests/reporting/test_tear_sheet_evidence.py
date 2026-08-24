"""
Unit tests for the tear-sheet reporting-layer refactor.

Pins that the reporting ``MetricArtifact`` is a thin wrapper around the
canonical contracts.metric_artifacts artifact + EvidenceStatus, that a
non-COMPUTED status renders a NOT_COMPUTED placeholder (never 0.0), and that
the historical panel contract (22 panels, same names, same ChartSpec output
shape) is preserved.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_evaluator.reporting.tear_sheet import (
    EvaluationResult,
    generate_tear_sheet,
    MetricArtifact,
    metric_artifact,
    wrap_artifact,
    NOT_COMPUTED,
    INSTITUTIONAL_PANELS,
)
from quant_evaluator.contracts.evidence_status import (
    EvidenceStatus,
    EvidenceReasonCode,
    evidence_for_computed,
    evidence_for_not_computed,
)
from quant_evaluator.contracts.metric_artifacts import ScalarMetricArtifact
from quant_evaluator.reporting.chart_spec import ChartSpec


def test_metric_artifact_holds_canonical_source_and_status():
    """The reporting wrapper carries the canonical artifact + status."""
    art = metric_artifact("ic_recent_vs_full", ic=[0.06])
    assert art.name == "ic_recent_vs_full"
    assert art.computed is True
    assert art.status is EvidenceStatus.COMPUTED
    assert art.data["ic"] == [0.06]


def test_wrap_artifact_accepts_canonical_contract_artifact():
    """wrap_artifact turns a canonical artifact into a COMPUTED wrapper."""
    canonical = ScalarMetricArtifact(metric_id="rank_ic", domain="ic", values=[0.1, 0.2])
    wrapped = wrap_artifact(canonical)
    assert isinstance(wrapped, MetricArtifact)
    assert wrapped.computed is True
    assert wrapped.source_artifact is canonical


def test_wrap_artifact_rejects_unknown_input():
    with pytest.raises(TypeError):
        wrap_artifact("not an artifact")


def test_wrap_artifact_passthrough():
    art = metric_artifact("rolling_rank_ic", windows={"a": [1]})
    assert wrap_artifact(art) is art


def test_non_computed_wrapper_renders_placeholder_not_zero():
    """A canonical artifact with a non-COMPUTED status must render NOT_COMPUTED,
    never a fabricated 0.0."""
    canonical = ScalarMetricArtifact(metric_id="rank_ic", domain="ic", values=[0.1])
    wrapped = MetricArtifact.from_canonical(
        canonical, status=EvidenceStatus.INSUFFICIENT_DATA
    )
    assert wrapped.computed is False
    result = EvaluationResult()
    result.artifacts = {"rolling_rank_ic": wrapped}
    panels = generate_tear_sheet(result)
    spec = panels["rolling_rank_ic"]
    assert spec.data.get("status") == NOT_COMPUTED
    assert spec.data.get("status") != 0.0


def test_absent_artifact_still_not_computed():
    """Absent artifacts still render NOT_COMPUTED (regression guard)."""
    result = EvaluationResult()
    panels = generate_tear_sheet(result)
    for panel_key, title, _, _ in INSTITUTIONAL_PANELS:
        spec = panels[panel_key]
        assert spec.data.get("status") == NOT_COMPUTED


def test_all_22_panels_present():
    keys = {p[0] for p in INSTITUTIONAL_PANELS}
    assert len(keys) == 22


def test_evaluation_result_defaults_are_none_not_zero():
    """The default EvaluationResult must NOT fabricate 0.0 scalars."""
    result = EvaluationResult()
    assert result.ic_mean is None
    assert result.max_drawdown is None
    assert result.sharpe_ratio is None
    assert result.avg_turnover is None
    assert result.status is EvidenceStatus.NOT_COMPUTED


def test_summary_statistics_render_none_as_em_dash_not_zero():
    """None scalar fields render '—' in the summary table, never 0."""
    panels = generate_tear_sheet(EvaluationResult())
    stats = panels["summary_statistics_table"].data["statistics"]
    for label, value in stats.items():
        assert value == "—", f"{label} should render em-dash, got {value!r}"


def test_metric_artifact_reason_code_when_not_computed():
    wrapper = MetricArtifact.not_computed("data_quality_missingness")
    assert wrapper.computed is False
    assert wrapper.reason_code == EvidenceReasonCode.NOT_YET_COMPUTED.value
