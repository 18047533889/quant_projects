import json
from pathlib import Path

import pytest

from quant_evaluator.scripts import coverage_memory_profile as profile
from quant_evaluator.scripts.benchmark_real_cos_factor_batch import _estimate_f32_peak_bytes


def test_profile_is_bound_to_published_coverage_measurement():
    receipt = json.loads((Path(__file__).resolve().parents[1] / "docs" /
                          "benchmarks" / profile.EVIDENCE_FILE).read_text())
    assert receipt["parity_pass"] is True
    assert receipt["request"]["metrics"] == ["coverage"]
    assert receipt["request"]["shape"] == [2586, 5461, 32]
    assert receipt["request"]["dtype"] == "float64"
    assert receipt["request"]["manifest_sha256"] == profile.MANIFEST_SHA256
    assert receipt["parent_peak_rss_kib"] == profile.PARENT_PEAK_RSS_KIB
    assert len(receipt["runs"]) == 6
    assert max(run["peak_rss_kib"] for run in receipt["runs"]) == profile.WORKER_PEAK_RSS_KIB
    peak = profile.measured_coverage_peak_bytes(profile.MANIFEST_SHA256, ("coverage",))
    assert peak >= profile.ARRAY_UPPER_BOUND_BYTES * 8
    combined = (profile.PARENT_PEAK_RSS_KIB + profile.WORKER_PEAK_RSS_KIB) * 1024
    assert peak >= combined * 1.2
    assert _estimate_f32_peak_bytes(profile.MANIFEST_SHA256, metric_ids=("coverage",)) == peak
    assert _estimate_f32_peak_bytes(profile.MANIFEST_SHA256) == 50 * 1024**3


@pytest.mark.parametrize("metric_ids", [None, (), ("rank_ic",),
                                      ("coverage", "rank_ic"), ["coverage"], "coverage"])
def test_other_metric_profiles_keep_the_conservative_envelope(metric_ids):
    assert profile.measured_coverage_peak_bytes(profile.MANIFEST_SHA256, metric_ids) is None
    assert _estimate_f32_peak_bytes(profile.MANIFEST_SHA256, metric_ids=metric_ids) == 50 * 1024**3


def test_unseen_manifest_does_not_borrow_coverage_evidence():
    assert profile.measured_coverage_peak_bytes("0" * 64, ("coverage",)) is None
    assert _estimate_f32_peak_bytes("0" * 64, metric_ids=("coverage",)) == _estimate_f32_peak_bytes("0" * 64)
