"""
A-share acceptance test for QE-METRIC.

Exercises the catalog, tear-sheet, library report, ChartSpec, and ArtifactStore
subsystems with realistic A-share factor data patterns.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import random

import numpy as np
import pytest

# Ensure single-threaded BLAS for deterministic numpy results
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from quant_evaluator.metrics.catalog import (
    Domain,
    get_metric_spec,
    list_all_domains,
    list_all_metric_ids,
)
from quant_evaluator.reporting.tear_sheet import (
    generate_tear_sheet,
    EvaluationResult,
)
from quant_evaluator.reporting.library_reports import (
    generate_library_report,
    compare_libraries,
)
from quant_evaluator.reporting.chart_spec import ChartSpec
from quant_evaluator.reporting.artifacts import ArtifactStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_DOMAINS = [
    Domain.IC,
    Domain.RANK_IC,
    Domain.QUANTILE,
    Domain.DRAWDOWN,
    Domain.TURNOVER,
    Domain.TAIL_RISK,
    Domain.COVERAGE,
    Domain.HHI,
    Domain.STABILITY,
    Domain.TEMPORAL,
]

# All 30 metric IDs registered in the catalog
ALL_30_METRIC_IDS = sorted(
    [
        # IC
        "pearson_ic", "spearman_ic", "rank_ic", "ic_summary",
        # RANK_IC
        "rank_ic_time_series", "rank_ic_cross_section",
        # QUANTILE
        "quantile_returns", "quantile_spread", "quantile_stability",
        # DRAWDOWN
        "max_drawdown", "drawdown_duration", "calmar_ratio",
        # TURNOVER
        "turnover_rate", "turnover_cost", "turnover_adjusted_ic",
        # TAIL_RISK
        "var_95", "var_99", "cvar_95", "cvar_99", "skewness", "kurtosis",
        # COVERAGE
        "factor_coverage", "return_coverage", "joint_coverage",
        # HHI
        "hhi_concentration", "hhi_effective_n",
        # STABILITY
        "ic_stability", "turnover_stability", "coverage_stability",
        # TEMPORAL
        "rolling_ic", "ic_decay", "autocorrelation_ic",
    ]
)

TEAR_SHEET_PANEL_COUNT = 22


# ---------------------------------------------------------------------------
# Helpers: realistic A-share data generation
# ---------------------------------------------------------------------------
def _make_ic_series(n: int = 250) -> np.ndarray:
    """Generate a plausible daily IC series for an A-share factor (mean ~0.04)."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.04, scale=0.03, size=n)


def _make_quantile_returns(n_quantiles: int = 5, n_periods: int = 250) -> np.ndarray:
    """Monotonically-ish increasing quantile returns (low->high)."""
    rng = np.random.default_rng(123)
    base = np.linspace(-0.001, 0.005, n_quantiles)
    noise = rng.normal(0, 0.001, size=(n_periods, n_quantiles))
    return (base + noise).mean(axis=0)  # shape (n_quantiles,)


def _make_drawdown_series(n: int = 250) -> np.ndarray:
    """Cumulative-return drawdown curve peaking around -0.15."""
    rng = np.random.default_rng(99)
    returns = rng.normal(0.0003, 0.012, size=n)
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    return cum - running_max  # always <= 0


def _build_evaluation_result() -> EvaluationResult:
    """Construct a full EvaluationResult with realistic A-share patterns."""
    n = 250
    rng = np.random.default_rng(77)

    ic_series = _make_ic_series(n)
    rank_ic_series = _make_ic_series(n) * 0.95  # rank IC slightly lower
    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std())
    icir = ic_mean / ic_std if ic_std > 0 else 0.0

    quantile_returns = _make_quantile_returns()
    quantile_spread = np.array([quantile_returns[-1] - quantile_returns[0]])

    long_short = rng.normal(0.0004, 0.008, size=n)
    cumulative = np.cumsum(long_short)

    drawdown = _make_drawdown_series(n)
    max_dd = float(drawdown.min())  # e.g. -0.15

    turnover = rng.uniform(0.1, 0.5, size=n)
    coverage = rng.uniform(0.85, 0.99, size=n)

    factor_corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    hhi = rng.uniform(0.05, 0.25, size=10)
    ic_stability = rng.uniform(0.02, 0.08, size=20)

    rolling_20 = rng.normal(0.04, 0.02, size=n)
    rolling_60 = rng.normal(0.035, 0.015, size=n)
    rolling_120 = rng.normal(0.03, 0.01, size=n)
    ic_decay = rng.normal(0.04, 0.01, size=12)
    ic_autocorr = rng.normal(0.3, 0.1, size=10)
    coverage_heatmap = rng.uniform(0.85, 0.99, size=(20, 10))

    ann_return = float(long_short.mean() * 252)
    ann_vol = float(long_short.std() * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    sortino = sharpe * 1.2
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    return EvaluationResult(
        ic_series=ic_series,
        rank_ic_series=rank_ic_series,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        quantile_returns=quantile_returns,
        quantile_spread=quantile_spread,
        quantile_names=[f"Q{i+1}" for i in range(5)],
        long_short_returns=long_short,
        cumulative_returns=cumulative,
        drawdown_series=drawdown,
        max_drawdown=max_dd,
        drawdown_durations=np.array([5, 10, 3, 20, 8]),
        turnover_series=turnover,
        avg_turnover=float(turnover.mean()),
        coverage_series=coverage,
        avg_coverage=float(coverage.mean()),
        factor_correlation=factor_corr,
        factor_names=["factor_a", "factor_b"],
        skewness=-0.3,
        kurtosis=3.5,
        variance=float(long_short.var()),
        cvar=float(drawdown[drawdown < np.percentile(drawdown, 5)].mean()) if np.any(drawdown < np.percentile(drawdown, 5)) else 0.0,
        sharpe_ratio=sharpe,
        annual_return=ann_return,
        annual_volatility=ann_vol,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        hhi=hhi,
        ic_stability=ic_stability,
        rolling_ic_20=rolling_20,
        rolling_ic_60=rolling_60,
        rolling_ic_120=rolling_120,
        ic_decay=ic_decay,
        ic_autocorrelation=ic_autocorr,
        coverage_heatmap=coverage_heatmap,
        attribution={"momentum": 0.4, "value": 0.3, "quality": 0.2, "size": 0.1},
    )


def _make_library_results(n_factors: int = 8) -> list[dict]:
    """Build a list of factor result dicts for library report testing."""
    rng = np.random.default_rng(200)
    results = []
    for i in range(n_factors):
        ic_mean = rng.uniform(0.02, 0.08)
        ic_std = rng.uniform(0.015, 0.04)
        icir = ic_mean / ic_std
        sharpe = rng.uniform(0.5, 2.0)
        max_dd = rng.uniform(-0.25, -0.05)
        turnover = rng.uniform(0.1, 0.5)
        results.append({
            "factor_name": f"factor_{i}",
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "turnover": turnover,
        })
    return results


# ===========================================================================
# Test cases
# ===========================================================================

class TestCatalogDomains:
    """Verify all 10 QE-METRIC domains are registered."""

    def test_all_domains_exist(self):
        domains = list_all_domains()
        for d in EXPECTED_DOMAINS:
            assert d in domains, f"Domain {d.value!r} not found in catalog"

    def test_domain_count(self):
        domains = list_all_domains()
        assert len(domains) == 10, f"Expected 10 domains, got {len(domains)}"

    def test_domain_enum_values(self):
        expected_values = {
            "ic", "rank_ic", "quantile", "drawdown", "turnover",
            "tail_risk", "coverage", "hhi", "stability", "temporal",
        }
        actual_values = {d.value for d in Domain}
        assert actual_values == expected_values


class TestCatalogMetricSpecs:
    """Verify at least 25 of the 30 metric IDs are accessible."""

    def test_at_least_25_metric_ids_accessible(self):
        ids = list_all_metric_ids()
        accessible = 0
        inaccessible = []
        for mid in ids:
            try:
                spec = get_metric_spec(mid)
                assert spec.metric_id == mid
                accessible += 1
            except Exception as e:
                inaccessible.append((mid, str(e)))
        assert accessible >= 25, (
            f"Expected at least 25 accessible metric IDs, got {accessible}. "
            f"Inaccessible: {inaccessible}"
        )

    def test_all_30_ids_in_catalog(self):
        ids = list_all_metric_ids()
        for mid in ALL_30_METRIC_IDS:
            assert mid in ids, f"Metric ID {mid!r} not in catalog"

    def test_metric_spec_has_required_fields(self):
        spec = get_metric_spec("pearson_ic")
        assert spec.domain == Domain.IC
        assert isinstance(spec.description, str) and len(spec.description) > 0
        assert isinstance(spec.required_inputs, (set, frozenset))
        assert spec.output_type in ("scalar", "timeseries", "series")


class TestTearSheet:
    """Verify tear sheet generates exactly 22 panels with realistic data."""

    def test_tear_sheet_panel_count(self):
        result = _build_evaluation_result()
        panels = generate_tear_sheet(result)
        assert isinstance(panels, dict)
        assert len(panels) == TEAR_SHEET_PANEL_COUNT, (
            f"Expected {TEAR_SHEET_PANEL_COUNT} panels, got {len(panels)}"
        )

    def test_tear_sheet_panel_keys_are_strings(self):
        result = _build_evaluation_result()
        panels = generate_tear_sheet(result)
        for key, spec in panels.items():
            assert isinstance(key, str), f"Panel key {key!r} is not a string"
            assert isinstance(spec, ChartSpec), (
                f"Panel {key!r} is not a ChartSpec"
            )

    def test_tear_sheet_expected_panel_names(self):
        result = _build_evaluation_result()
        panels = generate_tear_sheet(result)
        expected_keys = {
            "ic_time_series", "ic_distribution", "rank_ic_time_series",
            "quantile_returns_bar", "quantile_spread_time_series",
            "drawdown_curve", "drawdown_duration_histogram",
            "turnover_time_series", "coverage_time_series",
            "factor_correlation_heatmap", "hhi_bar_chart",
            "ic_stability_scatter", "rolling_ic", "ic_decay_curve",
            "variance_cvar_bar", "skewness_kurtosis_bar",
            "summary_statistics_table", "risk_return_scatter",
            "coverage_heatmap", "turnover_histogram",
            "ic_autocorrelation", "performance_attribution_pie",
        }
        assert set(panels.keys()) == expected_keys, (
            f"Panel key mismatch.\nMissing: {expected_keys - set(panels.keys())}\n"
            f"Extra: {set(panels.keys()) - expected_keys}"
        )


class TestLibraryReport:
    """Test library report and comparison with A-share-like data."""

    def test_generate_library_report(self):
        results = _make_library_results(n_factors=8)
        report = generate_library_report(results, library_name="A-Share Alpha")
        assert report.library_name == "A-Share Alpha"
        assert report.n_factors == 8
        assert isinstance(report.summary_stats, dict)
        assert len(report.charts) > 0
        # Check summary stats contain expected keys
        for key in ("mean_ic_mean", "mean_sharpe", "mean_max_drawdown"):
            assert key in report.summary_stats, f"Missing stat key {key!r}"

    def test_library_report_stats_in_valid_range(self):
        results = _make_library_results(n_factors=10)
        report = generate_library_report(results, library_name="Test Lib")
        stats = report.summary_stats
        # IC means should be between 0 and 0.1 for A-share factors
        assert 0.0 <= stats["mean_ic_mean"] <= 0.1
        # Sharpe between 0 and 3
        assert 0.0 <= stats["mean_sharpe"] <= 3.0

    def test_compare_libraries(self):
        results_a = _make_library_results(n_factors=5)
        results_b = _make_library_results(n_factors=6)
        comp = compare_libraries(
            results_a, results_b,
            name_a="Momentum Library", name_b="Value Library",
        )
        assert comp.library_a_name == "Momentum Library"
        assert comp.library_b_name == "Value Library"
        assert isinstance(comp.comparison_stats, dict)
        assert len(comp.charts) > 0


class TestChartSpec:
    """Verify ChartSpec content_hash deduplication works."""

    def test_content_hash_deterministic(self):
        data = {"x": [1, 2, 3], "y": [4, 5, 6]}
        cs1 = ChartSpec(title="Test", x_label="X", y_label="Y",
                        chart_type="line", data=data)
        cs2 = ChartSpec(title="Test", x_label="X", y_label="Y",
                        chart_type="line", data=data)
        assert cs1.content_hash == cs2.content_hash, (
            "Identical data should produce identical content_hash"
        )

    def test_content_hash_differs_on_different_data(self):
        cs1 = ChartSpec(title="A", x_label="X", y_label="Y",
                        chart_type="bar", data={"v": [1, 2]})
        cs2 = ChartSpec(title="A", x_label="X", y_label="Y",
                        chart_type="bar", data={"v": [3, 4]})
        assert cs1.content_hash != cs2.content_hash, (
            "Different data should produce different content_hash"
        )

    def test_serialization_roundtrip(self):
        data = {"series": [0.01, 0.02, 0.03]}
        cs = ChartSpec(title="IC Trend", x_label="Date", y_label="IC",
                       chart_type="line", data=data)
        json_str = cs.to_json()
        cs2 = ChartSpec.from_json(json_str)
        assert cs2.title == cs.title
        assert cs2.data == cs.data
        assert cs2.content_hash == cs.content_hash

    def test_to_dict_roundtrip(self):
        cs = ChartSpec(title="X", x_label="X", y_label="Y",
                       chart_type="scatter", data={"a": 1})
        d = cs.to_dict()
        cs2 = ChartSpec.from_dict(d)
        assert cs2.chart_type == "scatter"
        assert cs2.data == {"a": 1}


class TestArtifactStore:
    """Test ArtifactStore save / load / delete cycle."""

    @pytest.fixture(autouse=True)
    def _tmp_dir(self, tmp_path):
        self.artifact_dir = str(tmp_path / "test_artifacts")

    def test_save_and_load(self):
        store = ArtifactStore(base_dir=self.artifact_dir)
        cs = ChartSpec(title="Test Chart", x_label="X", y_label="Y",
                       chart_type="line", data={"values": [1.0, 2.0, 3.0]})
        artifact_id = store.save(cs)
        assert isinstance(artifact_id, str) and len(artifact_id) > 0

        loaded = store.load(artifact_id)
        assert loaded is not None
        assert loaded.title == "Test Chart"
        assert loaded.data == {"values": [1.0, 2.0, 3.0]}

    def test_deduplication_by_content_hash(self):
        store = ArtifactStore(base_dir=self.artifact_dir)
        cs = ChartSpec(title="Dedup", x_label="X", y_label="Y",
                       chart_type="bar", data={"a": 100})
        id1 = store.save(cs)
        id2 = store.save(cs)  # same content_hash -> same id
        assert id1 == id2, "Duplicate content should return existing artifact_id"

    def test_delete_cycle(self):
        store = ArtifactStore(base_dir=self.artifact_dir)
        cs = ChartSpec(title="To Delete", x_label="X", y_label="Y",
                       chart_type="heatmap", data={"m": [[1, 2], [3, 4]]})
        aid = store.save(cs)
        assert store.exists(aid)

        deleted = store.delete_artifact(aid)
        assert deleted is True
        assert not store.exists(aid)
        assert store.load(aid) is None

    def test_list_artifacts(self):
        store = ArtifactStore(base_dir=self.artifact_dir)
        for i in range(3):
            store.save(ChartSpec(
                title=f"Chart {i}", x_label="X", y_label="Y",
                chart_type="line", data={"idx": i, "val": float(i * 10)},
            ))
        artifacts = store.list_artifacts()
        assert len(artifacts) == 3

    def test_clear(self):
        store = ArtifactStore(base_dir=self.artifact_dir)
        for i in range(5):
            store.save(ChartSpec(
                title=f"C{i}", x_label="X", y_label="Y",
                chart_type="bar", data={"i": i},
            ))
        count = store.clear()
        assert count == 5
        assert len(store.list_artifacts()) == 0


# ---------------------------------------------------------------------------
# Integration: end-to-end data flow
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Smoke test: catalog -> tear sheet -> chart -> artifact round-trip."""

    def test_full_flow(self):
        # 1. Catalog lookup
        spec = get_metric_spec("pearson_ic")
        assert spec.domain == Domain.IC

        # 2. Build realistic result
        eval_result = _build_evaluation_result()

        # 3. Generate tear sheet
        panels = generate_tear_sheet(eval_result)
        assert len(panels) == TEAR_SHEET_PANEL_COUNT

        # 4. Save each panel to artifact store
        tmp_dir = tempfile.mkdtemp(prefix="qe_acceptance_")
        try:
            store = ArtifactStore(base_dir=tmp_dir)
            saved_ids = []
            for name, chart in panels.items():
                aid = store.save(chart, artifact_id=name)
                saved_ids.append(aid)
            assert len(saved_ids) == TEAR_SHEET_PANEL_COUNT

            # 5. Load and verify content hash consistency
            for name, chart in panels.items():
                loaded = store.load(name)
                assert loaded is not None
                assert loaded.content_hash == chart.content_hash
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
