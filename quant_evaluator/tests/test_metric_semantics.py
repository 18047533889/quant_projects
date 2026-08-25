"""
QE-P1-V: MetricSpec semantic correctness regression guard.

Pins that the authoritative catalog (``quant_evaluator.metrics.catalog``)
carries REAL per-metric semantics — direction, units, implementation identity
— rather than silently falling back to defaults.  A default-slipping regression
(e.g. ``max_drawdown`` reverting to ``higher_is_better``) is caught here.

Run with:

    PYTHONPATH=/home/sunhaiwei/quant_projects \\
    python -m pytest quant_evaluator/tests/test_metric_semantics.py -q --tb=short
"""

import pytest

from quant_evaluator.metrics import catalog as _catalog_mod


@pytest.fixture(autouse=True)
def _fresh_catalog():
    """Re-fetch catalog classes fresh.

    QE-P0-01: metrics.catalog re-exports Domain/MetricSpec from
    registry.metrics (single authority).  Other test modules reload the
    registry to restore BUILDING state, recreating the classes; this fixture
    re-imports the module so the names below always point at the CURRENT
    class objects.
    """
    import importlib

    importlib.reload(_catalog_mod)
    globals()["Domain"] = _catalog_mod.Domain
    globals()["get_metric_spec"] = _catalog_mod.get_metric_spec
    globals()["list_all_metric_ids"] = _catalog_mod.list_all_metric_ids
    yield


Domain = _catalog_mod.Domain
get_metric_spec = _catalog_mod.get_metric_spec
list_all_metric_ids = _catalog_mod.list_all_metric_ids


# ---------------------------------------------------------------------------
# Direction semantics for the KEY metrics the audit flagged.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric_id,expected_direction",
    [
        # Drawdown: a LARGER drawdown is WORSE.
        ("max_drawdown", "lower_is_better"),
        ("drawdown_duration", "lower_is_better"),
        # Turnover / cost: more churn / cost is WORSE.
        ("turnover_rate", "lower_is_better"),
        ("turnover_cost", "lower_is_better"),
        # Tail risk: a larger VaR/CVaR loss is WORSE.
        ("var_95", "lower_is_better"),
        ("var_99", "lower_is_better"),
        ("cvar_95", "lower_is_better"),
        ("cvar_99", "lower_is_better"),
        # Concentration: a more concentrated factor is WORSE.
        ("hhi_concentration", "lower_is_better"),
        # Predictive power: higher IC is BETTER.
        ("pearson_ic", "higher_is_better"),
        ("spearman_ic", "higher_is_better"),
        ("rank_ic", "higher_is_better"),
        ("quantile_spread", "higher_is_better"),
        ("calmar_ratio", "higher_is_better"),
        # Coverage: more data is BETTER.
        ("factor_coverage", "higher_is_better"),
        ("return_coverage", "higher_is_better"),
        ("joint_coverage", "higher_is_better"),
        # Stability: more stable is BETTER.
        ("ic_stability", "higher_is_better"),
        # Neutral: no monotone preference.
        ("skewness", "neutral"),
        ("kurtosis", "neutral"),
        ("ic_decay", "neutral"),
        ("autocorrelation_ic", "neutral"),
    ],
)
def test_key_metric_direction_semantics(metric_id, expected_direction):
    spec = get_metric_spec(metric_id)
    assert spec.direction == expected_direction, (
        f"{metric_id} direction should be {expected_direction!r}, "
        f"got {spec.direction!r}"
    )


# ---------------------------------------------------------------------------
# Units semantics.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric_id,expected_units",
    [
        ("max_drawdown", "fraction"),
        ("turnover_rate", "fraction"),
        ("turnover_cost", "bps"),
        ("var_95", "fraction"),
        ("cvar_99", "fraction"),
        ("pearson_ic", "correlation"),
        ("rank_ic", "correlation"),
        ("calmar_ratio", "ratio"),
        ("drawdown_duration", "periods"),
        ("hhi_concentration", "dimensionless"),
        ("skewness", "dimensionless"),
        ("quantile_returns", "return"),
    ],
)
def test_key_metric_units_semantics(metric_id, expected_units):
    spec = get_metric_spec(metric_id)
    assert spec.units == expected_units, (
        f"{metric_id} units should be {expected_units!r}, got {spec.units!r}"
    )


# ---------------------------------------------------------------------------
# Real implementation identity (not metric_id masquerading as an impl).
# ---------------------------------------------------------------------------


def test_implementation_id_is_a_real_module_function_path():
    for metric_id in list_all_metric_ids():
        spec = get_metric_spec(metric_id)
        impl = spec.implementation_id
        # A real implementation identity is a dotted module.function path.
        assert impl.count(".") >= 2, (
            f"{metric_id} implementation_id {impl!r} is not a real "
            "module.function path"
        )
        assert impl != metric_id, (
            f"{metric_id} implementation_id must not be the metric_id itself"
        )


def test_implementation_hash_is_stable_and_content_derived():
    # Same kernel -> same hash (pearson_ic / spearman_ic / rank_ic all use
    # compute_daily_ic).
    p = get_metric_spec("pearson_ic")
    s = get_metric_spec("spearman_ic")
    r = get_metric_spec("rank_ic")
    assert p.implementation_hash == s.implementation_hash == r.implementation_hash
    # Distinct kernels -> distinct hashes.
    assert p.implementation_hash != get_metric_spec("max_drawdown").implementation_hash
    # Hash is a stable 16-hex content fingerprint.
    assert len(p.implementation_hash) == 16
    int(p.implementation_hash, 16)  # must be valid hex


def test_metric_version_is_real_semver():
    for metric_id in list_all_metric_ids():
        spec = get_metric_spec(metric_id)
        parts = spec.metric_version.split(".")
        assert len(parts) == 3, (
            f"{metric_id} metric_version {spec.metric_version!r} is not semver"
        )
        for part in parts:
            assert part.isdigit(), (
                f"{metric_id} metric_version {spec.metric_version!r} is not semver"
            )


# ---------------------------------------------------------------------------
# No metric may be left at all-default semantics.
# ---------------------------------------------------------------------------


def test_no_metric_left_at_all_default_semantics():
    for metric_id in list_all_metric_ids():
        spec = get_metric_spec(metric_id)
        # Every registered metric must carry a real implementation identity
        # and a real semver (the two fields that were previously defaulted).
        assert spec.implementation_id != metric_id
        assert spec.metric_version != "0.1.0"
        # Direction must be explicitly meaningful (never the bare default
        # higher_is_better for a metric that is not a "more is better" metric).
        assert spec.direction in ("higher_is_better", "lower_is_better", "neutral")


def test_all_domains_present():
    from quant_evaluator.metrics.catalog import list_all_domains

    domains = {d for d in list_all_domains()}
    assert domains == set(Domain)
