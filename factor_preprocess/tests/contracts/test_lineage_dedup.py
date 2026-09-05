"""
R61-FI-043 — treatment lineage duplicate-guard tests (plan §26 F4).

Covers every redundancy class (rank(rank(x)), zscore(zscore(x)),
winsor(winsor(x)), identical EWMA twice, size-neutral(size-neutral(x))),
every legitimate superset collapse (industry neutral -> industry+size
neutral, both orders, and the explicit dual step remaining single) and the
versioned idempotent-table policy.
"""
import pytest

from factor_preprocess.contracts.treatment_lineage import (
    TransformStage,
    TransformSemanticID,
    TransformStep,
    TransformLineage,
)
from factor_preprocess.contracts.lineage_policy import (
    LINEAGE_POLICY_VERSION,
    IDEMPOTENT_SEMANTIC_CLASSES,
    NEUTRALIZATION_TOPOLOGY,
    RedundancyClass,
    RedundantTransformError,
    UnsupportedPolicyVersionError,
    LineagePolicyDecision,
    canonicalize_lineage,
    is_losslessly_collapsible,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _step(sid, stage=TransformStage.REPRESENTATION, name=None, params=None):
    return TransformStep(
        semantic_id=TransformSemanticID(sid),
        stage=stage,
        name=name or sid.split(":")[0].lower(),
        parameters=dict(params or {}),
    )


def _sids(lineage_or_decision):
    lineage = (
        lineage_or_decision.lineage
        if isinstance(lineage_or_decision, LineagePolicyDecision)
        else lineage_or_decision
    )
    return [s.semantic_id.value for s in lineage.steps]


# ---------------------------------------------------------------------------
# 1. redundant operations — canonical fold (lossless)
# ---------------------------------------------------------------------------


def test_rank_of_rank_collapses_to_single_step():
    lineage = TransformLineage((_step("CS_RANK:pct"), _step("CS_RANK:pct")))
    decision = canonicalize_lineage(lineage)
    assert decision.was_collapsed
    assert decision.is_valid
    assert _sids(decision) == ["CS_RANK:pct"]
    assert len(decision.lineage) == 1


def test_zscore_of_zscore_collapses_to_single_step():
    lineage = TransformLineage(
        (
            _step("CROSS_SECTIONAL_ZSCORE:cs"),
            _step("CROSS_SECTIONAL_ZSCORE:cs"),
        )
    )
    decision = canonicalize_lineage(lineage)
    assert decision.was_collapsed
    assert _sids(decision) == ["CROSS_SECTIONAL_ZSCORE:cs"]
    assert len(decision.lineage) == 1


def test_zscore_alias_forms_collapse():
    # cs_zscore (ZSCORE:cs alias) twice must fold like the canonical form.
    lineage = TransformLineage((_step("ZSCORE:cs"), _step("ZSCORE:cs")))
    decision = canonicalize_lineage(lineage)
    assert decision.was_collapsed
    assert _sids(decision) == ["ZSCORE:cs"]


def test_winsor_of_winsor_collapses_to_single_step():
    lineage = TransformLineage(
        (
            _step("WINSOR:cs", TransformStage.OUTLIER, params={"lower": 0.01, "upper": 0.99}),
            _step("WINSOR:cs", TransformStage.OUTLIER, params={"lower": 0.01, "upper": 0.99}),
        )
    )
    decision = canonicalize_lineage(lineage)
    assert decision.was_collapsed
    assert _sids(decision) == ["WINSOR:cs"]
    assert len(decision.lineage) == 1


def test_lossless_collapse_predicate_rank():
    ok, folded = is_losslessly_collapsible(_step("CS_RANK:pct"), _step("CS_RANK:pct"))
    assert ok
    assert folded == {}


def test_lossless_collapse_predicate_winsor_equal_bounds():
    ok, _ = is_losslessly_collapsible(
        _step("WINSOR:cs", TransformStage.OUTLIER, params={"lower": 0.01, "upper": 0.99}),
        _step("WINSOR:cs", TransformStage.OUTLIER, params={"lower": 0.01, "upper": 0.99}),
    )
    assert ok


# ---------------------------------------------------------------------------
# 2. redundant operations — rejected (no lossless fold)
# ---------------------------------------------------------------------------


def test_identical_ewma_twice_rejected():
    lineage = TransformLineage(
        (
            _step("SMOOTH:ewma", TransformStage.TEMPORAL, params={"halflife": 20}),
            _step("SMOOTH:ewma", TransformStage.TEMPORAL, params={"halflife": 20}),
        )
    )
    with pytest.raises(RedundantTransformError):
        canonicalize_lineage(lineage)


def test_ewma_twice_distinct_windows_rejected():
    lineage = TransformLineage(
        (
            _step("SMOOTH:ewma", TransformStage.TEMPORAL, params={"halflife": 10}),
            _step("SMOOTH:ewma", TransformStage.TEMPORAL, params={"halflife": 30}),
        )
    )
    with pytest.raises(RedundantTransformError):
        canonicalize_lineage(lineage)


def test_cross_smoother_cascade_rejected():
    # kama after ewma is a different-family cascade that no single step
    # reproduces -> rejected (conservative guard).
    lineage = TransformLineage(
        (
            _step("SMOOTH:ewma", TransformStage.TEMPORAL, params={"halflife": 20}),
            _step("SMOOTH:kama", TransformStage.TEMPORAL),
        )
    )
    with pytest.raises(RedundantTransformError):
        canonicalize_lineage(lineage)


def test_size_neutral_of_size_neutral_rejected():
    lineage = TransformLineage(
        (
            _step("SIZE_NEUTRAL:log_mktcap", TransformStage.NEUTRALIZATION),
            _step("SIZE_NEUTRAL:log_mktcap", TransformStage.NEUTRALIZATION),
        )
    )
    with pytest.raises(RedundantTransformError):
        canonicalize_lineage(lineage)


def test_dual_after_industry_rejected_as_subset_redundancy():
    # industry neutral is a subset of industry+size dual; the sequential form
    # must be collapsed by the caller, never left as two passes.
    lineage = TransformLineage(
        (
            _step("INDUSTRY_NEUTRAL:SW_L1", TransformStage.NEUTRALIZATION),
            _step("DUAL_NEUTRAL:industry_size", TransformStage.NEUTRALIZATION),
        )
    )
    with pytest.raises(RedundantTransformError):
        canonicalize_lineage(lineage)


def test_rank_then_zscore_same_axis_rejected():
    lineage = TransformLineage((_step("CS_RANK:pct"), _step("CROSS_SECTIONAL_ZSCORE:cs")))
    with pytest.raises(RedundantTransformError):
        canonicalize_lineage(lineage)


def test_ols_neutralize_twice_rejected():
    lineage = TransformLineage(
        (
            _step("NEUTRAL:ols", TransformStage.NEUTRALIZATION),
            _step("NEUTRAL:ols", TransformStage.NEUTRALIZATION),
        )
    )
    with pytest.raises(RedundantTransformError):
        canonicalize_lineage(lineage)


# ---------------------------------------------------------------------------
# 3. legitimate superset collapse
# ---------------------------------------------------------------------------


def test_industry_then_size_collapses_to_single_dual_step():
    lineage = TransformLineage(
        (
            _step("INDUSTRY_NEUTRAL:SW_L1", TransformStage.NEUTRALIZATION),
            _step("SIZE_NEUTRAL:log_mktcap", TransformStage.NEUTRALIZATION),
        )
    )
    decision = canonicalize_lineage(lineage)
    assert decision.was_collapsed
    assert _sids(decision) == ["DUAL_NEUTRAL:industry_size"]
    assert len(decision.lineage) == 1


def test_size_then_industry_collapses_to_single_dual_step():
    lineage = TransformLineage(
        (
            _step("SIZE_NEUTRAL:log_mktcap", TransformStage.NEUTRALIZATION),
            _step("INDUSTRY_NEUTRAL:SW_L1", TransformStage.NEUTRALIZATION),
        )
    )
    decision = canonicalize_lineage(lineage)
    assert decision.was_collapsed
    assert _sids(decision) == ["DUAL_NEUTRAL:industry_size"]


def test_superset_collapse_leaves_explicit_dual_single():
    # A single explicit dual step must pass through unchanged (no double
    # application, no erroneous second fold).
    lineage = TransformLineage(
        (_step("DUAL_NEUTRAL:industry_size", TransformStage.NEUTRALIZATION),)
    )
    decision = canonicalize_lineage(lineage)
    assert decision.redundancy_class is RedundancyClass.OK
    assert _sids(decision) == ["DUAL_NEUTRAL:industry_size"]


def test_superset_collapse_predicate():
    ok, folded = is_losslessly_collapsible(
        _step("INDUSTRY_NEUTRAL:SW_L1", TransformStage.NEUTRALIZATION),
        _step("SIZE_NEUTRAL:log_mktcap", TransformStage.NEUTRALIZATION),
    )
    assert ok
    assert folded["canonical_sid"] == "DUAL_NEUTRAL:industry_size"
    ok_rev, _ = is_losslessly_collapsible(
        _step("SIZE_NEUTRAL:log_mktcap", TransformStage.NEUTRALIZATION),
        _step("INDUSTRY_NEUTRAL:SW_L1", TransformStage.NEUTRALIZATION),
    )
    assert ok_rev


# ---------------------------------------------------------------------------
# 4. distinct orders remain distinct; stage semantics preserved
# ---------------------------------------------------------------------------


def test_rank_pre_and_post_neutralization_stay_distinct():
    lineage = TransformLineage(
        (
            _step("CS_RANK:pct", TransformStage.PRE_NEUTRALIZATION),
            _step("INDUSTRY_NEUTRAL:SW_L1", TransformStage.NEUTRALIZATION),
            _step("CS_RANK:pct", TransformStage.POST_NEUTRALIZATION),
        )
    )
    decision = canonicalize_lineage(lineage)
    assert decision.redundancy_class is RedundancyClass.OK
    assert _sids(decision) == [
        "CS_RANK:pct",
        "INDUSTRY_NEUTRAL:SW_L1",
        "CS_RANK:pct",
    ]


def test_distinct_orders_remain_distinct_semantics():
    # zscore->rank and rank->zscore over the SAME adjacent axis are both
    # redundant double standardizations -> both rejected. The DISTINCT-order
    # preservation guarantee applies when a neutralization separates the two
    # passes (covered by test_rank_pre_and_post_neutralization_stay_distinct).
    for ordered in (
        (_step("CROSS_SECTIONAL_ZSCORE:cs"), _step("CS_RANK:pct")),
        (_step("CS_RANK:pct"), _step("CROSS_SECTIONAL_ZSCORE:cs")),
    ):
        with pytest.raises(RedundantTransformError):
            canonicalize_lineage(TransformLineage(ordered))


def test_winsor_rank_valid_single_pass():
    # winsor followed by rank is the canonical outlier->representation order
    # and must remain valid (not rejected).
    lineage = TransformLineage(
        (
            _step("WINSOR:cs", TransformStage.OUTLIER, params={"lower": 0.01, "upper": 0.99}),
            _step("CS_RANK:pct", TransformStage.REPRESENTATION),
        )
    )
    decision = canonicalize_lineage(lineage)
    assert decision.redundancy_class is RedundancyClass.OK
    assert _sids(decision) == ["WINSOR:cs", "CS_RANK:pct"]


# ---------------------------------------------------------------------------
# 5. idempotent-table versioning
# ---------------------------------------------------------------------------


def test_policy_version_constant_exports():
    assert LINEAGE_POLICY_VERSION
    assert "CS_RANK:pct" in IDEMPOTENT_SEMANTIC_CLASSES
    assert (
        NEUTRALIZATION_TOPOLOGY["DUAL_NEUTRAL:industry_size"]
        == ("industry", "size")
    )


def test_unsupported_policy_version_fails_closed():
    with pytest.raises(UnsupportedPolicyVersionError):
        canonicalize_lineage(
            TransformLineage((_step("CS_RANK:pct"),)),
            policy_version="1999-01-01.0",
        )


def test_canonicalization_records_declared_policy_version():
    decision = canonicalize_lineage(
        TransformLineage((_step("CS_RANK:pct"), _step("CS_RANK:pct")))
    )
    assert decision.policy_version == LINEAGE_POLICY_VERSION


def test_non_redundant_lineage_reports_ok():
    lineage = TransformLineage(
        (
            _step("FILL:forward", TransformStage.MISSINGNESS),
            _step("WINSOR:cs", TransformStage.OUTLIER),
            _step("SMOOTH:ewma", TransformStage.TEMPORAL, params={"halflife": 20}),
            _step("CS_RANK:pct", TransformStage.REPRESENTATION),
        )
    )
    decision = canonicalize_lineage(lineage)
    assert decision.is_valid
    assert decision.redundancy_class is RedundancyClass.OK
    assert not decision.was_collapsed
    assert len(decision.rejections) == 0
