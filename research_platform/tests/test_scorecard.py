from research_platform.scorecard import GeneratorScorecard
from research_platform.firewall import FirewallResult, FirewallStage


def _result(stage, passed):
    return FirewallResult(stage=stage, passed=passed)


def test_summary_rates():
    sc = GeneratorScorecard(generator_type="llm")
    sc.record(results=[_result(FirewallStage.GRAMMAR_LEGALITY, True),
                       _result(FirewallStage.FE_COMPILATION, True)],
               passed=True, admitted=True, novelty=0.8,
               incremental_ic=0.05, compute_cost=10.0,
               wall_clock_ms=100.0, api_token_cost=5.0, family="momentum")
    sc.record(results=[_result(FirewallStage.GRAMMAR_LEGALITY, False),
                       _result(FirewallStage.FE_COMPILATION, False)],
               passed=False, admitted=False, family="momentum")
    s = sc.summary()
    assert s["proposal_count"] == 2
    assert s["legality_pass_rate"] == 0.5
    assert s["compile_pass_rate"] == 0.5
    assert s["final_admission_rate"] == 0.5
    assert s["family_diversity"] == 1
    assert s["mean_novelty"] == 0.8


def test_duplicate_counts_in_summary():
    sc = GeneratorScorecard()
    sc.exact_duplicate_count = 1
    sc.behavior_duplicate_count = 2
    sc.proposal_count = 4
    sc._stage_total["grammar_legality"] = 4
    sc._stage_pass["grammar_legality"] = 3
    s = sc.summary()
    assert s["exact_duplicate_rate"] == 0.25
    assert s["behavior_duplicate_rate"] == 0.5
    assert s["legality_pass_rate"] == 0.75


def test_empty_summary_does_not_divide_by_zero():
    sc = GeneratorScorecard()
    s = sc.summary()
    assert s["proposal_count"] == 0
    assert s["final_admission_rate"] == 0.0
    assert s["mean_novelty"] == 0.0


def test_l0_l3_pass_rate_shape():
    sc = GeneratorScorecard()
    sc.record(stage_names=["l0"], passed=True)
    sc.record(stage_names=["l1", "l2"], passed=True)
    s = sc.summary()
    assert set(s["l0_l3_pass_rate"].keys()) == {"l0", "l1", "l2", "l3"}
