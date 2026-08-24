from research_platform.firewall import (
    AlphaGenerationFirewall, FactorCandidateArtifact, FirewallStage,
    FirewallResult, _ValidationContext,
)


def _cand(**kw):
    kw.setdefault("artifact_id", "c1")
    kw.setdefault("artifact_type", "factor_candidate")
    kw.setdefault("candidate_id", "c1")
    kw.setdefault("generator_type", "llm")
    kw.setdefault("formula", "f_close / f_open")
    kw.setdefault("hypothesis", "test hypothesis")
    return FactorCandidateArtifact(**kw)


def test_full_pass_when_nothing_known():
    fw = AlphaGenerationFirewall(
        known_formulas=None, known_fingerprints=None,
        known_result_identities=None, available_fields=None,
    )
    res = fw.check(_cand())
    assert len(res) == 10
    assert all(r.passed for r in res)
    report = fw.check_and_report(_cand())
    assert report["passed"] is True


def test_exact_duplicate_rejected():
    fw = AlphaGenerationFirewall(known_formulas={"f_close / f_open"})
    res = fw.check(_cand())
    dup = [r for r in res if r.stage == FirewallStage.FORMULA_EXACT_DUPLICATE]
    assert dup and dup[0].passed is False
    assert fw.check_and_report(_cand())["passed"] is False


def test_unbalanced_parens_rejected():
    fw = AlphaGenerationFirewall()
    res = fw.check(_cand(formula="(f_close * f_open"))
    g = [r for r in res if r.stage == FirewallStage.GRAMMAR_LEGALITY][0]
    assert g.passed is False


def test_fe_compilation_injected():
    def checker(formula):
        return formula.startswith("ok_")
    fw = AlphaGenerationFirewall(compile_checker=checker)
    res = fw.check(_cand())
    c = [r for r in res if r.stage == FirewallStage.FE_COMPILATION][0]
    assert c.passed is False
    res2 = fw.check(_cand(formula="ok_f_close"))
    c2 = [r for r in res2 if r.stage == FirewallStage.FE_COMPILATION][0]
    assert c2.passed is True


def test_candidate_does_not_require_cot_field():
    c = _cand()
    # coerce to dict; must not contain chain-of-thought
    d = c.to_dict()
    assert "chain_of_thought" not in d
    assert "reasoning_trace" not in d


def test_known_duplicate_with_known_set():
    fw = AlphaGenerationFirewall(known_formulas={"f_close / f_open"})
    report = fw.check_and_report(_cand())
    assert report["passed"] is False
    assert FirewallStage.FORMULA_EXACT_DUPLICATE.value in report["failed_stages"]
