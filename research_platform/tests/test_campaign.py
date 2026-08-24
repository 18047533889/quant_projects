from research_platform.campaign import ResearchCampaignArtifact, TrialLedger


def test_campaign_admission_rate():
    c = ResearchCampaignArtifact(
        artifact_id="c1", artifact_type="research_campaign",
        campaign_id="c1", candidate_count=4,
        admitted_trial_ids=("t1",),
        all_trial_ids=("t1", "t2", "t3", "t4"),
    )
    assert c.admission_rate == 0.25


def test_ledger_counts():
    ledger = TrialLedger()
    ledger.append("t1", "llm", True)
    ledger.append("t2", "llm", False)
    ledger.append("t3", "mutation", False)
    assert ledger.multiple_testing_count == 3
    assert ledger.admission_rate == 1 / 3
    bd = ledger.generator_breakdown()
    assert bd["llm"] == {"trials": 2, "admitted": 1}
    assert bd["mutation"] == {"trials": 1, "admitted": 0}


def test_ledger_duplicate_rejected():
    ledger = TrialLedger()
    ledger.append("t1", "llm", True)
    try:
        ledger.append("t1", "llm", False)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_export_campaign_has_hash():
    ledger = TrialLedger()
    ledger.append("t1", "llm", True)
    ledger.append("t2", "llm", False)
    art = ledger.export_campaign("c-1")
    assert art.candidate_count == 2
    assert art.admitted_trial_ids == ("t1",)
    assert art.content_hash
