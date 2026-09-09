from jobs.e2e_d_risk_shift import run_e2e_d_risk_shift


def _exposure(result, candidate):
    bundle = result.evaluations[candidate]
    return sum(abs(bundle.get_metric(mid, candidate).value)
               for mid in ("industry_exposure", "size_exposure"))


def test_e2e_d_selects_noninferior_real_exposure_improvement():
    result = run_e2e_d_risk_shift(utility_damage=False)
    assert result.raw_dsl == "close"
    assert result.snapshot_ref.startswith("synthetic-snapshot:")
    assert result.catalog_ref.startswith("semantic-catalog:")
    assert result.winner_id in {"industry_neutral", "size_neutral"}
    assert not result.raw_preserved
    assert not result.noninferiority_conflicts
    assert _exposure(result, result.winner_id) < _exposure(result, "raw")
    assert result.trace["verdict"] == "NEUTRALIZATION_SELECTED"
    assert result.trace["factor_definition"].startswith("factor-definition:")
    assert result.trace["recipe"]
    assert result.trace["evaluation"] == result.evaluations[result.winner_id].request_id
    assert result.trace["parent_trial"] is None
    assert result.trace["health_policy"] is None
    assert result.trace["library"] is None
    assert result.trace["feature_version"] is None


def test_e2e_d_signal_damage_cannot_overwrite_raw():
    result = run_e2e_d_risk_shift(utility_damage=True)
    assert set(result.noninferiority_conflicts) == {"industry_neutral", "size_neutral"}
    assert result.winner_id == "raw"
    assert result.raw_preserved
    assert result.trace["verdict"] == "RAW_PRESERVED"
    assert result.trace["recipe"] is None
    raw_ic = result.evaluations["raw"].get_metric("rank_ic", "raw").value
    assert all(result.evaluations[c].get_metric("rank_ic", c).value < .9 * raw_ic
               for c in result.noninferiority_conflicts)
