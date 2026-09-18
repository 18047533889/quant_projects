"""R49 numerical repairs cannot reuse old semantic identities."""
import pytest
from factor_engine.backend import operator_semantic_version as versions
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.factor_identity import OperatorSemanticContractDigest

@pytest.mark.parametrize("canonical,current", [
    ("index_reconstitution_churn", 2),
    ("intra_max_drawdown", 2),
    ("intra_max_drawup", 2),
    ("intra_segment_realized_vol", 2),
    ("intra_positive_jump_variation", 2),
    ("intra_negative_jump_variation", 2),
    ("intra_signed_jump_ratio", 2),
    ("intra_jump_count", 2),
    ("intra_jump_concentration", 2),
    ("intra_jump_first_time", 2),
    ("intra_jump_last_time", 2),
    ("intra_jump_clustering", 2),
    ("intra_positive_tail_variation", 2),
    ("intra_negative_tail_variation", 2),
    ("intra_tail_event_count", 2),
    ("intra_signed_tail_variation_ratio", 2),
    ("intra_limit_duration", 3),
    ("intra_limit_first_hit_time", 3),
    ("intra_limit_reopen_count", 3),
    ("holder_freeze_ratio", 2),
    ("holder_locked_share_ratio", 2),
    ("holder_pledge_ratio", 2),
    ("holder_class_entropy", 2),
    ("holder_nature_entropy", 2),
    ("holder_class_js_shift", 2),
    ("holder_common_holding_peer_return", 2),
    ("holder_concentration_acceleration", 2),
    ("holder_concentration_slope", 2),
    ("holder_freeze_concentration", 2),
    ("holder_pledge_churn", 2),
    ("holder_pledge_concentration", 2),
    ("holder_pledged_holder_count", 2),
    ("holder_shareholder_network_centrality", 2),
    ("holder_concentration", 2),
    ("holder_topk_share_sum", 2),
    ("holder_company_ownership_hhi", 3),
    ("holder_observed_topk_hhi", 2),
    ("holder_disclosure_count", 2),
    ("holder_disclosure_coverage", 2),
    ("holder_entry_share", 2),
    ("holder_exit_share", 2),
    ("holder_net_entry_share", 2),
    ("holder_id_matched_entry_share", 2),
    ("holder_id_matched_exit_share", 2),
    ("holder_id_matched_churn", 2),
    ("holder_weighted_churn", 2),
    ("holder_rank_stability", 2),
    ("holder_id_overlap_ratio", 2),
    ("holder_share_weighted_rank_migration", 2),
    ("holder_shareholder_overlap_ratio", 2),
    ("holder_float_concentration_gap", 2),
])
def test_repaired_identities_change(monkeypatch, canonical, current):
    load_all()
    assert versions.semantic_version(canonical) == current
    repaired = OperatorSemanticContractDigest.for_canonical(canonical)
    assert repaired.semantic_version == f"{current}.0"
    monkeypatch.setitem(versions.OPERATOR_SEMANTIC_VERSIONS, canonical, current-1)
    old_row = {**OperatorRegistry._catalog[canonical], "semantic_version": f"{current-1}.0"}
    monkeypatch.setattr(OperatorRegistry, "_catalog", {**OperatorRegistry._catalog, canonical: old_row})
    previous = OperatorSemanticContractDigest.for_canonical(canonical)
    assert previous.contract_hash() != repaired.contract_hash()
