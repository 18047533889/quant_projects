"""Holder identifiers/weights retain real contracts without lifting data gates."""
import numpy as np
import pandas as pd
import pytest
from factor_engine.fields import FIELD_REGISTRY
from factor_engine.fields.spec import semantic_kind_of_field
from factor_engine.api.columns import field

@pytest.mark.parametrize("table",["StockTopTenShareholder","StockTopTenFloatShareholder"])
def test_holder_catalog_semantics_preserve_mining_and_scaling(table):
    ratio=FIELD_REGISTRY.require("ShareRatio",table=table)
    identity=FIELD_REGISTRY.require("ShareholderId",table=table)
    assert semantic_kind_of_field(ratio,production=True)=="NonNegativeWeight"
    assert semantic_kind_of_field(identity,production=True)=="GroupKey"
    assert ratio.scale_to_canonical==0.01
    assert ratio.cardinality==identity.cardinality=="one_to_many"
    assert not ratio.mining_allowed and not identity.mining_allowed
    with pytest.raises(ValueError,match="mining_allowed=False"):
        field("ShareRatio",table=table,for_mining=True)

def test_share_and_peer_actual_kernels_match_independent_calculation():
    from factor_engine.cleaned_operators.relation.ops import RelationCategoryShare,RelationPeerWeightedMeanExSelf
    value=pd.DataFrame([[.7,.4,.6,.5],[.7,np.nan,.6,.5]])
    weight=pd.DataFrame([[.4,.2,.3,.1],[.4,.2,.3,.1]])
    group=pd.DataFrame([[101,101,101,202]]*2)
    share=RelationCategoryShare()._calculate_series(value,group)
    expected=np.array([[.7/1.7,.4/1.7,.6/1.7,1.],[.7/1.3,np.nan,.6/1.3,1.]])
    np.testing.assert_allclose(share,expected,equal_nan=True)
    peer=RelationPeerWeightedMeanExSelf()._calculate_series(value,weight,group)
    independent=np.full((2,4),np.nan)
    for row in range(2):
        for col in range(4):
            if not np.isfinite(value.iloc[row,col]):continue
            others=[j for j in range(4) if j!=col and group.iloc[row,j]==group.iloc[row,col] and np.isfinite(value.iloc[row,j]) and weight.iloc[row,j]>0]
            if others:independent[row,col]=np.average(value.iloc[row,others],weights=weight.iloc[row,others])
    np.testing.assert_allclose(peer,independent,equal_nan=True)
    with pytest.raises(ValueError,match="NON-NEGATIVE"):
        RelationCategoryShare()._calculate_series(-value,group)

def test_rank_band_js_matches_scipy_and_preserves_missingness():
    from scipy.spatial.distance import jensenshannon
    from factor_engine.cleaned_operators.advanced_structure import HolderClassJsShift
    cur=np.array([.3,.2,.15,.1,.05]);prev=np.array([.2,.2,.2,.1,.1])
    panels=[pd.DataFrame([[x,x]]) for x in np.r_[cur,prev]]
    panels[2].iloc[0,1]=np.nan
    actual=HolderClassJsShift()._calculate_series(*panels)
    assert actual.iloc[0,0]==pytest.approx(jensenshannon(cur,prev,base=2))
    assert np.isnan(actual.iloc[0,1])
