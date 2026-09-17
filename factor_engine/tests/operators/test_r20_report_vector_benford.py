import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import jensenshannon
from factor_engine.cleaned_operators.report_vector_benford import ReportVectorBenfordJsDivergence

def test_same_report_vector_matches_scipy_not_time_history():
    # Stay away from decimal boundaries: scaled binary floats just below 3eN
    # genuinely have leading digit 2, so a boundary-based scale test is invalid.
    values=[1.25,22.5,325.,4250.,52500.,625000.]
    frames=[pd.DataFrame({"A":[x,x*10,-x],"B":[x*1e-300,x,0]}) for x in values]
    actual=ReportVectorBenfordJsDivergence()._calculate_series(*frames)
    p=np.array([1/6]*6+[0]*3)
    expected=jensenshannon(p,np.log10(1+1/np.arange(1.,10.)),base=2)
    np.testing.assert_allclose(actual["A"],expected,rtol=1e-13)
    np.testing.assert_allclose(actual["B"][:2],expected,rtol=1e-13)
    assert np.isnan(actual["B"].iloc[2])

def test_missing_and_identity_rejected():
    frames=[pd.DataFrame({"A":[float(i),np.nan]}) for i in range(1,7)]
    out=ReportVectorBenfordJsDivergence()._calculate_series(*frames)
    assert np.isfinite(out.iloc[0,0]) and np.isnan(out.iloc[1,0])
    frames[-1].index=[1,2]
    with pytest.raises(ValueError,match="identity"):
        ReportVectorBenfordJsDivergence()._calculate_series(*frames)
