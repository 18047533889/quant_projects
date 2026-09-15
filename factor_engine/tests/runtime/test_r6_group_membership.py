"""Missing group labels must never form artificial peer groups."""
import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators.group_ext import GroupExSelfMean, GroupExSelfWeightedMean

@pytest.mark.parametrize("missing", ["", None, np.inf, -np.inf, pd.NA])
@pytest.mark.parametrize("weighted", [False, True])
def test_missing_memberships_are_not_a_group(missing, weighted):
    x = pd.DataFrame([[1., 3., 10., 20.]], columns=list("abcd"))
    groups = pd.DataFrame([[missing, missing, "A", "A"]], columns=x.columns)
    if weighted:
        result = GroupExSelfWeightedMean().calculate(x, x * 0 + 1, groups)
    else:
        result = GroupExSelfMean().calculate(x, groups)
    expected = pd.DataFrame([[np.nan, np.nan, 20., 10.]], columns=x.columns)
    pd.testing.assert_frame_equal(result, expected)
