import numpy as np
import pytest
from quant_evaluator.metrics import multiple_testing as mt

METHODS = [mt.bonferroni_correction, mt.benjamini_hochberg_correction,
           mt.holm_bonferroni_correction, mt.sidak_correction]

@pytest.mark.parametrize("method", METHODS)
def test_complex_pvalues_cannot_be_silently_projected_to_real(method):
    with pytest.raises(ValueError, match="real"):
        method(np.array([.001 + .5j, .5 + 0j]))

@pytest.mark.parametrize("method", METHODS)
def test_extended_precision_out_of_range_is_checked_before_float64_cast(method):
    invalid = np.nextafter(np.longdouble(1), np.longdouble(2))
    if invalid == np.float64(invalid):
        pytest.skip("platform longdouble has no extra precision")
    with pytest.raises(ValueError, match="outside"):
        method(np.array([invalid, np.longdouble(".02")], dtype=np.longdouble))
