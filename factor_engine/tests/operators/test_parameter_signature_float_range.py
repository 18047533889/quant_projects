import math
import sys

import numpy as np
import pytest

from factor_engine.parameter_canonicalizer import _output_signature, _round_sig


@pytest.mark.parametrize('value', [1e-300, -1e-300, 5e-324, -5e-324,
                                 1e300, -1e300, sys.float_info.max,
                                 -sys.float_info.max, 0.0, 1.23456789012345])
def test_rounding_preserves_finite_range(value):
    result = _round_sig(value, 12)
    assert math.isfinite(result)
    assert result == pytest.approx(value, rel=1e-11, abs=0)


def test_tiny_outputs_keep_distinct_compositional_signatures():
    values = np.array([1e-300, 2e-300, 3e-300])
    assert _output_signature(values) != _output_signature(values * 2)


def test_ordinary_significant_digits():
    assert _round_sig(1.23456789012345, 12) == 1.23456789012
    assert _round_sig(-1.23456789012345, 12) == -1.23456789012
    assert _round_sig(1.25, 2) == 1.3
    assert _round_sig(-1.35, 2) == -1.3
