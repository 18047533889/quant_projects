# -*- coding: utf-8 -*-
"""Focused regression tests for final model-operator certification gaps."""
from __future__ import annotations

import numpy as np

from factor_engine.cleaned_operators.dependence_ext import _partial_dcor_proxy


def test_partial_distance_proxy_singular_control_fails_closed() -> None:
    """A perfect conditioning geometry is unidentified, not epsilon-regularizable."""
    x = np.linspace(-2.0, 2.0, 64)
    y = x.copy()
    z = x.copy()
    assert np.isnan(_partial_dcor_proxy(x, y, z))
