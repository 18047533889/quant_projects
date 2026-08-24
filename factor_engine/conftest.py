# -*- coding: utf-8 -*-
"""Pytest conftest for factor_engine tests.

Ensures the operator registry is writable before any test module imports
that trigger operator registration.
"""
from factor_engine.cleaned_operators.registry import OperatorRegistry


def setup_module(module):
    """Reset registry lifecycle to allow operator registration during test imports."""
    if OperatorRegistry._lifecycle != OperatorRegistry.Lifecycle.BUILDING:
        OperatorRegistry._lifecycle = OperatorRegistry.Lifecycle.BUILDING
