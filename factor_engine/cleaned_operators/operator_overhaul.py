# -*- coding: utf-8 -*-
"""Load the audited daily-factor operator overhaul.

The module is imported after historical implementations.  Registered audited
operators therefore replace the active canonical/backend implementation.  The
final cleanup is called by ``cleaned_operators.load_all`` after legacy dedupe.
"""
from cleaned_operators.overhaul import finalize, register_all

register_all()


def finalize_operator_overhaul() -> None:
    finalize()
