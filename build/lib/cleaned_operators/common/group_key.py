# -*- coding: utf-8 -*-
"""Unified group-key missing-value normalisation (R11 P1-10).

Every ``group_*`` / ``industry_*`` / ``peer_*`` / holder-relation kernel must
judge a group label missing with the SAME rule.  Before this module each kernel
inlined its own check (``lab is None``, ``np.isnan(lab)``, ``pd.isna``,
``.dropna()``), so a ``pd.NA`` / ``NaT`` / ``""`` / ``None`` / ``np.nan`` key
could be treated as a REAL group in one kernel and dropped in another — a
handful of ``pd.NA`` cells could even become one phantom group.  All kernels
use :func:`is_missing_group_key` / :func:`normalize_group_key` instead.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def is_missing_group_key(lab: Any) -> bool:
    """True when ``lab`` is a missing group key under the unified rule.

    A missing key is: ``None``, ``np.nan``, ``pd.NA``, ``NaT``, or the empty
    string ``""``.  Anything else — including a numeric ``0`` — is a REAL key.
    Every pandas missing sentinel collapses to one representation so a phantom
    group (e.g. several ``pd.NA`` cells joined as a single label) can never
    appear.
    """
    if lab is None:
        return True
    if isinstance(lab, float) and np.isnan(lab):
        return True
    if isinstance(lab, (np.floating,)) and np.isnan(lab):
        return True
    if isinstance(lab, str) and lab == "":
        return True
    if isinstance(lab, np.ndarray):  # a per-cell array is not a scalar key
        return False
    try:
        result = pd.isna(lab)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        if isinstance(result, np.ndarray):
            return bool(np.all(result))
        return bool(result)
    except (TypeError, ValueError):
        # Unhashable / exotic key object: treat as a real key, never a phantom
        # missing (fail-open on exotic keys is safe — they cannot group either).
        return False


def normalize_group_key(lab: Any) -> Any:
    """Normalise a group key to a single comparable form.

    Every missing representation becomes ``None``; numpy scalars are unwrapped
    to native Python so dict/set membership and equality behave consistently.
    """
    if is_missing_group_key(lab):
        return None
    if isinstance(lab, (np.integer, np.floating, np.bool_)):
        return lab.item()
    return lab
