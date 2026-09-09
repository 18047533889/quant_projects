"""Canonical probe schedule: H counts price intervals after entry (v2)."""
import numpy as np


def cohort_schedule(num_times: int, holding: int):
    if isinstance(holding, bool) or not isinstance(holding, (int, np.integer)) or holding < 1:
        raise ValueError("holding must be a positive integer number of post-entry intervals")
    signals = np.arange(num_times, dtype=np.int64)
    entries = signals + 1
    exits = entries + holding
    return entries, exits, exits < num_times
