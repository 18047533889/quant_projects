"""Metadata-only scan-cost forwarding for bounded source wrappers."""
from __future__ import annotations
import inspect
from typing import Any
import pandas as pd

def estimate_window_scan_cost(inner: Any, *, start: Any, end: Any,
                             fields: Any, time_range: Any = None,
                             instruments: Any = None, dataset: Any = None):
    estimator = getattr(inner, 'estimate_scan_cost', None)
    if not callable(estimator):
        raise NotImplementedError('inner source has no scan-cost estimator')
    if time_range is not None and (not isinstance(time_range, (tuple, list)) or len(time_range) != 2):
        raise ValueError('time_range must be a pair of inclusive bounds')
    requested_start, requested_end = time_range if time_range is not None else (None, None)
    values = [pd.Timestamp(v) for v in (requested_start, requested_end, start, end) if v is not None]
    zone = next((v.tz for v in values if v.tz is not None), None)
    def normalize(value):
        if value is None:
            return None
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError('scan-cost bounds must not be NaT')
        if zone is not None:
            return stamp.tz_localize(zone) if stamp.tz is None else stamp.tz_convert(zone)
        return stamp
    lowers = [normalize(v) for v in (requested_start, start) if v is not None]
    uppers = [normalize(v) for v in (requested_end, end) if v is not None]
    lower = max(lowers) if lowers else None
    upper = min(uppers) if uppers else None
    if lower is not None and upper is not None and lower > upper:
        raise ValueError('disjoint window and scan-cost time ranges')
    span = (lower, upper) if lower is not None or upper is not None else None
    if instruments is None:
        instruments = getattr(inner, 'instrument_filter', None)
    signature = inspect.signature(estimator)
    accepts_extra = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    arguments = {'fields': fields}
    for key, value in (('time_range', span), ('instruments', instruments), ('dataset', dataset)):
        parameter = signature.parameters.get(key)
        if accepts_extra or (parameter is not None and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY):
            arguments[key] = value
    # Unsupported filters leave the original, conservative full-source receipt
    # intact. Never scale row counts/bytes or hide estimator/snapshot failures.
    return estimator(**arguments)
