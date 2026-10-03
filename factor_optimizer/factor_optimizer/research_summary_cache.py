"""Bounded exact-content memoization for successful RAW metric summaries."""
from __future__ import annotations

import numpy as np

MAX_PANEL_KEY_BYTES = 128 * 1024


class RawMetricSummaryCache:
    """One entry with at most 128 KiB panel-key bytes; not a process RSS cap."""

    def __init__(self):
        self._key = None
        self._metrics = None

    @staticmethod
    def _content_key(series, periods_per_year):
        # Invalid or unhashable period inputs must still reach summarize(), which
        # owns validation and its public error behavior.
        try:
            hash(periods_per_year)
        except TypeError:
            return None

        panel = np.asarray(series)
        # Object bytes contain process-specific pointers, not stable content.
        # summarize remains authoritative for these uncommon inputs; simply do
        # not memoize them.
        if panel.dtype.hasobject or panel.nbytes > MAX_PANEL_KEY_BYTES:
            return None
        contiguous = np.ascontiguousarray(panel)
        dtype = repr(panel.dtype.descr) if panel.dtype.fields else panel.dtype.str
        return (panel.shape, dtype, contiguous.tobytes(),
                type(periods_per_year), periods_per_year)

    def summarize(self, series, *, periods_per_year=252):
        """Return a fresh metric mapping, caching only successful summaries."""
        key = self._content_key(series, periods_per_year)
        if key is not None and key == self._key and self._metrics is not None:
            return dict(self._metrics)

        # Import on each miss so instrumentation and wrappers around the public
        # summarizer remain observable; metric logic and validation stay there.
        from factor_optimizer import research_fitness
        metrics = research_fitness.summarize(
            series, periods_per_year=periods_per_year)
        if key is not None:
            self._key, self._metrics = key, dict(metrics)
        return dict(metrics)
