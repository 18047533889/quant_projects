from __future__ import annotations

import contextvars

from backend.panel_polars import get_request_perf_counters


def test_contextvar_default_is_lazily_isolated_per_context() -> None:
    first = contextvars.Context().run(get_request_perf_counters)
    second = contextvars.Context().run(get_request_perf_counters)
    assert first is not second
    first.incr("x")
    assert second.get("x") == 0
