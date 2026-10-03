"""Exact-content memoization for RAW summaries."""
import numpy as np
import pytest


def test_identical_series_reuses_successful_summary_and_returns_fresh_dict(monkeypatch):
    """Repeated candidates should not recompute an identical RAW summary."""
    from factor_optimizer import research_fitness
    from factor_optimizer.research_summary_cache import RawMetricSummaryCache

    series = np.column_stack((np.linspace(-.1, .1, 60),
                              np.sin(np.arange(60)) * .01,
                              np.full(60, .4)))
    original = research_fitness.summarize
    calls = []

    def counted(values, *, periods_per_year=252):
        calls.append(values)
        return original(values, periods_per_year=periods_per_year)

    monkeypatch.setattr(research_fitness, 'summarize', counted)
    cache = RawMetricSummaryCache()
    expected = original(series)
    first = cache.summarize(series)
    first['sharpe'] = 999.
    second = cache.summarize(series.copy())

    assert len(calls) == 1
    assert second == expected
    assert second is not first


def _valid_series(dtype=float):
    n = np.arange(60)
    return np.column_stack((np.sin(n / 5), np.cos(n / 7) * .01,
                            np.full(60, .4))).astype(dtype)


def test_different_missingness_panels_and_mutated_contents_are_distinct():
    from factor_optimizer.research_summary_cache import RawMetricSummaryCache

    cache = RawMetricSummaryCache()
    series = _valid_series()
    cache.summarize(series)
    changed_missingness = series.copy()
    changed_missingness[10, 0] = np.nan
    try:
        cache.summarize(changed_missingness)
    except ValueError as exc:
        assert 'missing' in str(exc)
    else:
        raise AssertionError('different missingness must be summarized independently')

    # A successful panel mutation must also produce a fresh summary.
    changed = series.copy()
    changed[0, 0] += .25
    assert cache.summarize(changed) != cache.summarize(series)


def test_dtype_and_period_type_are_part_of_cache_identity():
    from factor_optimizer import research_fitness
    from factor_optimizer.research_summary_cache import RawMetricSummaryCache

    cache = RawMetricSummaryCache()
    series = _valid_series()
    calls = []
    original = research_fitness.summarize

    def counted(values, *, periods_per_year=252):
        calls.append((values.dtype, type(periods_per_year)))
        return original(values, periods_per_year=periods_per_year)

    research_fitness.summarize = counted
    try:
        cache.summarize(series)
        cache.summarize(series.astype(np.float32))
        cache.summarize(series, periods_per_year=1)
        try:
            cache.summarize(series, periods_per_year=True)
        except ValueError as exc:
            assert 'periods_per_year' in str(exc)
        else:
            raise AssertionError('bool annualization must not hit int-keyed cache')
    finally:
        research_fitness.summarize = original

    assert len(calls) == 4
    assert calls[0][0] == np.dtype(float)
    assert calls[1][0] == np.dtype(np.float32)
    assert calls[2][1] is int
    assert calls[3][1] is bool


def test_noncontiguous_equal_content_reuses_entry(monkeypatch):
    from factor_optimizer import research_fitness
    from factor_optimizer.research_summary_cache import RawMetricSummaryCache

    series = _valid_series()
    wide = np.zeros((60, 6))
    wide[:, ::2] = series
    view = wide[:, ::2]
    assert not view.flags.c_contiguous
    original = research_fitness.summarize
    calls = []

    def counted(values, *, periods_per_year=252):
        calls.append(1)
        return original(values, periods_per_year=periods_per_year)

    monkeypatch.setattr(research_fitness, 'summarize', counted)
    cache = RawMetricSummaryCache()
    expected = cache.summarize(series)
    assert cache.summarize(view) == expected
    assert len(calls) == 1


@pytest.mark.parametrize('failure', ['missing', 'undefined'])
def test_typed_failures_are_not_cached_and_keep_details(monkeypatch, failure):
    from factor_optimizer import research_fitness
    from factor_optimizer.research_summary_cache import RawMetricSummaryCache

    series = _valid_series()
    if failure == 'missing':
        series[3, 0] = np.nan
    else:
        series[:, :2] = 0.
    original = research_fitness.summarize
    with pytest.raises(research_fitness.JointMetricsUnavailable) as expected:
        original(series)
    calls = []

    def counted(values, *, periods_per_year=252):
        calls.append(1)
        return original(values, periods_per_year=periods_per_year)

    monkeypatch.setattr(research_fitness, 'summarize', counted)
    cache = RawMetricSummaryCache()
    with pytest.raises(research_fitness.JointMetricsUnavailable) as first:
        cache.summarize(series)
    with pytest.raises(research_fitness.JointMetricsUnavailable) as second:
        cache.summarize(series)
    assert len(calls) == 2
    assert first.value.code == second.value.code == expected.value.code
    assert first.value.metrics == second.value.metrics == expected.value.metrics
    assert str(first.value) == str(second.value) == str(expected.value)


def test_one_entry_evicts_previous_content(monkeypatch):
    from factor_optimizer import research_fitness
    from factor_optimizer.research_summary_cache import RawMetricSummaryCache

    a = _valid_series()
    b = a.copy()
    b[0, 0] += .25
    original = research_fitness.summarize
    calls = []

    def counted(values, *, periods_per_year=252):
        calls.append(1)
        return original(values, periods_per_year=periods_per_year)

    monkeypatch.setattr(research_fitness, 'summarize', counted)
    cache = RawMetricSummaryCache()
    expected = cache.summarize(a)
    cache.summarize(b)
    assert cache.summarize(a) == expected
    assert len(calls) == 3


def test_large_content_is_summarized_without_retaining_panel_bytes(monkeypatch):
    from factor_optimizer import research_fitness
    from factor_optimizer.research_summary_cache import RawMetricSummaryCache

    # 240 kB: much larger than the 128 KiB key envelope, still a tiny fixture.
    series = np.ones((10_000, 3), dtype=np.float64)
    calls = []

    def counted(values, *, periods_per_year=252):
        calls.append(1)
        return {'rank_ic': .1}

    monkeypatch.setattr(research_fitness, 'summarize', counted)
    cache = RawMetricSummaryCache()
    assert cache.summarize(series) == {'rank_ic': .1}
    assert cache.summarize(series) == {'rank_ic': .1}
    assert len(calls) == 2
    assert cache._key is None


@pytest.mark.parametrize('extra_elements, expected_calls', [(0, 1), (1, 2)])
def test_panel_key_byte_limit_is_inclusive(monkeypatch, extra_elements, expected_calls):
    from factor_optimizer import research_fitness
    from factor_optimizer.research_summary_cache import (
        MAX_PANEL_KEY_BYTES, RawMetricSummaryCache,
    )

    # A summarizer double isolates byte admission, not metric shape validation.
    panel = np.zeros(MAX_PANEL_KEY_BYTES // 8 + extra_elements, dtype=np.float64)
    calls = []

    def counted(values, *, periods_per_year=252):
        calls.append(1)
        return {'rank_ic': .1}

    monkeypatch.setattr(research_fitness, 'summarize', counted)
    cache = RawMetricSummaryCache()
    assert cache.summarize(panel) == {'rank_ic': .1}
    assert cache.summarize(panel) == {'rank_ic': .1}
    assert len(calls) == expected_calls
