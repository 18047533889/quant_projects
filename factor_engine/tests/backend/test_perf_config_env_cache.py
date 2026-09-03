# -*- coding: utf-8
"""100k GO §26: PerfConfig.from_env() env-cache signature coverage.

Regression: ``from_env()`` reads ``FACTOR_ENGINE_SCHEDULER`` /
``FACTOR_ENGINE_RESOURCE_PROFILE`` / ``FACTOR_ENGINE_COEXIST`` /
``FACTOR_ENGINE_NATIVE_FUSION`` when building the config, but these four
keys were missing from the env-cache key tuple, so mutating them did not
invalidate the cached ``PerfConfig`` and a later ``from_env()`` returned the
stale configuration (100k production run could silently run under the wrong
scheduler / resource profile / coexist / fusion policy).
"""
from __future__ import annotations

import os

import pytest

from factor_engine.runtime.perf_config import PerfConfig

_ENV_KEYS = (
    "FACTOR_ENGINE_SCHEDULER",
    "FACTOR_ENGINE_RESOURCE_PROFILE",
    "FACTOR_ENGINE_COEXIST",
    "FACTOR_ENGINE_NATIVE_FUSION",
)


@pytest.fixture(autouse=True)
def _fresh_env(monkeypatch):
    # Isolate each test from real env + reset the module-level cache.
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    PerfConfig.invalidate_env_cache()
    yield
    PerfConfig.invalidate_env_cache()


def _read_attrs(cfg: PerfConfig) -> tuple:
    return (cfg.scheduler, cfg.resource_profile, cfg.coexist, cfg.native_fusion)


@pytest.mark.parametrize(
    "key,set_to,attr",
    [
        ("FACTOR_ENGINE_SCHEDULER", "serial", "scheduler"),
        ("FACTOR_ENGINE_RESOURCE_PROFILE", "performance", "resource_profile"),
        ("FACTOR_ENGINE_COEXIST", "false", "coexist"),
        ("FACTOR_ENGINE_NATIVE_FUSION", "false", "native_fusion"),
    ],
)
def test_env_change_invalidates_cache(monkeypatch, key, set_to, attr):
    """Mutating one of the four previously-uncached env keys must reload."""
    cfg_a = PerfConfig.from_env()
    assert getattr(cfg_a, attr) != _EXPECTED[key]  # guard: real env not already set

    monkeypatch.setenv(key, set_to)
    cfg_b = PerfConfig.from_env()  # cache key now differs -> must rebuild
    assert getattr(cfg_b, attr) == _EXPECTED[key], f"{key} change did not reload"


_DEFAULTS = {
    "FACTOR_ENGINE_SCHEDULER": "adaptive",
    "FACTOR_ENGINE_RESOURCE_PROFILE": "balanced",
    "FACTOR_ENGINE_COEXIST": True,
    "FACTOR_ENGINE_NATIVE_FUSION": True,
}
_EXPECTED = {
    "FACTOR_ENGINE_SCHEDULER": "serial",
    "FACTOR_ENGINE_RESOURCE_PROFILE": "performance",
    "FACTOR_ENGINE_COEXIST": False,
    "FACTOR_ENGINE_NATIVE_FUSION": False,
}


def test_cache_key_covers_all_read_keys():
    """Every env key read in from_env() must be in the cache signature.

    Direct source-inspection regression so a future read key cannot silently
    drop out of the cache-key tuple again.
    """
    import re
    import inspect

    src = inspect.getsource(PerfConfig.from_env)
    # keys read via os.environ.get("X") / _env_*(..., "X")
    read_keys = set()
    for m in re.finditer(r'(?:os\.environ\.get|_env_(?:str|int|float|bool|backtest_engine))\(\s*"([A-Z0-9_]+)"', src):
        read_keys.add(m.group(1))
    # cache_names tuple literal between the assignment and cache_key
    body = src[src.find("cache_names = ("): src.find("cache_key = ")]
    cached_keys = set(re.findall(r'"([A-Z0-9_]+)"', body))
    missing = sorted(read_keys - cached_keys)
    assert not missing, f"env keys read but missing from cache key: {missing}"
