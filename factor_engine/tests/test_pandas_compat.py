"""``pandas_compat``：默认解析为 pandas；缓存可重置。"""

from backend.pandas_compat import reset_pandas_module_cache_for_tests, resolve_pandas_module


def test_resolve_pandas_defaults_to_real_pandas():
    reset_pandas_module_cache_for_tests()
    pd = resolve_pandas_module()
    assert pd.__name__ == "pandas"


def test_build_backend_pandas_modin_sets_env(monkeypatch):
    import os

    from backend.factory import build_backend
    from backend.pandas_compat import reset_pandas_module_cache_for_tests

    reset_pandas_module_cache_for_tests()
    monkeypatch.delenv("FACTOR_ENGINE_USE_MODIN", raising=False)
    build_backend("pandas_modin")
    assert os.environ.get("FACTOR_ENGINE_USE_MODIN") == "1"
    monkeypatch.delenv("FACTOR_ENGINE_USE_MODIN", raising=False)
    reset_pandas_module_cache_for_tests()
