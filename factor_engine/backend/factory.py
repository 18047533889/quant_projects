import os

from .debug_backend import DebugBackend
from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend


def build_backend(backend_type: str):
    """构建后端。``pandas_modin`` 会在首次使用 pandas API 前设置 ``FACTOR_ENGINE_USE_MODIN=1``。"""
    normalized = backend_type.strip().lower()

    if normalized == "debug":
        return DebugBackend()  # 打印计划树，不调数据源
    if normalized == "pandas":
        return PandasBackend()
    if normalized == "pandas_modin":
        os.environ.setdefault("FACTOR_ENGINE_USE_MODIN", "1")
        return PandasBackend()  # 与 pandas 同类，经 pandas_compat 走 modin.pandas
    if normalized == "polars":
        return PolarsBackend()
    if normalized == "polars_lazy":
        return PolarsBackend(use_lazy=True)  # LazyFrame 延迟计算再 collect

    raise ValueError(f"Unsupported backend type: {backend_type}")
