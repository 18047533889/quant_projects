class CacheManager:
    """简单内存键值缓存。

    - **列数据**可由数据源自行管理；本类主要用于 ``PandasBackend`` **子树求值结果**复用
     （键为计划子树的结构化字符串，见 ``backend/pandas_backend._plan_cache_key``）。
    """

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    def get(self, key: str):
        return self._cache.get(key)

    def set(self, key: str, value) -> None:
        self._cache[key] = value
