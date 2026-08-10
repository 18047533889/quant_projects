"""R29-P0 #205 —— DataReadSession：job 级读会话（resolution 复用 + 上下文绑定）。

FactorEngine 批量挖因子时对同一批 dataset/factor_id 反复 ``read``——每次 prepare
都重新 glob、stat、镜像检查、schema 校验。本会话把：

    1. **request-scoped 执行上下文**（principal/authorizer/credential/run_mode）
       绑定整个 job：``security_digest`` 一致、嵌套读（calendar / universe /
       factor catalog / 远程 credential）自动继承请求级授权，不再退回 process 级；
    2. **calendar 世界在 job 首读即冻结**：PIT 语义跨因子稳定（R27-I）；
    3. **resolution 缓存注入 store**：``(dataset, params, time_range, instruments)
       → (paths, files)``，prepare 命中跳过 glob 重解析 / 文件 stat / 镜像检查
       ——同一批因子重复读同一 dataset 时 **prepare ONCE**。

生命周期：:

    with DataReadSession(store, principal=..., credential_provider=...) as s:
        s.read("factor_lake", factor_id="a")
        s.read("factor_lake", factor_id="b")   # 同一 (dataset,params) 命中缓存

退出时恢复 store 原 ``_resolution_cache`` 与执行上下文。

**契约**：仅**读**场景使用——job 内写数据会导致 resolution 缓存陈旧
（路径/文件集在写后变化）。写场景请用 Store 直连或新开会话。
"""
from __future__ import annotations

from typing import Any

from data_access.security.execution_context import (
    _execution_ctx_var,
    resolve_execution_context,
)


class DataReadSession:
    """job 级读会话：上下文绑定 + calendar 冻结 + resolution 复用。"""

    def __init__(
        self,
        store: Any,
        *,
        principal: Any = None,
        authorizer: Any = None,
        credential_provider: Any = None,
        run_mode: Any = None,
        request_id: str | None = None,
    ) -> None:
        self._store = store
        base = resolve_execution_context(
            principal=principal,
            authorizer=authorizer,
            credential_provider=credential_provider,
            run_mode=run_mode,
        )
        self._ctx = type(base)(
            principal=base.principal,
            authorizer=base.authorizer,
            access_policy=base.access_policy,
            credential_provider=base.credential_provider,
            request_id=request_id,
            run_mode=base.run_mode,
            security_digest=base.security_digest,
        )
        self._resolution_cache: dict[tuple[Any, ...], Any] = {}
        self._token: Any = None
        self._prev_cache: Any = None
        self._closed = False

    def __enter__(self) -> "DataReadSession":
        if self._closed:
            raise RuntimeError("DataReadSession 已关闭，无法再次进入")
        # 1) 绑定请求级执行上下文（整个 job）。
        self._token = _execution_ctx_var.set(self._ctx)
        # 2) 注入 resolution 缓存（退出时恢复原值）。
        self._prev_cache = getattr(self._store, "_resolution_cache", None)
        self._store._resolution_cache = self._resolution_cache
        # 3) job 级冻结 calendar 世界（PIT 语义跨因子稳定）。
        try:
            self._store.lock_calendars()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _execution_ctx_var.reset(self._token)
            self._token = None
        if self._prev_cache is not None or hasattr(self._store, "_resolution_cache"):
            self._store._resolution_cache = self._prev_cache
        self._closed = True

    # ---- 便捷代理（resolution 缓存自动生效）----

    def read(self, *args: Any, **kwargs: Any) -> Any:
        return self._store.read(*args, **kwargs)

    def read_joined(self, *args: Any, **kwargs: Any) -> Any:
        return self._store.read_joined(*args, **kwargs)

    def read_factors(self, *args: Any, **kwargs: Any) -> Any:
        return self._store.read_factors(*args, **kwargs)

    def scan(self, *args: Any, **kwargs: Any) -> Any:
        return self._store.scan(*args, **kwargs)

    def prepare_read(self, *args: Any, **kwargs: Any) -> Any:
        return self._store.prepare_read(*args, **kwargs)

    def execute_prepared_read(self, *args: Any, **kwargs: Any) -> Any:
        return self._store.execute_prepared_read(*args, **kwargs)

    def describe_dataset(self, *args: Any, **kwargs: Any) -> Any:
        return self._store.describe_dataset(*args, **kwargs)

    def clear_resolution_cache(self) -> None:
        """job 内显式失效 resolution 缓存（如外部数据变化后）。"""
        self._resolution_cache.clear()


__all__ = ["DataReadSession"]
