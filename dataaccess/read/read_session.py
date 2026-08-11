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

from dataclasses import dataclass
from typing import Any, Callable

from data_access.security.execution_context import (
    _execution_ctx_var,
    resolve_execution_context,
)


@dataclass(frozen=True)
class PhysicalResolutionContext:
    """R40 #56：物理解析缓存的**身份维度**。

    resolution cache key 过去只含 ``(dataset, params, time_range, instruments)``——
    不区分 namespace / contract_digest / source_profile / mutation_generation。
    两个 job 用相同 params/范围读同一 dataset，但 contract 已更新、或数据源
    profile 变化、或发生了 mutation——旧缓存仍会命中（把「变了的版本」当成
    同一版本读）。本 dataclass 把这些维度打包进 cache key。
    """

    namespace: str = ""
    contract_digest: str = ""
    source_profile: str = ""
    mutation_generation: str = ""

    def to_cache_tuple(self) -> tuple[str, str, str, str]:
        return (self.namespace, self.contract_digest, self.source_profile, self.mutation_generation)

    @classmethod
    def from_store(cls, store: Any, dataset: str) -> "PhysicalResolutionContext":
        """从 store 防御性派生各维度（任一步失败 → 空串，不抛）。"""
        namespace = ""
        source_profile = ""
        mutation_generation = ""
        try:
            ds = store._registry.get(dataset)
            namespace = str(getattr(ds, "namespace", "") or getattr(ds, "provider", "") or "")
            source_profile = "|".join([
                str(getattr(ds, "storage", "") or getattr(ds, "source", "") or ""),
                str(getattr(ds, "format", "") or ""),
                str(getattr(ds, "base_path", "") or ""),
            ])
        except Exception:
            pass
        try:
            token = store.manifest_version(dataset)
            if isinstance(token, dict):
                mutation_generation = str(
                    token.get("manifest_generation_id")
                    or token.get("source_epoch")
                    or ""
                )
        except Exception:
            pass
        contract_digest = ""
        try:
            contract_digest = str(store._contract_digest_for(dataset) or "")
        except Exception:
            pass
        return cls(
            namespace=namespace,
            contract_digest=contract_digest,
            source_profile=source_profile,
            mutation_generation=mutation_generation,
        )


class _ResolutionCache(dict):
    """R40 #56：dict 子类——get/set 时按当前 PhysicalResolutionContext 扩键。

    Store 以 ``(dataset, params_fingerprint, time_range, instruments)`` 为基键
    访问 resolution cache；本类在基键后追加 ``namespace / contract_digest /
    source_profile / mutation_generation`` 四维——contract 更新 / mutation /
    source profile 变化都会改变 cache key，旧条目不再误命中。
    """

    def __init__(self, context_provider: Callable[[str], PhysicalResolutionContext] | None = None) -> None:
        super().__init__()
        self._context_provider = context_provider
        self._memo: dict[str, PhysicalResolutionContext] = {}

    def _enrich(self, key: Any) -> Any:
        if self._context_provider is None or not isinstance(key, tuple) or not key:
            return key
        dataset = key[0]
        if not isinstance(dataset, str):
            return key
        if dataset not in self._memo:
            try:
                self._memo[dataset] = self._context_provider(dataset)
            except Exception:
                return key
        return key + self._memo[dataset].to_cache_tuple()

    def get(self, key: Any, default: Any = None) -> Any:
        return super().get(self._enrich(key), default)

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(self._enrich(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(self._enrich(key), value)

    def __contains__(self, key: object) -> bool:
        return super().__contains__(self._enrich(key))


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
        # R40 #56：resolution cache 键含 PhysicalResolutionContext 维度（namespace /
        # contract_digest / source_profile / mutation_generation）。
        self._resolution_cache: dict[tuple[Any, ...], Any] = _ResolutionCache(
            context_provider=lambda ds: PhysicalResolutionContext.from_store(store, ds)
        )
        self._token: Any = None
        self._mode_token: Any = None
        self._prev_cache: Any = None
        self._closed = False

    def __enter__(self) -> "DataReadSession":
        if self._closed:
            raise RuntimeError("DataReadSession 已关闭，无法再次进入")
        # 1) 绑定请求级执行上下文（整个 job）。
        self._token = _execution_ctx_var.set(self._ctx)
        # 1b) R39 P0 #50：job 级 RuntimeModeIdentity 单一权威——整个 session 内
        #     QueryBudget / is_strict_semantics / PIT / calendar 都读它，不再各层
        #     重读 env。退出时恢复。
        from data_access.runtime.mode_identity import (
            reset_runtime_mode_identity,
            set_runtime_mode_identity,
        )

        rm = getattr(self._ctx, "run_mode", None)
        if rm is not None:
            self._mode_token = set_runtime_mode_identity(
                rm, source="DataReadSession"
            )
        # 2) R38 P0-050：resolution 缓存放 **ContextVar**（request-scoped），不再
        #    修改 Store 全局属性——并发 session A/B overlap 时 A exit 不会清掉
        #    B 的 cache。Store ``_resolution_cache`` 仅作向后兼容显示。
        from data_access.runtime.read_session_context import (
            reset_resolution_cache,
            set_resolution_cache,
        )

        self._cache_token = set_resolution_cache(self._resolution_cache)
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
        try:
            from data_access.runtime.read_session_context import (
                reset_resolution_cache,
            )

            if getattr(self, "_cache_token", None) is not None:
                reset_resolution_cache(self._cache_token)
                self._cache_token = None
        except Exception:
            pass
        try:
            from data_access.runtime.mode_identity import reset_runtime_mode_identity

            if getattr(self, "_mode_token", None) is not None:
                reset_runtime_mode_identity(self._mode_token)
                self._mode_token = None
        except Exception:
            pass
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
