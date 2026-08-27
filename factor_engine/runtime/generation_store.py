# -*- coding: utf-8 -*-
"""R45: GenerationStore 抽象 —— generation 原子提交的存储后端平面。

STRICT_REMOTE（production）下增量 planner 必须**零本地持久写**：factor part /
state part / manifest / CURRENT 指针全部经远程 ``ObjectStore`` 落盘。research /
perf 场景允许本地目录或 tmpfs。本模块把这条 "后端差异" 收敛成一个最小协议：

- ``GenerationStore`` Protocol —— ``put_part`` / ``write_manifest`` /
  ``set_current`` 三步构成一次完整的 generation 原子提交：
    1. 先写所有 immutable parts；
    2. 再写 ``manifest``（提交点）—— 读者要么看到完整旧代要么看到完整新代；
    3. 最后翻转 ``CURRENT`` 指针。
- ``LocalGenerationStore``：本地目录后端（research / 测试）。part 直接写进
  ``generation=<id>/`` 根目录（与既有 IncrementalCommitTransaction 布局一致），
  单 part 走 temp + ``os.replace`` 原子落盘；manifest temp + ``os.replace``；
  CURRENT temp + ``os.replace``。
- ``TmpfsGenerationStore``：tmpfs 内存盘后端（research / perf，非持久权威）。
  布局与 Local 相同，但根目录落在 tmpfs（默认 ``/dev/shm``，可用
  ``FACTOR_ENGINE_TMPFS_ROOT`` 覆盖）。
- ``ObjectStoreGenerationStore``：生产后端。**不产生任何本地 Path 写入**——
  全部通过 :class:`ObjectStore` 写远程对象。part 键严格分离命名空间：
  ``factor/<generation>/<name>`` vs ``state/<generation>/<name>``，杜绝
  factor/state 同名互相覆盖。manifest 键 ``generation/<id>/manifest.json``，
  CURRENT 键 ``generation/CURRENT``。LocalObjectStore 的 ``put_object`` 已是
  temp + ``os.replace`` 原子写，故三个步骤逐对象原子。

命名空间守卫：无论在哪个后端，**同一个 part 名**不能既注册为 factor 又注册为
state —— 提交层在 ``IncrementalCommitTransaction.stage`` 里先做同名交集校验
（抛 :class:`NamespaceConflictError``），生产后端再以 ``factor/`` vs ``state/``
物理前缀把命名空间彻底分离，做到"同名覆盖不可能发生"。

R44-P0 契约：ObjectGenerationStore 零本地持久字节；本地后端仅 research /
tmpfs 可用，production（STRICT_REMOTE）必须用 ObjectGenerationStore。
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ObjectStore 仅类型引用；运行时惰性 import 避免在未接入对象平面时拉入
# data_access 的循环依赖。
try:  # pragma: no cover - 类型引用（运行时惰性）
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from data_access.read.object_store import ObjectStore
    else:
        ObjectStore = Any  # type: ignore
except Exception:  # pragma: no cover
    ObjectStore = Any  # type: ignore

__all__ = [
    "GenerationStore",
    "LocalGenerationStore",
    "TmpfsGenerationStore",
    "ObjectGenerationStore",
    "NamespaceConflictError",
    "resolve_generation_store",
]


class NamespaceConflictError(RuntimeError):
    """同一 part 名被同时注册为 factor 与 state（跨命名空间同名覆盖守卫）。"""


@runtime_checkable
class GenerationStore(Protocol):
    """generation 提交平面。

    ``put_part`` 写一个 immutable part；``write_manifest`` 写提交点；
    ``set_current`` 翻转 CURRENT 指针。调用方保证：parts 全部写完后再
    write_manifest，最后 set_current —— 读者要么看到完整旧代要么完整新代。
    """

    def put_part(self, generation: str, part_kind: str, name: str, data: bytes) -> None: ...

    def write_manifest(self, generation: str, manifest: dict[str, Any]) -> None: ...

    def set_current(self, generation: str) -> None: ...

    @property
    def is_object_backed(self) -> bool:
        """生产后端（零本地 Path 写入）为 True。"""
        ...


class LocalGenerationStore:
    """本地目录后端（research / dev-only）。

    布局与既有 ``IncrementalCommitTransaction`` 完全一致：part 直接落
    ``generation=<id>/`` 根目录，单 part temp + ``os.replace``；manifest 与
    CURRENT 亦 temp + ``os.replace`` 原子翻转。

    P0-15: 本地后端是 research/dev-only —— production（STRICT_REMOTE）必须用
    :class:`ObjectGenerationStore` 零本地持久写。``production_guard`` 开启时
    在 production 模式构造本类直接抛 :class:`RuntimeError`（显式 guard，避免
    生产执行静默落到本地目录后端）。

    参数:
        root: 世代根目录（与 CURRENT 指针平级）。
        backend: ``"local"``（磁盘）或 ``"tmpfs"``（临时盘）。
        production_guard: production 模式下拒绝构造（默认关闭——研究/测试
          不强制；生产入口显式传 True）。
    """

    #: 本地 / 研究 writer —— 显式标记 research/dev-only。
    RESEARCH_DEV_ONLY = True

    def __init__(
        self,
        root: str | Path,
        *,
        backend: str = "local",
        production_guard: bool = False,
    ) -> None:
        if production_guard:
            from factor_engine.runtime.production_policy import is_production_mode

            if is_production_mode():
                raise RuntimeError(
                    "production 模式禁止 LocalGenerationStore（本地持久写）。"
                    "production 必须用 ObjectGenerationStore（零本地字节）。"
                )
        self.root = Path(root)
        self.backend = backend

    @property
    def is_object_backed(self) -> bool:
        return False

    # -- 路径 --
    def _gen_dir(self, generation: str) -> Path:
        return self.root / f"generation={generation}"

    def _current_file(self) -> Path:
        return self.root / "CURRENT"

    # -- 写入 --
    def put_part(self, generation: str, part_kind: str, name: str, data: bytes) -> None:
        # 本地布局：factor 与 state part 都落在同一 generation 目录根；命名空间
        # 冲突由事务层 `stage` 守卫（同一 name 不能跨 kind 注册）。
        dst = self._gen_dir(generation) / name
        _write_atomic(dst, data)

    def write_manifest(self, generation: str, manifest: dict[str, Any]) -> None:
        gen_dir = self._gen_dir(generation)
        gen_dir.mkdir(parents=True, exist_ok=True)
        tmp = gen_dir / f".manifest.{uuid.uuid4().hex}.tmp"
        tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        os.replace(str(tmp), str(gen_dir / "manifest.json"))

    def set_current(self, generation: str) -> None:
        cur = self._current_file()
        tmp = cur.with_name(f".CURRENT.{uuid.uuid4().hex}.tmp")
        tmp.write_text(generation, encoding="utf-8")
        os.replace(str(tmp), str(cur))


class TmpfsGenerationStore(LocalGenerationStore):
    """tmpfs 内存盘后端（research / perf，非持久）。

    默认根目录为 ``/dev/shm/r45_genstore``；可用环境变量
    ``FACTOR_ENGINE_TMPFS_ROOT`` 覆盖。语义与 :class:`LocalGenerationStore`
    相同，只是把根目录指向 tmpfs。
    """

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            base = Path(os.environ.get("FACTOR_ENGINE_TMPFS_ROOT", "/dev/shm"))
            root = base / "r45_genstore"
        super().__init__(root, backend="tmpfs")


def _write_atomic(dst: Path, data: bytes) -> None:
    """本地原子写单个 part 文件（temp + os.replace）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.tmp.{uuid.uuid4().hex}"
    tmp.write_bytes(data)
    os.replace(str(tmp), str(dst))


class ObjectGenerationStore:
    """生产后端：全部通过 :class:`ObjectStore` 落远程对象，零本地持久字节。

    键布局（POSIX 相对键，互不相交）：

    * factor part : ``factor/<generation>/<name>``
    * state part  : ``state/<generation>/<name>``
    * manifest    : ``generation/<generation>/manifest.json``
    * CURRENT     : ``generation/CURRENT``

    每个对象经 ``ObjectStore.put_object`` 原子写入（LocalObjectStore 实现即
    temp + ``os.replace``），故三阶段提交单对象原子。part 名命名空间分离杜绝
    factor/state 同名互相覆盖。生产（STRICT_REMOTE）下必须使用本后端。
    """

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    @property
    def is_object_backed(self) -> bool:
        return True

    @property
    def object_store(self) -> ObjectStore:
        return self._store

    # -- 键工厂 --
    def _part_key(self, generation: str, part_kind: str, name: str) -> str:
        if part_kind == "factor":
            return f"factor/{generation}/{name}"
        if part_kind == "state":
            return f"state/{generation}/{name}"
        raise ValueError(f"未知 part_kind: {part_kind!r}")

    def _manifest_key(self, generation: str) -> str:
        return f"generation/{generation}/manifest.json"

    def _current_key(self) -> str:
        return "generation/CURRENT"

    # -- 写入 --
    def put_part(self, generation: str, part_kind: str, name: str, data: bytes) -> None:
        self._store.put_object(self._part_key(generation, part_kind, name), bytes(data))

    def write_manifest(self, generation: str, manifest: dict[str, Any]) -> None:
        self._store.put_object(
            self._manifest_key(generation),
            json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )

    def set_current(self, generation: str) -> None:
        self._store.put_object(self._current_key(), generation.encode("utf-8"))

    # -- 读取辅助（生产读者） --
    def load_manifest(self, generation: str) -> dict[str, Any]:
        key = self._manifest_key(generation)
        reader = self._store.open_reader(key)
        if reader is None:
            return {}
        try:
            raw = reader.read()
        finally:
            try:
                reader.close()
            except Exception:
                pass
        return json.loads(raw.decode("utf-8"))

    def read_part(self, generation: str, part_kind: str, name: str) -> bytes:
        key = self._part_key(generation, part_kind, name)
        reader = self._store.open_reader(key)
        if reader is None:
            return b""
        try:
            return reader.read()
        finally:
            try:
                reader.close()
            except Exception:
                pass


def resolve_generation_store(
    *,
    root: str | Path | None = None,
    object_store: ObjectStore | None = None,
    backend: str = "local",
) -> GenerationStore:
    """按后端解析一个 ``GenerationStore``（生产 object_backed 优先）。

    优先级：
      1. ``object_store`` 显式给出 → ObjectGeneration（production，零本地）。
      2. ``backend="tmpfs"`` → TmpfsGenerationStore。
      3. ``backend="object"`` 且提供了 object_store → ObjectGeneration。
      4. 缺省 → LocalGenerationStore（research / 测试）。
    """
    if object_store is not None or backend == "object":
        if object_store is None:
            raise ValueError("backend='object' 需要显式 object_store")
        return ObjectGenerationStore(object_store)
    if backend == "tmpfs":
        return TmpfsGenerationStore(root)
    return LocalGenerationStore(root)  # type: ignore[arg-type]
