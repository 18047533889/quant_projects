# -*- coding: utf-8 -*-
"""R39-P0-PERF-013：per-root 不再整份复制共享 cache dict。

:class:`ReadOnlyOverlayMap` 为每个 root 提供一个独立的写层（``local``），共享
cache（``base``）只读。读优先 local、miss 再查 base；写只进 local，绝不变更
base —— 从而把 ``runtime.batch_service._execute_root_with_path`` 的 4 个
``dict(...)`` 全量拷贝降为 O(root_count) 的轻量 overlay 创建（Hard Gate-09：
root context shared mapping 不做全量 dict clone）。
"""
from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Any

#: 模块级 dict 条目拷贝计数：新路径（overlay）永远保持 0，是「无全量 dict copy
#: per root」的进程级不变量（R39 Gate-09）。
_dict_entry_copy_count = 0

_MISSING = object()


def _bump_dict_entry_copy_count(n: int) -> None:
    global _dict_entry_copy_count
    _dict_entry_copy_count += n


def dict_entry_copy_count() -> int:
    """进程级已发生的「共享 cache 全量条目拷贝」总数（R39 Gate-09 探针）。"""
    return _dict_entry_copy_count


def reset_dict_entry_copy_count() -> None:
    """测试用：清零进程级拷贝计数。"""
    global _dict_entry_copy_count
    _dict_entry_copy_count = 0


class ReadOnlyOverlayMap(MutableMapping):
    """local-first 覆盖映射：``base`` 只读，写只进 ``local``。

    语义契约（R39-P0-PERF-013）：
    - ``d[k]``：local 命中返回 local；否则返回 base。
    - ``d[k] = v``：只写 local，绝不写 base。
    - ``keys()/items()/values()/__iter__/__contains__/__len__`` 反映 local 与
      base 的并集（local 优先，跨层去重）。
    - ``pop(k, default)``：只从 local 删除；k 仅在 base 时返回 base 值而不删除
      （base 只读，禁止从共享状态删除）。
    - ``__delitem__``：local 命中删除 local；base 命中抛 KeyError（fail-closed，
      不允许改动共享只读状态）。
    - ``clear()``：只清 local（base 保持只读）。
    - 每个 root 的 overlay 拥有独立的 ``local`` dict（``base`` 永不 mutate），
      因此并行 worker 各自写自己的一层，与 R20-241..252 worker-isolation 契约
      一致。
    """

    __slots__ = ("_base", "_local")

    def __init__(self, base: Any | None = None, local: dict | None = None) -> None:
        self._base: Any = base if base is not None else {}
        self._local: dict = local if local is not None else {}

    @property
    def base(self) -> Any:
        return self._base

    @property
    def local(self) -> dict:
        return self._local

    def __getitem__(self, key: Any) -> Any:
        if key in self._local:
            return self._local[key]
        return self._base[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._local[key] = value

    def __delitem__(self, key: Any) -> None:
        if key in self._local:
            del self._local[key]
            return
        if key in self._base:
            raise KeyError(
                f"ReadOnlyOverlayMap: cannot delete base-only key {key!r}: "
                "shared base is read-only"
            )
        raise KeyError(key)

    def __iter__(self) -> Iterator[Any]:
        seen: set[Any] = set()
        for key in self._local:
            seen.add(key)
            yield key
        for key in self._base:
            if key not in seen:
                seen.add(key)
                yield key

    def __contains__(self, key: Any) -> bool:
        return key in self._local or key in self._base

    def __len__(self) -> int:
        if not self._local:
            return len(self._base)
        if not self._base:
            return len(self._local)
        overlap = 0
        for key in self._local:
            if key in self._base:
                overlap += 1
        return len(self._local) + len(self._base) - overlap

    def pop(self, key: Any, default: Any = _MISSING) -> Any:
        """只从 local 删除；base-only 键返回 base 值但不删除（base 只读）。"""
        if key in self._local:
            return self._local.pop(key)
        if key in self._base:
            return self._base[key]
        if default is _MISSING:
            raise KeyError(key)
        return default

    def clear(self) -> None:
        self._local.clear()

    def popitem(self) -> tuple[Any, Any]:
        if self._local:
            return self._local.popitem()
        raise KeyError(
            "ReadOnlyOverlayMap: popitem with empty local and read-only base"
        )

    def copy(self) -> dict:
        """显式深拷贝当前视图（local+base 并集），用于真正需要隔离快照的场景。"""
        return {k: self[k] for k in self}

    def __repr__(self) -> str:
        return (
            f"ReadOnlyOverlayMap(local={self._local!r}, "
            f"base_keys={list(self._base)!r})"
        )
