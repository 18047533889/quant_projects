"""COS URI 安全路径包装器。

标准的 pathlib.Path 会吃掉 cos:// 的双斜杠变为 cos:/，
导致 admin-cos 无法识别。本模块提供安全包装。"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any


class UriPath:
    """兼容 Path 接口的 URI 安全路径包装器。

    行为:
      - 对于 cos:// 路径，以纯字符串运算保留 URI 完整性。
      - 对于普通路径，直接委托给 pathlib.Path。
      - 支持 / 拼接、str()、exists()、mkdir() 等常用操作。
    """

    def __init__(self, path: str | Path | "UriPath"):
        if isinstance(path, UriPath):
            self._str = path._str
            self._is_cos = path._is_cos
            return
        s = str(path)
        if s.startswith("cos://") or s.startswith("cos:"):
            # 确保用双斜杠
            if s.startswith("cos:/") and not s.startswith("cos://"):
                s = "cos://" + s[5:]
            self._str = s.rstrip("/")
            self._is_cos = True
        else:
            self._path = Path(s)
            self._str = str(self._path)
            self._is_cos = False

    @property
    def _p(self) -> Path:
        if not self._is_cos:
            return self._path
        # COS 路径不支持 mkdir/exists 等文件系统操作
        raise NotImplementedError(f"COS 路径不支持文件系统操作: {self._str}")

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return f"UriPath({self._str!r})"

    def __truediv__(self, other: str) -> "UriPath":
        if self._is_cos:
            return UriPath(self._str + "/" + str(other).lstrip("/"))
        return UriPath(self._path / other)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, UriPath):
            return self._str == other._str
        return self._str == str(other)

    def __hash__(self) -> int:
        return hash(self._str)

    @property
    def parent(self) -> "UriPath":
        if self._is_cos:
            idx = self._str.rfind("/")
            if idx <= 6:  # cos:// 之后没有 /
                return UriPath(self._str)
            return UriPath(self._str[:idx])
        return UriPath(self._path.parent)

    def exists(self) -> bool:
        if self._is_cos:
            from cos_utils import CosPath
            return CosPath(self._str).exists()
        return self._path.exists()

    def mkdir(self, parents: bool = False, exist_ok: bool = False):
        if not self._is_cos:
            self._path.mkdir(parents=parents, exist_ok=exist_ok)
            return
        from cos_utils import CosPath
        CosPath(self._str).ensure_dir()

    def resolve(self) -> "UriPath":
        if self._is_cos:
            return self
        return UriPath(self._path.resolve())

    def is_dir(self) -> bool:
        if self._is_cos:
            from cos_utils import CosPath
            return CosPath(self._str).exists()
        return self._path.is_dir()

    def iterdir(self):
        if self._is_cos:
            from cos_utils import CosPath
            for entry in CosPath(self._str).list_dir():
                yield UriPath(self._str + "/" + entry["key"])
        else:
            for p in self._path.iterdir():
                yield UriPath(p)

    def read_text(self, encoding: str = "utf-8") -> str:
        if self._is_cos:
            raise NotImplementedError("COS 路径暂不支持 read_text")
        return self._path.read_text(encoding=encoding)
