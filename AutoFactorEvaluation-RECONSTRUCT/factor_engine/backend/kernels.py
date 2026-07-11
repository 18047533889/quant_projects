"""算子名 → 可调用 kernel 的简单注册表（PandasBackend 在初始化时填满）。"""


class KernelRegistry:
    """字符串 ``op`` 到 ``(node, ctx) -> value`` 的映射。"""

    def __init__(self) -> None:
        self._kernels: dict[str, object] = {}

    def register(self, op: str, kernel: object) -> None:
        """注册一个算子实现。"""
        self._kernels[op] = kernel

    def get(self, op: str) -> object:
        """按 IR/Plan 的 ``op`` 取 kernel；缺失时由上层捕获。"""
        return self._kernels[op]
