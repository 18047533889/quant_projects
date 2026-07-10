"""算子 kernel 注册表。

PandasBackend 在初始化时向本模块的 ``KernelRegistry`` 批量注册
``column`` / ``literal`` 内建 kernel 以及 ``cleaned_operators`` 桥接 kernel，
执行期按逻辑计划节点的 ``op`` 字段分派到对应可调用实现。
"""


class KernelRegistry:
    """字符串算子名 ``op`` 到 ``(node, ctx) -> value`` kernel 的映射表。

    由 ``PandasBackend._register_kernels`` 在启动阶段填满；
    ``_eval`` 执行时通过 ``get`` 查找并调用。
    """

    def __init__(self) -> None:
        """初始化空的 kernel 映射字典。"""
        self._kernels: dict[str, object] = {}

    def register(self, op: str, kernel: object) -> None:
        """注册一个算子实现。

        参数
        ----
        op : str
            逻辑计划节点 ``PlanNode.op`` 名称。
        kernel : object
            可调用对象，签名为 ``(node: PlanNode, ctx: ExecutionContext) -> Any``。
        """
        self._kernels[op] = kernel

    def get(self, op: str) -> object:
        """按 IR/Plan 的 ``op`` 字段取 kernel。

        参数
        ----
        op : str
            逻辑计划节点算子名。

        返回
        ----
        object
            已注册的 kernel 可调用对象。

        异常
        ----
        KeyError
            算子名未注册时由 ``dict`` 抛出，上层 ``_eval`` 会转为 ``NotImplementedError``。
        """
        return self._kernels[op]
