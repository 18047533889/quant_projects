"""数据源抽象：按列名返回与引擎约定一致的 MultiIndex Series（timestamp × instrument）。"""

from abc import ABC, abstractmethod
from typing import Any


def clean_execution_spec(payload: dict[str, Any]) -> dict[str, Any]:
    """剔除值为 ``None`` 的配置项（#收官轮 P0）。

    ``storage.factory._pop_option`` 把 None 值当作「未指定」——但键会留在
    options 里被 ``_ensure_no_extra_options`` 拒绝。execution_spec 必须剔除
    None 项才能被 factory 无副作用重建；``[]`` / ``""`` / ``0`` / ``False``
    等显式值保留（None 与 EMPTY 的语义区分不能丢）。
    """
    return {k: v for k, v in payload.items() if v is not None}


class DataSource(ABC):
    """数据源抽象基类，按列名返回 MultiIndex Series。

    参数:
        无
    """

    @abstractmethod
    def load_column(self, name: str) -> Any:
        """加载名为 ``name`` 的一列时间序列面板数据。

        参数:
            name: 逻辑列名

        返回:
            Any
        """
        raise NotImplementedError

    def execution_spec(self) -> dict[str, Any] | None:
        """返回可序列化、可重建（``storage.factory.build_data_source``）的执行规格。

        #收官轮 P0：程序化 ``materialize()`` 未显式传 ``data_source_config`` 时，
        orchestrator 用它还原正在执行的真实 source contract，供 lineage /
        semantic identity / full factor definition / 事件增量 rebuild 复用——
        否则 catalog 里只剩 ``{}``，重建引擎无法还原。

        具体 DataSource 子类返回完整 canonical 配置；基类无法推导时返回 ``None``
        （调用方按「无可推导配置」处理）。
        """
        return None
