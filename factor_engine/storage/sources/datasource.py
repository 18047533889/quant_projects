"""数据源抽象：按列名返回与引擎约定一致的 MultiIndex Series（timestamp × instrument）。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TemporalContract:
    """R24-084..087: a source's TEMPORAL contract, the single authority for
    whether a source is PIT-sensitive and how it may be joined.

    Detection must come from this contract — NEVER from a ``dataset`` name
    blacklist or a missing ``dataset`` attribute (a source without a dataset
    is ``unknown``, which is PIT-sensitive / fail-closed, not "safe").

    * ``temporal_sensitivity`` — ``none`` (plain panel), ``event`` (knowledge
      time), ``pit`` (knowledge + revision + period), ``unknown`` (fail-closed).
    * ``snapshot_capability`` — ``none`` / ``token`` / ``manifest`` / ``unknown``.
    * ``join_capability`` — ``generic_asof`` (safe for Composite merge_asof),
      ``exact_only`` (Composite may exact-align), ``pit_required`` (must go
      through a certified PIT resolver), ``unknown`` (fail-closed).
    """

    temporal_sensitivity: str = "unknown"
    snapshot_capability: str = "unknown"
    join_capability: str = "unknown"

    @property
    def pit_sensitive(self) -> bool:
        """True when generic asof must NOT be used on this source."""
        if self.join_capability == "generic_asof":
            return False
        # pit_required / exact_only / unknown → PIT-sensitive (fail-closed).
        return True


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

    def temporal_contract(self) -> TemporalContract:
        """R24-084/085: the source's declared temporal contract.

        Every DataSource MUST implement this.  The base class returns the
        fail-closed UNKNOWN contract — a source without a declared contract is
        PIT-sensitive (never "safe by omission").  Concrete sources that are
        plain panels should return ``TemporalContract(join_capability=
        "generic_asof")``.
        """
        return TemporalContract()
