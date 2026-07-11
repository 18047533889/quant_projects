"""执行期上下文：数据源、可选列缓存、MultiIndex 层级列名。"""

from dataclasses import dataclass
from typing import Any

from storage.cache import CacheManager
from storage.datasource import DataSource


@dataclass
class ExecutionContext:
    """后端从 ``data_source`` 按列名拉数；``timestamp_col``/``instrument_col`` 与数据对齐。"""

    data_source: DataSource
    cache: CacheManager | None = None
    timestamp_col: str = "timestamp"
    instrument_col: str = "instrument"
    #: 多因子 CSE 后，预计算的共享子树结果 ``sid -> Series``（通常为 pandas）
    shared_result_cache: dict[str, Any] | None = None
    #: 可选性能提示（分块、内存上限等），见 :mod:`runtime.perf_config`
    perf: Any | None = None
