"""R25 ContractIR v2 —— 运行时契约（RuntimeDatasetContract）核心数据类型。

把 datasets.yaml + COSDatasetContract + SemanticFieldCatalog + Mirror/storage 声明
编译成单一事实源 ``RuntimeDatasetContract``，运行时的 store/mirror/remote/planner
只消费它，不再各自重新解释。

模块：
    - ``physical_partition``  物理分区时钟/布局/缺失语义（P0-001/002/016）
    - ``filters``             FilterRequirement 结构化过滤契约（P0-003/004）
    - ``temporal_axis``       TemporalAxisSpec / AvailabilityResult（P0-005/017）
    - ``runtime_contract``    RuntimeDatasetContract + ContractCompiler
"""

from .physical_partition import (
    MissingPartitionSemantics,
    PhysicalLayout,
    PhysicalPartitionSpec,
)
from .filters import FilterRequirement, validate_filter_requirements
from .temporal_axis import AvailabilityResult, TemporalAxisSpec
from .runtime_contract import (
    ContractCompiler,
    RuntimeDatasetContract,
    compile_runtime_contract,
)

__all__ = [
    "PhysicalLayout",
    "MissingPartitionSemantics",
    "PhysicalPartitionSpec",
    "FilterRequirement",
    "validate_filter_requirements",
    "TemporalAxisSpec",
    "AvailabilityResult",
    "RuntimeDatasetContract",
    "ContractCompiler",
    "compile_runtime_contract",
]
