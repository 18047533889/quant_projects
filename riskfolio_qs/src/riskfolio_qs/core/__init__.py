"""核心层：数据契约与校验。"""

from .contracts import InputBundle, OutputBundle, OptimizationContext, RunMetadata
from .validator import SchemaValidator

__all__ = [
    "InputBundle",
    "OutputBundle",
    "OptimizationContext",
    "RunMetadata",
    "SchemaValidator",
]
