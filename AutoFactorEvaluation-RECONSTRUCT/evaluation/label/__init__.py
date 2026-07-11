"""因子贴标与入库推荐子模块（4.3）"""
from .scripts import (
    run_label_pipeline, LabelModuleConfig, LabelRunResult,
    DeepSeekClient, DeepSeekConfig, default_label_module_config,
)
__all__ = [
    "run_label_pipeline", "LabelModuleConfig", "LabelRunResult",
    "DeepSeekClient", "DeepSeekConfig", "default_label_module_config",
]
