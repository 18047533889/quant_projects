"""trainer 包（兼容层）。

注意：本包内 trainer/trainer.py、trainer/pool.py 依赖 torch/qlib/transformers 等
重型依赖（本测试环境无 GPU wheel）。测试走 alphaprobe.trainer.checkpoint /
alphaprobe.trainer.trainer 的**模块级**导入（不触发 __init__ 全量链）。
"""

from __future__ import annotations

__all__: list[str] = []
