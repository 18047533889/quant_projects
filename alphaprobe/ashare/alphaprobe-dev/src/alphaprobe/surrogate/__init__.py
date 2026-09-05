"""surrogate 包：multi-fidelity EVI promotion（plan.md Task 17 / Part F5）。

模块：
- :mod:`features`：特征提取 Protocol + 默认实现（FE 静态分析 / L0/L1 / 参数族 /
  cluster / schema/logic / parent fitness / 历史 action success / FE/QE 成本估算）。
- :mod:`promotion`：确定性 fallback + 可注入 sklearn 兼容模型的 surrogate + EVI
  计算与 promotion 建议序。

纪律（Non-negotiable Part G）：
- #23 sealed Test 零读取：标签空间严格限定 research-caliber segment（train /
  validation）；任何 ``test`` / ``sealed`` / ``held_out`` 标记的标签直接 raise
  :class:`SealedTestLabelError`（fail-closed）。
- #7 不手算指标：只消费现成 metric_bundle 数值，不重算任何 IC/Sharpe。
- #30 全开关：learned model 分支全部由显式开关控制；无模型 / 数据不足 →
  纯确定性 fallback（回到现有 funnel 阈值语义）。
"""

from __future__ import annotations
