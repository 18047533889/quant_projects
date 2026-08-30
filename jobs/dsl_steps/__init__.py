"""DSL 计算步骤生成器（详情页「计算步骤」区块）。

职责：把 factor 的 dsl 解析成一步步可读的中文计算步骤。
分层策略（由高到低优先级）：
  1. 手写步骤（factor_manual_61 61 条 / formula_lqtp.json 62 条）
  2. 结构化翻译（pandas code 语句级，能覆盖绝大多数 fallback DSL）
  3. 算子正则式（对 DSL 中出现的算子做逐个释义，兜底）
"""
from .generator import build_steps_block, resolve_steps
