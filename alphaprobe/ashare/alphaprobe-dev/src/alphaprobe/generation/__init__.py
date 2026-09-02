"""alphaprobe.generation（任务书 §27 / §37）：结构化 generation。

- ``StructuredGenerator``：action JSON → AST Transform 构造器 → FE validate →
  候选池（非法 action / 非白名单 op / FE 校验失败一律拒绝并记录原因）；
- ``normalize_llm_output``：llm_fn 契约升级为返回结构化对象；旧文本输出走
  identity action fallback（向后兼容，打日志）；
- ``make_structured_stub_llm_fn``：确定性、可种子复现的 action JSON stub；
- ``fe_operator_surface``：FE 已注册算子表面（daily allowlist），运行时校验用。
"""

from alphaprobe.generation.structured import (
    DEFAULT_ACTION_OP_WHITELIST,
    StructuredGenerator,
    build_formula,
    fe_operator_surface,
    make_structured_stub_llm_fn,
    normalize_llm_output,
    validate_action,
)

__all__ = [
    "DEFAULT_ACTION_OP_WHITELIST",
    "StructuredGenerator",
    "build_formula",
    "fe_operator_surface",
    "make_structured_stub_llm_fn",
    "normalize_llm_output",
    "validate_action",
]
