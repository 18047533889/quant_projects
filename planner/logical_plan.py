"""逻辑执行计划节点（与 IR 结构相同，列表型 ``inputs`` 便于后端递归求值）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class PlanNode:
    """单棵计划子树：算子名、有序子节点、属性字典。

    字段：
        op: 算子 canonical 名（与 ``cleaned_operators`` 一致）
        inputs: 子节点列表（自左向右 / positional 顺序）
        attrs: 算子参数与元数据（窗口、列名、CSE sid 等）
        semantic_attrs: 字段目录元数据（unit/domain/frequency/available_at/
            source_vintage/flow_semantics/price_basis/universe/semantic_kind 与
            mixed_* 冲突标记），供 backend type / optimizer rewrite / SQL / PIT /
            grain / unit 校验使用。由 Lowerer 从 ``IRNode.semantic_attrs`` 拷贝，
            独立于 ``attrs`` 且不参与相等/哈希（避免改变 plan 缓存键语义）。
        node_id: 可选调试/溯源 id，不参与结构哈希
    """

    op: str
    inputs: tuple["PlanNode", ...] = field(default_factory=tuple)
    attrs: Mapping[str, Any] = field(default_factory=dict)
    semantic_attrs: Mapping[str, Any] = field(default_factory=dict, compare=False)
    node_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))
        object.__setattr__(
            self, "semantic_attrs", MappingProxyType(dict(self.semantic_attrs))
        )


def canonical_target_index(template: Any) -> Any:
    """Canonical target index 提取（R20-100..102）。

    算子结果对齐必须统一使用同一个 target-index 解析：模板可能是 ``pd.Series``
    或 ``pd.Index``（``ExecutionContext.template_index``）。所有归一分支
    （含 1-D ndarray 分支）都必须走本函数 —— 不允许某个分支直接
    ``template.index`` 而在另一个分支用 ``template`` 本身，导致 axis-only 模板
    下 1-D 结果被错误对齐到错误长度。
    """
    index_attr = getattr(template, "index", None)
    if index_attr is not None:
        return index_attr
    return template
