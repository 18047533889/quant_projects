# -*- coding: utf-8 -*-
"""算子文档与语义说明（与 runtime 实现代码分离）。"""
from factor_engine.cleaned_operators.docs.operator_doc_semantics import (
    OpDoc,
    audit_operator_docs,
    get_operator_doc,
    operator_doc_tier,
)

__all__ = ["OpDoc", "get_operator_doc", "operator_doc_tier", "audit_operator_docs"]
