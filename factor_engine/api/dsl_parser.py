"""Restricted factor DSL parser with orthogonal surface and dialect controls."""
from __future__ import annotations
import ast
from typing import Any
from api.columns import field
from api.factor import Factor
from api.operator_registry import build_dsl_allowlist
from expr.base import Expr

class DSLParseError(ValueError): pass

class _ExprBuilder:
    def __init__(self,*,surface:str="daily",dialect:str="native",dialect_version:str|None=None)->None:
        if surface=="lqtp" and dialect=="native":dialect="lqtp"
        self._surface=str(surface or "daily");self._dialect=str(dialect or "native").lower();self._dialect_version=dialect_version
        self._allowed=build_dsl_allowlist(surface=self._surface,dialect=self._dialect,dialect_version=self._dialect_version)
    def build(self,text:str)->Expr:
        normalized=str(text)
        if self._dialect=="lqtp" or self._surface=="lqtp":
            from api.lqtp_compat import normalize_lqtp_formula
            from api.derived_field_compat import normalize_lqtp_derived_fields
            normalized=normalize_lqtp_derived_fields(normalize_lqtp_formula(normalized))
        try:parsed=ast.parse(normalized,mode="eval")
        except SyntaxError as exc:raise DSLParseError(f"Invalid expression syntax: {text}") from exc
        expr=self._visit(parsed.body)
        if not isinstance(expr,Expr):raise DSLParseError("Expression must evaluate to an Expr object.")
        return expr
    def _visit(self,node:ast.AST)->Any:
        if isinstance(node,ast.Call):return self._visit_call(node)
        if isinstance(node,ast.Compare):return self._visit_compare(node)
        if isinstance(node,ast.BinOp):return self._visit_binop(node)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -self._visit(node.operand)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.UAdd):return self._visit(node.operand)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Invert):
            from api.cleaned_ops import make_cleaned_call_factory
            return make_cleaned_call_factory("not_")(self._visit(node.operand))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):
            from api.cleaned_ops import make_cleaned_call_factory
            return make_cleaned_call_factory("not_")(self._visit(node.operand))
        if isinstance(node,ast.Constant):
            if isinstance(node.value,(str,int,float,bool)) or node.value is None:return node.value
            raise DSLParseError(f"Unsupported literal: {node.value!r}")
        if isinstance(node,ast.Name):
            if node.id in self._allowed:raise DSLParseError(f"Bare name {node.id!r} is not a column reference; use {node.id}(...) for operators.")
            if not _is_field_identifier(node.id):raise DSLParseError(f"Unsupported name: {node.id}")
            return field(node.id, strict=False)
        if isinstance(node,ast.Attribute):raise DSLParseError("Unsupported data-source attribute. Under dialect='lqtp', only registered DataTable.Field or parameterized DataTable(...).Field references are accepted.")
        raise DSLParseError(f"Unsupported syntax node: {type(node).__name__}")
    def _visit_call(self,node:ast.Call)->Any:
        if isinstance(node.func,ast.Name):
            name=node.func.id
            if name not in self._allowed:raise DSLParseError(f"Unsupported function: {name}")
            func=self._allowed[name]
        else:func=self._visit(node.func)
        args=[self._visit(arg) for arg in node.args];kwargs={}
        for kw in node.keywords:
            if kw.arg is None:raise DSLParseError("Keyword-only **kwargs are not supported.")
            kwargs[kw.arg]=self._visit(kw.value)
        if not callable(func):raise DSLParseError("Call target is not callable.")
        try:return func(*args,**kwargs)
        except (ValueError,TypeError) as exc:raise DSLParseError(str(exc)) from exc
    def _visit_compare(self,node:ast.Compare)->Expr:
        if len(node.ops)!=1 or len(node.comparators)!=1:raise DSLParseError("Chained comparisons are not supported; use one comparison only.")
        left,right,op=self._visit(node.left),self._visit(node.comparators[0]),node.ops[0]
        from api.cleaned_ops import make_cleaned_call_factory
        mapping={ast.Lt:"lt",ast.LtE:"le",ast.Eq:"eq",ast.Gt:"gt",ast.GtE:"ge",ast.NotEq:"ne"}
        for kind,canonical in mapping.items():
            if isinstance(op,kind):return make_cleaned_call_factory(canonical)(left,right)
        raise DSLParseError(f"Unsupported comparison: {type(op).__name__}")
    def _visit_binop(self,node:ast.BinOp)->Any:
        left,right=self._visit(node.left),self._visit(node.right)
        if isinstance(node.op,ast.Add):return left+right
        if isinstance(node.op,ast.Sub):return left-right
        if isinstance(node.op,ast.Mult):return left*right
        if isinstance(node.op,ast.Div):return left/right
        if isinstance(node.op,ast.Pow):
            from api.cleaned_ops import make_cleaned_call_factory
            return make_cleaned_call_factory("power")(left,right)
        # CogAlpha pandas code uses bitwise & | ~ for elementwise boolean logic;
        # map onto the daily logical operators and_/or_/not_.
        if isinstance(node.op,ast.BitAnd):
            from api.cleaned_ops import make_cleaned_call_factory
            return make_cleaned_call_factory("and_")(left,right)
        if isinstance(node.op,ast.BitOr):
            from api.cleaned_ops import make_cleaned_call_factory
            return make_cleaned_call_factory("or_")(left,right)
        raise DSLParseError(f"Unsupported binary operator: {type(node.op).__name__}")

def _is_field_identifier(name:str)->bool:return bool(name) and not name[0].isdigit() and all(c.isalnum() or c=="_" for c in name)
def parse_expr(text:str,*,surface:str="daily",dialect:str="native",dialect_version:str|None=None)->Expr:return _ExprBuilder(surface=surface,dialect=dialect,dialect_version=dialect_version).build(text)
def parse_factor(text:str,*,name:str="factor",freq:str="1d",universe:str|None=None,description:str|None=None,surface:str="daily",dialect:str="native",dialect_version:str|None=None)->Factor:return Factor(name=name,expr=parse_expr(text,surface=surface,dialect=dialect,dialect_version=dialect_version),freq=freq,universe=universe,description=description,source_expr=text)
