"""Restricted factor DSL parser with orthogonal surface and dialect controls.

Round-11 long-tail: the parser is STRICTLY a syntax layer.
  * #16 — string literals are never coerced at parse time (``"000001"`` stays
    ``"000001"``); controlled numeric-string conversion happens against a
    declared numeric ``ParamSpec`` at the operator call site / runtime binder.
  * #17 — non-finite numeric literals (``1e309`` -> inf, ``-inf``, ``nan``) are
    rejected at the DSL boundary.
  * #18 — a configurable :class:`ComplexityBudget` bounds AST size / depth /
    call arity / literal magnitude so GP/LLM-generated pathological formulas
    fail at parse, not at compute time.
"""
from __future__ import annotations
import ast
import math
from dataclasses import dataclass
from typing import Any
from api.columns import field
from api.factor import Factor
from api.operator_registry import build_dsl_allowlist
from expr.base import Expr

class DSLParseError(ValueError): pass


@dataclass(frozen=True)
class ComplexityBudget:
    """Round-11 #18: parse-time bounds on expression size/complexity.

    Guards the automatic mining / LLM path against pathological formulas
    (10k-node trees, huge variadic nodes, absurd literals) before they burn
    compute budget.
    """
    max_ast_nodes: int = 256
    max_depth: int = 32
    max_call_arity: int = 12
    max_literal_magnitude: float = 1e9
    max_variadic_inputs: int = 32


class _ExprBuilder:
    def __init__(self,*,surface:str="daily",dialect:str="native",dialect_version:str|None=None,
                 budget:ComplexityBudget|None=None)->None:
        if surface=="lqtp" and dialect=="native":dialect="lqtp"
        self._surface=str(surface or "daily");self._dialect=str(dialect or "native").lower();self._dialect_version=dialect_version
        self._allowed=build_dsl_allowlist(surface=self._surface,dialect=self._dialect,dialect_version=self._dialect_version)
        self._budget=budget or ComplexityBudget()
        self._nodes=0
        self._depth=0
    def _enter(self,node:ast.AST)->None:
        self._nodes+=1
        if self._nodes>self._budget.max_ast_nodes:
            raise DSLParseError(
                f"expression exceeds ComplexityBudget.max_ast_nodes={self._budget.max_ast_nodes}"
            )
        self._depth+=1
        if self._depth>self._budget.max_depth:
            self._depth-=1
            raise DSLParseError(
                f"expression exceeds ComplexityBudget.max_depth={self._budget.max_depth}"
            )
    def _exit(self)->None:
        self._depth-=1
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
        self._enter(node)
        try:
            return self._dispatch(node)
        finally:
            self._exit()
    def _dispatch(self,node:ast.AST)->Any:
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
            if node.value is None or isinstance(node.value,bool):
                return node.value
            if isinstance(node.value,(int,float)) and not isinstance(node.value,bool):
                # Round-11 #17: non-finite numeric literals are rejected at the
                # DSL boundary (``1e309`` parses to ``inf``; NaN/±Inf never enter
                # a factor expression).
                if isinstance(node.value,float) and not math.isfinite(node.value):
                    raise DSLParseError(f"Non-finite numeric literal is not allowed: {node.value!r}")
                if not isinstance(node.value,bool) and abs(float(node.value))>self._budget.max_literal_magnitude:
                    raise DSLParseError(
                        f"literal magnitude {node.value!r} exceeds ComplexityBudget."
                        f"max_literal_magnitude={self._budget.max_literal_magnitude}"
                    )
                return node.value
            if isinstance(node.value,str):
                # Round-11 #16: strings are NEVER coerced at the parser.  A
                # numeric-looking string is a string (``"000001"``, ``"2024Q1"``,
                # a category/version/security code); a declared numeric ParamSpec
                # does the controlled conversion at the operator call site.
                return node.value
            raise DSLParseError(f"Unsupported literal: {node.value!r}")
        if isinstance(node,ast.Name):
            if node.id in self._allowed:raise DSLParseError(f"Bare name {node.id!r} is not a column reference; use {node.id}(...) for operators.")
            if not _is_field_identifier(node.id):raise DSLParseError(f"Unsupported name: {node.id}")
            return field(node.id, strict=False)
        if isinstance(node,ast.Attribute):raise DSLParseError("Unsupported data-source attribute. Under dialect='lqtp', only registered DataTable.Field or parameterized DataTable(...).Field references are accepted.")
        raise DSLParseError(f"Unsupported syntax node: {type(node).__name__}")
    def _visit_call(self,node:ast.Call)->Any:
        # Round-11 #18: arity budget at the AST boundary (before recursion) so a
        # giant variadic call is rejected without visiting every argument node.
        if len(node.args)>self._budget.max_variadic_inputs:
            raise DSLParseError(
                f"call arity {len(node.args)} exceeds ComplexityBudget."
                f"max_variadic_inputs={self._budget.max_variadic_inputs}"
            )
        if len(node.args)+len(node.keywords)>self._budget.max_call_arity:
            raise DSLParseError(
                f"call arity {len(node.args)+len(node.keywords)} exceeds "
                f"ComplexityBudget.max_call_arity={self._budget.max_call_arity}"
            )
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
        if isinstance(node.func,ast.Name):
            # Round-11 #16: controlled numeric-string conversion against the
            # operator's DECLARED ParamSpec.  String parameters (codes, category
            # ids, version ids) keep exact strings; numeric parameters coerce
            # "20" -> 20 at the binder, never the parser.
            args,kwargs=_coerce_call_literals(name,args,kwargs)
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

def _coerce_call_literals(name:str,args:tuple,kwargs:dict):
    """Round-11 #16: contract-driven numeric-string conversion at a call site.

    A string literal is coerced to ``int``/``float`` ONLY when the named
    operator's declared contract says the target parameter is numeric
    (``ParamSpec(dtype=int|float)`` or a numeric ``param_types`` entry).  String
    parameters — category/version/security codes, enum choices — keep the exact
    string.  Operators without a resolvable metadata (not yet loaded) are left
    untouched: the runtime binder applies the identical rule.
    """
    str_args=[isinstance(a,str) for a in args]
    str_kwargs={k:(isinstance(v,str)) for k,v in kwargs.items()}
    if not any(str_args) and not any(str_kwargs.values()):
        return args,kwargs
    try:
        from cleaned_operators.base import _coerce_declared_numeric_string
        from cleaned_operators.registry import OperatorRegistry
    except Exception:
        return args,kwargs
    op=None
    try:
        op=OperatorRegistry.get(name)
    except Exception:
        op=None
    if op is None:
        try:
            canon=OperatorRegistry.resolve_canonical(name)
            if canon!=name:op=OperatorRegistry.get(canon)
        except Exception:
            op=None
    if op is None:
        return args,kwargs
    meta=getattr(op,"metadata",None)
    if meta is None:
        return args,kwargs
    names=list(getattr(meta,"param_names",None) or ())
    specs=getattr(meta,"param_specs",None) or {}
    types=getattr(meta,"param_types",None) or {}
    new_args=list(args)
    for i,a in enumerate(args):
        if isinstance(a,str):
            pname=names[i] if i<len(names) else ""
            new_args[i]=_coerce_declared_numeric_string(a,pname,types.get(pname),specs.get(pname))
    new_kwargs=dict(kwargs)
    for k,v in kwargs.items():
        if isinstance(v,str):
            new_kwargs[k]=_coerce_declared_numeric_string(v,k,types.get(k),specs.get(k))
    return tuple(new_args),new_kwargs


def _is_field_identifier(name:str)->bool:return bool(name) and not name[0].isdigit() and all(c.isalnum() or c=="_" for c in name)
def parse_expr(text:str,*,surface:str="daily",dialect:str="native",dialect_version:str|None=None,budget:ComplexityBudget|None=None)->Expr:return _ExprBuilder(surface=surface,dialect=dialect,dialect_version=dialect_version,budget=budget).build(text)
def parse_factor(text:str,*,name:str="factor",freq:str="1d",universe:str|None=None,description:str|None=None,surface:str="daily",dialect:str="native",dialect_version:str|None=None,budget:ComplexityBudget|None=None)->Factor:return Factor(name=name,expr=parse_expr(text,surface=surface,dialect=dialect,dialect_version=dialect_version,budget=budget),freq=freq,universe=universe,description=description,source_expr=text)
