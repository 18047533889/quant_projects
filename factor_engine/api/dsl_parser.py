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
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping
from factor_engine.api.columns import field
from factor_engine.api.factor import Factor
from factor_engine.api.operator_registry import build_dsl_allowlist
from factor_engine.expr.base import Expr

class DSLParseError(ValueError): pass

class DSLUnknownOperatorError(DSLParseError):
    """A function call names an operator absent from the selected allowlist."""


@dataclass(frozen=True)
class ComplexityBudget:
    """Round-11 #18: parse-time bounds on expression size/complexity.

    Guards the automatic mining / LLM path against pathological formulas
    (10k-node trees, huge variadic nodes, absurd literals) before they burn
    compute budget.

    R21-037..043 adds *pre-parse* size limits: ``ast.parse`` happens only after
    the raw text passes ``max_formula_bytes/max_formula_chars``, a string
    literal is capped before it becomes one giant AST Constant, and
    identifier/keyword length limits prevent pathological names.
    """
    max_ast_nodes: int = 256
    max_depth: int = 32
    max_call_arity: int = 12
    max_literal_magnitude: float = 1e9
    max_variadic_inputs: int = 32
    # R21-037..041
    max_formula_bytes: int = 65536
    max_formula_chars: int = 65536
    max_string_literal_bytes: int = 8192
    max_identifier_length: int = 256
    max_keyword_length: int = 128


def _native_source_col(*items: Any) -> Any:
    """Strict parameterized source reference for native research formulas.

    This is intentionally narrower than arbitrary function execution: every
    table/field and identity keys are string literals (rank is an integer), and the table/field must exist in the frozen
    A-share catalog, and every table-required identity parameter must be
    present.  Benchmark index bars additionally require ``index`` because the
    physical table contains many index instruments.
    """
    if len(items) < 2 or len(items) % 2:
        raise ValueError(
            "source_col requires table, field, then zero or more declared key/value literal pairs"
        )
    if not all(isinstance(item, str) for item in items[:2]):
        raise TypeError("source_col table and field must be string literals")
    table, field_name = items[:2]
    pairs = items[2:]
    params: dict[str, Any] = {}
    for index in range(0, len(pairs), 2):
        key, value = pairs[index], pairs[index + 1]
        if not isinstance(key,str) or not key or key in params:
            raise ValueError(f"source_col duplicate, non-string or empty parameter {key!r}")
        if key in {"ShareholderRank", "rank"}:
            if isinstance(value,bool) or not isinstance(value,int) or not 1 <= value <= 10:
                raise ValueError("source_col shareholder rank must be integer 1..10")
        elif not isinstance(value,str):
            raise TypeError("source_col identity values must be string literals")
        params[key] = value
    from factor_engine.fields import FIELD_REGISTRY
    table_spec = FIELD_REGISTRY.resolve_table(table)
    if table_spec is None:
        raise ValueError(f"source_col unknown table {table!r}")
    FIELD_REGISTRY.require(field_name, table=table_spec.name)
    # R58 (2026-09-21): adjusted-only gate for the native research DSL surface.
    # Reject unadjusted A-share OHLCV (price_basis == "RAW") read from
    # StockDailyBar; the back-adjusted StockDailyBarAdj is the authoritative
    # price basis. Limit prices (RAW_OFFICIAL_LIMIT) and Factor/Volume/
    # amount/ret remain permitted. Mirrors the guard in
    # lqtp_compat._guarded_source_col so the policy holds on every surface,
    # including the native research surface used by ``source_col`` here.
    if str(table_spec.name) == "StockDailyBar":
        try:
            from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY
            _spec = MULTI_MARKET_FIELD_REGISTRY.registry_for("ashare").get(
                str(field_name), table="StockDailyBar", strict=False)
            if _spec is not None and getattr(_spec, "price_basis", None) == "RAW":
                raise ValueError(
                    f"source_col('StockDailyBar', {field_name!r}) is an UNADJUSTED "
                    f"price column (price_basis=RAW) and must not be read by the "
                    f"factor engine. Use source_col('StockDailyBarAdj', {field_name!r}) "
                    f"(back-adjusted) instead."
                )
        except ValueError:
            raise
        except Exception:
            pass
    required = set(table_spec.required_parameters)
    if table_spec.name == "IndexDailyBar":
        required.add("index")
    missing = sorted(required - set(params))
    if missing:
        raise ValueError(
            f"source_col {table!r} requires exact parameters {missing}"
        )
    allowed = set(required)
    if table_spec.name in {"StockTopTenShareholder","StockTopTenFloatShareholder"}:
        allowed.update({"ShareholderRank","rank"})
        selected=[params[k] for k in ("ShareholderRank","rank") if k in params]
        if not selected or len(set(selected)) != 1:
            raise ValueError("source_col shareholder table requires one exact ShareholderRank=1..10")
    unexpected = sorted(set(params) - allowed)
    if unexpected:
        raise ValueError(
            f"source_col {table!r} received undeclared parameters {unexpected}"
        )
    from factor_engine.api.source_ref import source_col
    return source_col(table, field_name, dialect="lqtp", **params)


def _native_holder_top10_daily_id_churn(table: str) -> Any:
    """Bounded macro: 20 ranked observations and their previous-session values.

    This measures disclosure changes on the daily decision grid, not a
    forward-filled last-report churn estimate. Expansion is always exactly
    ten ranks, independent of user-supplied data or expression size.
    """
    if table not in {"StockTopTenShareholder","StockTopTenFloatShareholder"}:
        raise ValueError("holder_top10_daily_id_churn requires an official top-ten table")
    allowed=build_dsl_allowlist(surface="compat_research",dialect="native")
    current=[_native_source_col(table,field_name,"ShareholderRank",rank)
             for field_name in ("ShareRatio","ShareholderId") for rank in range(1,11)]
    previous=[allowed["delay"](value,1) for value in current]
    return allowed["holder_id_matched_churn"](*current,*previous)


def _native_holder_top10_id_stat(table: str, statistic: str) -> Any:
    if table not in {"StockTopTenShareholder","StockTopTenFloatShareholder"}:
        raise ValueError("holder top-ten ID statistic requires an official top-ten table")
    allowed=build_dsl_allowlist(surface="compat_research",dialect="native")
    current=[_native_source_col(table,field_name,"ShareholderRank",rank)
             for field_name in ("ShareRatio","ShareholderId") for rank in range(1,11)]
    if statistic=="disclosure_count":
        return allowed["holder_disclosure_count"](*current)
    if statistic=="two_day_rank_migration":
        previous=[allowed["delay"](value,2) for value in current]
        return allowed["holder_share_weighted_rank_migration"](*current,*previous)
    raise ValueError("unknown top-ten ID statistic")


def _native_holder_top10_stat(table: str, statistic: str) -> Any:
    """Fixed ten-rank macros; expand into governed primitives, not fake backends."""
    if table not in {"StockTopTenShareholder","StockTopTenFloatShareholder"}:
        raise ValueError("holder top-ten statistic requires an official top-ten table")
    allowed=build_dsl_allowlist(surface="compat_research",dialect="native")
    def op(name,*args):return allowed[name](*args)
    def ranked(name):return [_native_source_col(table,name,"ShareholderRank",rank) for rank in range(1,11)]
    def total(values):
        value=values[0]
        for item in values[1:]:value=op("add",value,item)
        return value
    amounts=ranked("ShareNumber")
    if statistic=="pledge_ratio":
        return op("safe_div_null",total(ranked("SharePledge")),total(amounts))
    if statistic=="weighted_std":
        weights=ranked("ShareRatio")
        denom=total(weights)
        mean=op("safe_div_null",total([op("multiply",x,w) for x,w in zip(amounts,weights)]),denom)
        variance=op("safe_div_null",total([op("multiply",w,op("square",op("subtract",x,mean))) for x,w in zip(amounts,weights)]),denom)
        return op("sqrt",variance)
    raise ValueError("unknown top-ten statistic")


class _ExprBuilder:
    def __init__(self,*,surface:str="daily",dialect:str="native",dialect_version:str|None=None,
                 budget:ComplexityBudget|None=None,
                 allowed:Mapping[str,Callable[...,Any]]|None=None)->None:
        if surface=="lqtp" and dialect=="native":dialect="lqtp"
        self._surface=str(surface or "daily");self._dialect=str(dialect or "native").lower();self._dialect_version=dialect_version
        if allowed is not None:
            self._allowed = allowed
        else:
            built = dict(build_dsl_allowlist(
                surface=self._surface,dialect=self._dialect,
                dialect_version=self._dialect_version
            ))
            if self._dialect == "native" and self._surface in {"compat_research", "all"}:
                built["source_col"] = _native_source_col
                built["holder_top10_daily_id_churn"] = _native_holder_top10_daily_id_churn
                built["holder_top10_weighted_std"] = lambda table: _native_holder_top10_stat(table, "weighted_std")
                built["holder_top10_pledge_ratio"] = lambda table: _native_holder_top10_stat(table, "pledge_ratio")
                built["holder_top10_disclosure_count"] = lambda table: _native_holder_top10_id_stat(table, "disclosure_count")
                built["holder_top10_two_day_rank_migration"] = lambda table: _native_holder_top10_id_stat(table, "two_day_rank_migration")
            self._allowed = built
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
    def _validate_text_budget(self, text: str, *, stage: str = "expression") -> None:
        # UTF-8 has at least one byte per character. Check this lower bound
        # before encoding so even a huge rejected string needs no huge copy.
        if len(text) > self._budget.max_formula_bytes:
            raise DSLParseError(
                f"{stage} exceeds ComplexityBudget.max_formula_bytes={self._budget.max_formula_bytes}"
            )
        if len(text) > self._budget.max_formula_chars:
            raise DSLParseError(
                f"{stage} chars exceed ComplexityBudget.max_formula_chars={self._budget.max_formula_chars}"
            )
        try:
            size = len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise DSLParseError(f"{stage} contains invalid Unicode") from exc
        if size > self._budget.max_formula_bytes:
            raise DSLParseError(
                f"{stage} bytes {size} exceed ComplexityBudget.max_formula_bytes={self._budget.max_formula_bytes}"
            )

    def build(self,text:str)->Expr:
        normalized=str(text)
        # Bound raw and normalized text before ast.parse allocates.
        self._validate_text_budget(normalized)
        if self._dialect=="lqtp" or self._surface=="lqtp":
            from factor_engine.api.lqtp_compat import normalize_lqtp_formula
            from factor_engine.api.derived_field_compat import normalize_lqtp_derived_fields
            normalized=normalize_lqtp_derived_fields(normalize_lqtp_formula(normalized))
            self._validate_text_budget(normalized, stage="normalized expression (LQTP expansion)")
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
        if isinstance(node,ast.BoolOp):
            # LQTP formulas carry ``A and B`` / ``A or B`` (the platform DSL
            # uses the python keywords).  Map to the same elementwise
            # and_/or_ operators the bitwise path uses.
            from factor_engine.api.cleaned_ops import make_cleaned_call_factory
            left=self._visit(node.values[0])
            for value in node.values[1:]:
                right=self._visit(value)
                if isinstance(node.op,ast.And):
                    left=make_cleaned_call_factory("and_")(left,right)
                elif isinstance(node.op,ast.Or):
                    left=make_cleaned_call_factory("or_")(left,right)
            return left
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -self._visit(node.operand)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.UAdd):return self._visit(node.operand)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Invert):
            from factor_engine.api.cleaned_ops import make_cleaned_call_factory
            return make_cleaned_call_factory("not_")(self._visit(node.operand))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):
            from factor_engine.api.cleaned_ops import make_cleaned_call_factory
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
                if not isinstance(node.value,bool) and abs(node.value) > self._budget.max_literal_magnitude:
                    raise DSLParseError(
                        f"literal magnitude {node.value!r} exceeds ComplexityBudget."
                        f"max_literal_magnitude={self._budget.max_literal_magnitude}"
                    )
                return node.value
            if isinstance(node.value,str):
                # R21-040: a several-MB string is still one AST Constant — cap
                # literal bytes so a huge string cannot bypass the node budget.
                try:
                    literal_bytes = len(node.value.encode("utf-8"))
                except UnicodeEncodeError as exc:
                    raise DSLParseError("string literal contains invalid Unicode") from exc
                if literal_bytes >self._budget.max_string_literal_bytes:
                    raise DSLParseError(
                        f"string literal bytes {literal_bytes} exceed "
                        f"ComplexityBudget.max_string_literal_bytes="
                        f"{self._budget.max_string_literal_bytes}"
                    )
                # Round-11 #16: strings are NEVER coerced at the parser.  A
                # numeric-looking string is a string (``"000001"``, ``"2024Q1"``,
                # a category/version/security code); a declared numeric ParamSpec
                # does the controlled conversion at the operator call site.
                return node.value
            raise DSLParseError(f"Unsupported literal: {node.value!r}")
        if isinstance(node,ast.Name):
            if node.id in self._allowed:raise DSLParseError(f"Bare name {node.id!r} is not a column reference; use {node.id}(...) for operators.")
            if not _is_field_identifier(node.id):raise DSLParseError(f"Unsupported name: {node.id}")
            if len(node.id)>self._budget.max_identifier_length:
                raise DSLParseError(
                    f"identifier length {len(node.id)} exceeds "
                    f"ComplexityBudget.max_identifier_length={self._budget.max_identifier_length}"
                )
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
        aliases:dict[str,str]={}
        param_names:tuple[str,...]=()
        if isinstance(node.func,ast.Name):
            name=node.func.id
            if len(name)>self._budget.max_keyword_length:
                raise DSLParseError(
                    f"keyword length {len(name)} exceeds "
                    f"ComplexityBudget.max_keyword_length={self._budget.max_keyword_length}"
                )
            if name not in self._allowed:
                raise DSLUnknownOperatorError(f"Unsupported function: {name}")
            func=self._allowed[name]
            total_arity=len(node.args)+len(node.keywords)
            if (node.keywords or total_arity>self._budget.max_call_arity
                    or any(isinstance(arg,(ast.Tuple,ast.List)) for arg in node.args)):
                aliases,param_names=_call_parameter_binding(name)
            if "..." in param_names:
                if len(node.keywords)>self._budget.max_call_arity:
                    raise DSLParseError(
                        f"keyword arity {len(node.keywords)} exceeds ComplexityBudget."
                        f"max_call_arity={self._budget.max_call_arity}"
                    )
            elif total_arity>self._budget.max_call_arity:
                raise DSLParseError(
                    f"call arity {total_arity} exceeds ComplexityBudget."
                    f"max_call_arity={self._budget.max_call_arity}"
                )
        else:
            if len(node.args)+len(node.keywords)>self._budget.max_call_arity:
                raise DSLParseError(
                    f"call arity {len(node.args)+len(node.keywords)} exceeds "
                    f"ComplexityBudget.max_call_arity={self._budget.max_call_arity}"
                )
            func=self._visit(node.func)
        positional_names=param_names[:param_names.index("...")] if "..." in param_names else param_names
        args=[]
        for index,arg in enumerate(node.args):
            pname=positional_names[index] if index<len(positional_names) else ""
            args.append(self._visit_call_argument(arg,name,pname))
        kwargs={}
        seen_keywords:set[str]=set()
        # Parameters after a variadic input marker are keyword-only. Multiple
        # series inputs do not positionally consume e.g. row_sum's min_count.
        bound_canonical=set(positional_names[:len(node.args)])
        for kw in node.keywords:
            if kw.arg is None:raise DSLParseError("Keyword-only **kwargs are not supported.")
            canonical_kw=aliases.get(kw.arg,kw.arg)
            if kw.arg in seen_keywords or canonical_kw in bound_canonical:
                raise DSLParseError(f"Duplicate parameter {canonical_kw!r} is not allowed.")
            seen_keywords.add(kw.arg)
            bound_canonical.add(canonical_kw)
            kwargs[kw.arg]=self._visit_call_argument(kw.value,name,canonical_kw)
        if isinstance(node.func,ast.Name):
            # R59: under-called declared-signature ops fail at parse time
            # (same acceptance set as the runtime binder — see helper).
            try:
                _g_aliases,_g_param_names=_call_parameter_binding(name)
            except Exception:
                _g_param_names=()
            if _g_param_names and "..." not in _g_param_names:
                _missing=_missing_required_params(
                    name,tuple(_g_param_names[:len(node.args)]),set(kwargs)
                )
                if _missing:
                    raise DSLParseError(
                        f"{name}: missing required parameter(s) {_missing} — declared "
                        f"signature {list(_g_param_names)}"
                    )
        if not callable(func):raise DSLParseError("Call target is not callable.")
        if isinstance(node.func,ast.Name):
            # Round-11 #16: controlled numeric-string conversion against the
            # operator's DECLARED ParamSpec.  String parameters (codes, category
            # ids, version ids) keep exact strings; numeric parameters coerce
            # "20" -> 20 at the binder, never the parser.
            args,kwargs=_coerce_call_literals(name,args,kwargs)
        try:return func(*args,**kwargs)
        except (ValueError,TypeError) as exc:raise DSLParseError(str(exc)) from exc
    def _visit_call_argument(self,node:ast.AST,operator_name:str,param_name:str)->Any:
        if not isinstance(node,(ast.Tuple,ast.List)):
            return self._visit(node)
        expected=tuple if isinstance(node,ast.Tuple) else list
        spec=_declared_sequence_spec(operator_name,param_name,expected)
        if spec is None:
            raise DSLParseError(
                f"{operator_name}.{param_name or '?'} does not declare a "
                f"{expected.__name__} scalar parameter"
            )
        if len(node.elts)>self._budget.max_variadic_inputs:
            raise DSLParseError(
                f"sequence length {len(node.elts)} exceeds ComplexityBudget."
                f"max_variadic_inputs={self._budget.max_variadic_inputs}"
            )
        max_items=getattr(spec,"max_items",None)
        min_items=getattr(spec,"min_items",None)
        if max_items is not None and len(node.elts)>int(max_items):
            raise DSLParseError(f"{operator_name}.{param_name} exceeds max_items={max_items}")
        if min_items is not None and len(node.elts)<int(min_items):
            raise DSLParseError(f"{operator_name}.{param_name} requires min_items={min_items}")
        self._enter(node)
        try:
            values=[]
            for element in node.elts:
                if isinstance(element,(ast.Tuple,ast.List,ast.Set,ast.Dict,
                                       ast.ListComp,ast.SetComp,ast.DictComp,
                                       ast.GeneratorExp,ast.Name,ast.Call,ast.Attribute)):
                    raise DSLParseError(
                        "sequence parameters accept flat scalar literals only"
                    )
                value=self._visit(element)
                if isinstance(value,Expr) or not isinstance(value,(type(None),bool,int,float,str)):
                    raise DSLParseError(
                        "sequence parameters accept flat scalar literals only"
                    )
                values.append(value)
            return tuple(values) if expected is tuple else values
        finally:
            self._exit()
    def _visit_compare(self,node:ast.Compare)->Expr:
        if len(node.ops)!=1 or len(node.comparators)!=1:raise DSLParseError("Chained comparisons are not supported; use one comparison only.")
        left,right,op=self._visit(node.left),self._visit(node.comparators[0]),node.ops[0]
        from factor_engine.api.cleaned_ops import make_cleaned_call_factory
        mapping={ast.Lt:"lt",ast.LtE:"le",ast.Eq:"eq",ast.Gt:"gt",ast.GtE:"ge",ast.NotEq:"ne"}
        for kind,canonical in mapping.items():
            if isinstance(op,kind):return make_cleaned_call_factory(canonical)(left,right)
        raise DSLParseError(f"Unsupported comparison: {type(op).__name__}")
    def _visit_binop(self,node:ast.BinOp)->Any:
        left,right=self._visit(node.left),self._visit(node.right)
        try:
            return self._apply_binop(node, left, right)
        except (TypeError, ArithmeticError) as exc:
            raise DSLParseError("Invalid operands for binary operation") from exc

    def _apply_binop(self,node:ast.BinOp,left:Any,right:Any)->Any:
        if isinstance(left,(str,int,float)) and not isinstance(left,bool) and isinstance(right,(str,int,float)) and not isinstance(right,bool):
            return self._bounded_constant_binop(node.op,left,right)
        if isinstance(node.op,ast.Add):return left+right
        if isinstance(node.op,ast.Sub):return left-right
        if isinstance(node.op,ast.Mult):return left*right
        if isinstance(node.op,ast.Div):return left/right
        if isinstance(node.op,ast.Pow):
            from factor_engine.api.cleaned_ops import make_cleaned_call_factory
            return make_cleaned_call_factory("power")(left,right)
        # CogAlpha pandas code uses bitwise & | ~ for elementwise boolean logic;
        # map onto the daily logical operators and_/or_/not_.
        if isinstance(node.op,ast.BitAnd):
            from factor_engine.api.cleaned_ops import make_cleaned_call_factory
            return make_cleaned_call_factory("and_")(left,right)
        if isinstance(node.op,ast.BitOr):
            from factor_engine.api.cleaned_ops import make_cleaned_call_factory
            return make_cleaned_call_factory("or_")(left,right)
        raise DSLParseError(f"Unsupported binary operator: {type(node.op).__name__}")

    def _bounded_constant_binop(self,op:ast.operator,left:Any,right:Any)->Any:
        """Evaluate a scalar constant operation only after pre-allocation checks."""
        if isinstance(left,str) or isinstance(right,str):
            if isinstance(op,ast.Add) and isinstance(left,str) and isinstance(right,str):
                size=len(left.encode("utf-8"))+len(right.encode("utf-8"))
            elif isinstance(op,ast.Mult) and isinstance(left,str) and isinstance(right,int):
                if right < 0:
                    size=0
                else:
                    unit=len(left.encode("utf-8"))
                    limit=self._budget.max_string_literal_bytes
                    if unit and right > limit // unit:
                        raise DSLParseError("constant string result exceeds ComplexityBudget.max_string_literal_bytes")
                    size=unit*right
            elif isinstance(op,ast.Mult) and isinstance(right,str) and isinstance(left,int):
                return self._bounded_constant_binop(op,right,left)
            else:
                raise DSLParseError("Unsupported string arithmetic in factor DSL.")
            if size>self._budget.max_string_literal_bytes:
                raise DSLParseError("constant string result exceeds ComplexityBudget.max_string_literal_bytes")
            return left+right if isinstance(op,ast.Add) else left*right
        try:
            if isinstance(op,ast.Add):result=left+right
            elif isinstance(op,ast.Sub):result=left-right
            elif isinstance(op,ast.Mult):result=left*right
            elif isinstance(op,ast.Div):result=left/right
            else:raise DSLParseError(f"Unsupported constant operator: {type(op).__name__}")
        except (ArithmeticError,OverflowError) as exc:
            raise DSLParseError(f"Invalid constant arithmetic: {exc}") from exc
        if isinstance(result,float) and not math.isfinite(result):
            raise DSLParseError("constant arithmetic produced a non-finite value")
        if abs(result)>self._budget.max_literal_magnitude:
            raise DSLParseError("constant result exceeds ComplexityBudget.max_literal_magnitude")
        return result

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
        from factor_engine.cleaned_operators.base import _coerce_declared_numeric_string
        from factor_engine.cleaned_operators.registry import OperatorRegistry
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


def _call_parameter_binding(name:str)->tuple[dict[str,str],tuple[str,...]]:
    """Resolve canonical parameter names from the existing metadata authority."""
    try:
        from factor_engine.backend.parameter_aliases import _effective_alias_map
        from factor_engine.cleaned_operators.registry import OperatorRegistry
    except ImportError as exc:
        raise DSLParseError(f"Parameter binding authority unavailable for {name!r}: {exc}") from exc
    try:
        canonical=OperatorRegistry.resolve_canonical(name)
        op=OperatorRegistry.get(canonical,mode="any")
    except (KeyError,ValueError,RuntimeError) as exc:
        raise DSLParseError(f"Operator metadata resolution failed for {name!r}: {exc}") from exc
    if op is None:
        return {},()
    meta=getattr(op,"metadata",None)
    if meta is None:
        raise DSLParseError(f"Operator metadata unavailable for {name!r}; cannot verify parameter binding.")
    return dict(_effective_alias_map(canonical)),tuple(getattr(meta,"param_names",None) or ())


def _missing_required_params(name:str,bound_positional,bound_kwargs):
    """R59: parse-time mirror of the runtime binder's required-parameter gate.

    A call that omits a declared panel/scalar parameter with neither a kernel
    default nor a ParamSpec default (e.g. ``ts_abdi_ranaldo_spread(close, 20)``
    — the kernel needs close/high/low) is rejected here with a typed
    DSLParseError instead of crashing inside the kernel at runtime.  The
    acceptance set is IDENTICAL to ``bind_operator_call`` (same panels/scalars,
    same ``_kernel_param_defaults`` source, same ParamSpec-default escape), so
    nothing the binder accepts can be rejected here.
    """
    if not bound_positional and not bound_kwargs:
        return []
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        from factor_engine.cleaned_operators.base import (
            MISSING as _MISSING,
            _kernel_param_defaults,
        )
        canonical=OperatorRegistry.resolve_canonical(name)
        operator=OperatorRegistry.get(canonical,mode="any")
        meta=getattr(operator,"metadata",None)
        if meta is None:
            return []
        panels=tuple(getattr(meta,"panel_params",None) or ())
        scalars=tuple(getattr(meta,"scalar_params",None) or ())
        if not (panels or scalars):
            return []
        specs=getattr(meta,"param_specs",None) or {}
        defaults=_kernel_param_defaults(operator)
        provided=set(bound_positional)|set(bound_kwargs)
        return [
            p for p in (*panels,*scalars)
            if p not in provided
            and p not in (defaults or {})
            and getattr(specs.get(p),"default",_MISSING) is _MISSING
        ]
    except Exception:
        return []


def _declared_sequence_spec(name:str,param_name:str,container:type):
    """Return the exact declared list/tuple ParamSpec, never infer one."""
    if not param_name:
        return None
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        canonical=OperatorRegistry.resolve_canonical(name)
        operator=OperatorRegistry.get(canonical,mode="any")
    except (KeyError,RuntimeError,ValueError):
        return None
    metadata=getattr(operator,"metadata",None)
    if metadata is None or param_name not in set(
        getattr(metadata,"scalar_params",None) or ()
    ):
        return None
    root=(getattr(metadata,"param_specs",None) or {}).get(param_name)
    candidates=(root,*(getattr(root,"alternatives",None) or ()))
    for candidate in candidates:
        if getattr(candidate,"dtype",None) is container:
            return candidate
    return None


def _is_field_identifier(name:str)->bool:return bool(name) and not name[0].isdigit() and all(c.isalnum() or c=="_" for c in name)


class DSLParser:
    """Reusable parser session with an explicit, immutable allowlist snapshot.

    Construct a session for bulk parsing when every formula must use the same
    authoring surface.  A registry lifecycle/version change invalidates the
    session and raises on its next parse; create a new session to refresh it.
    Ordinary :func:`parse_expr` calls remain fresh and never use a
    process-global cache.
    """

    def __init__(self, *, surface: str = "daily", dialect: str = "native",
                 dialect_version: str | None = None,
                 budget: ComplexityBudget | None = None) -> None:
        seed = _ExprBuilder(surface=surface, dialect=dialect,
                            dialect_version=dialect_version, budget=budget)
        self._surface = seed._surface
        self._dialect = seed._dialect
        self._dialect_version = seed._dialect_version
        self._budget = seed._budget
        # Copy before wrapping: callers cannot mutate the mapping returned by
        # build_dsl_allowlist and silently widen this session's surface.
        self._allowed = MappingProxyType(dict(seed._allowed))
        self._registry_token = _current_registry_token()
        if self._registry_token[1] not in {"finalized", "frozen"}:
            raise DSLParseError(
                "DSLParser sessions require a finalized or frozen operator registry"
            )

    def parse(self, text: str) -> Expr:
        if _current_registry_token() != self._registry_token:
            raise DSLParseError(
                "operator registry changed after this DSLParser session was "
                "created; create a new session"
            )
        # A fresh builder resets node/depth counters for every formula and also
        # makes one session safe to reuse after a rejected over-budget formula.
        return _ExprBuilder(
            surface=self._surface, dialect=self._dialect,
            dialect_version=self._dialect_version, budget=self._budget,
            allowed=self._allowed,
        ).build(text)

    def parse_many(self, texts: Iterable[str]) -> list[Expr]:
        """Parse an iterable in order, failing at the first invalid formula."""
        return [self.parse(text) for text in texts]


def _current_registry_token() -> tuple[int, str]:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry.version(), OperatorRegistry.lifecycle()


def parse_expr(text:str,*,surface:str="daily",dialect:str="native",dialect_version:str|None=None,budget:ComplexityBudget|None=None)->Expr:return _ExprBuilder(surface=surface,dialect=dialect,dialect_version=dialect_version,budget=budget).build(text)


def parse_recommended_expr(text:str,*,dialect:str="native",dialect_version:str|None=None,budget:ComplexityBudget|None=None)->Expr:
    """Parse an Agent/research formula against the complete public runtime surface."""
    return parse_expr(text, surface="all", dialect=dialect,
                      dialect_version=dialect_version, budget=budget)
def parse_factor(text:str,*,name:str="factor",freq:str="1d",universe:str|None=None,description:str|None=None,surface:str="daily",dialect:str="native",dialect_version:str|None=None,budget:ComplexityBudget|None=None)->Factor:return Factor(name=name,expr=parse_expr(text,surface=surface,dialect=dialect,dialect_version=dialect_version,budget=budget),freq=freq,universe=universe,description=description,source_expr=text,surface=surface,dialect=dialect,dialect_version=dialect_version)
