# -*- coding: utf-8 -*-
"""Versioned LQTP/JQ compatibility shell."""
from __future__ import annotations
import ast, io, token, tokenize
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_LQTP_DIALECT_VERSION="2026-07-19"
DEFAULT_MARKET_INDEX="000985.SH"
class LQTPCompatibilityError(ValueError): pass
class LQTPDataDependencyError(LQTPCompatibilityError): pass
@dataclass(frozen=True)
class LQTPDialectSpec:
    name:str="lqtp"; version:str=DEFAULT_LQTP_DIALECT_VERSION; strict_case:bool=True

def _factory(canonical:str)->Callable[...,Any]:
    from api.cleaned_ops import make_cleaned_call_factory
    return make_cleaned_call_factory(canonical)
def _null_like(x:Any):
    z=_factory("subtract")(x,x); return _factory("safe_div_null")(z,z)
def _sma_dispatch(*args:Any,**kwargs:Any):
    if kwargs:
        if set(kwargs)<={"window"} and len(args)==1:return _factory("ts_mean")(*args,**kwargs)
        if set(kwargs)<={"n","m"} and len(args)==1 and {"n","m"}<=set(kwargs):return _factory("ts_sma_cn")(*args,**kwargs)
        raise LQTPCompatibilityError("sma accepts sma(x, window) or sma(x, n, m)")
    if len(args)==2:return _factory("ts_mean")(*args)
    if len(args)==3:return _factory("ts_sma_cn")(*args)
    raise LQTPCompatibilityError("sma accepts exactly 2 or 3 arguments")
def _safe_log_dispatch(x:Any):return _factory("where")(_factory("gt")(x,0.0),_factory("log")(x),_null_like(x))
def _nullif_zero_dispatch(x:Any):return _factory("where")(_factory("eq")(x,0.0),_null_like(x),x)
def _clean_dispatch(x:Any,replacement:Any=0.0):return _factory("coalesce")(x,replacement)
def _momentum_dispatch(x:Any,window:Any):return _factory("ts_delta")(x,window)
def _source_col(table:str,field:str,**params:Any):
    from api.source_ref import source_col
    return source_col(table,field,dialect="lqtp",dialect_version=DEFAULT_LQTP_DIALECT_VERSION,**params)
def _benchmark_return(index:str=DEFAULT_MARKET_INDEX):
    c=_source_col("BenchmarkIndexDailyBar","Close",index=str(index));p=_source_col("BenchmarkIndexDailyBar","PreClose",index=str(index))
    return _factory("subtract")(_factory("safe_div_null")(c,p),1.0)
def _market_ret_dispatch(*args:Any,**kwargs:Any):
    if args or kwargs:raise LQTPCompatibilityError("market_ret takes no arguments")
    return _benchmark_return()
def _rolling_beta_dispatch(ret:Any,index_or_benchmark:Any,window:Any|None=None):
    if window is None:window=index_or_benchmark;bench=_benchmark_return()
    elif isinstance(index_or_benchmark,str):bench=_benchmark_return(index_or_benchmark)
    else:bench=index_or_benchmark
    return _factory("rolling_beta_to_market")(ret,bench,window)
def _tail_beta_dispatch(*args:Any):
    if len(args)==3:ret,window,q=args;bench=_benchmark_return()
    elif len(args)==4:
        ret,idx,window,q=args;bench=_benchmark_return(idx) if isinstance(idx,str) else idx
    else:raise LQTPCompatibilityError("tail_beta accepts (ret, window, q) or (ret, index, window, q)")
    return _factory("tail_beta")(ret,bench,window,q)
def _residual_momentum_dispatch(*args:Any):
    if len(args)==2:ret,window=args;bench=_benchmark_return()
    elif len(args)==3:
        ret,idx,window=args;bench=_benchmark_return(idx) if isinstance(idx,str) else idx
    else:raise LQTPCompatibilityError("residual_momentum_capm accepts (ret, window) or (ret, index, window)")
    return _factory("residual_momentum_capm")(ret,bench,window)
def _market_three_arg_dispatch(canonical:str)->Callable[...,Any]:
    def dispatch(ret:Any,index_or_benchmark:Any,window:Any):
        bench=_benchmark_return(index_or_benchmark) if isinstance(index_or_benchmark,str) else index_or_benchmark
        return _factory(canonical)(ret,bench,window)
    return dispatch
def _downside_beta_dispatch(ret:Any,index_or_benchmark:Any,window:Any):
    bench=_benchmark_return(index_or_benchmark) if isinstance(index_or_benchmark,str) else index_or_benchmark;cond=_factory("lt")(bench,0.0)
    return _factory("ts_beta")(_factory("where")(cond,ret,_null_like(ret)),_factory("where")(cond,bench,_null_like(bench)),window)
def _historical_var_dispatch(ret:Any,window:Any,q:Any):return _factory("neg")(_factory("ts_quantile")(ret,window,q))
def _historical_cvar_dispatch(ret:Any,window:Any,q:Any):return _factory("lqtp_historical_cvar")(ret,window,q)
def _real_turnover_rate_dispatch(*args:Any):
    if len(args)==0:
        from api.columns import col
        return _factory("real_turnover_rate")(col("volume"),_source_col("TurnoverBaseDaily","TurnoverBase"))
    if len(args)==2:return _factory("real_turnover_rate")(*args)
    raise LQTPCompatibilityError("real_turnover_rate accepts () or (volume, turnover_base)")
def _asof_dispatch(x:Any):
    from api.source_ref import transform_source_col
    return transform_source_col(x,"financial_asof")
def _financial_lag_dispatch(x:Any,quarters:Any):
    from api.source_ref import transform_source_col
    return transform_source_col(x,"financial_lag",quarters=quarters)
def _intermediate_dispatch(name:Any,version:Any):
    from api.source_ref import intermediate_col
    return intermediate_col(str(name),int(version))
def _minute_at_dispatch(field:Any,hhmm:Any):
    from api.source_ref import transform_source_col
    return transform_source_col(field,"minute_at",hhmm=str(hhmm))
def _minute_range_dispatch(field:Any,start:Any,end:Any):
    from api.source_ref import transform_source_col
    return transform_source_col(field,"minute_range",start=str(start),end=str(end))
def _minute_resample_dispatch(field:Any,period:Any):
    from api.source_ref import transform_source_col
    return transform_source_col(field,"minute_resample",period=int(period))
def _minute_bar_dispatch(field:Any,period:Any,index:Any):
    from api.source_ref import transform_source_col
    return transform_source_col(field,"minute_bar",period=int(period),index=int(index))

_EXACT_COMPAT={"decay_linear":"ts_decay_linear","ts_rank_pct":"ts_rank","ts_ewm_mean":"ts_ema","ts_expanding_rank":"expanding_rank","ts_hump_decay":"hump_decay"}
_AMBIGUOUS_EXTERNAL_NAMES=frozenset({"ts_regression_slope_sequence","ts_sumac"})
_BLOCKED_SOURCE_NAMES={"l2_sum":"StockTransaction/L2 source contract is not configured","l2_sum_if":"StockTransaction/L2 source contract is not configured","l2_count":"StockTransaction/L2 source contract is not configured","l2_count_if":"StockTransaction/L2 source contract is not configured"}
_BLOCKED_SEMANTIC_NAMES={"group_minmax":"exact LQTP group_minmax contract is not installed","group_quantile_mask":"exact LQTP group_quantile_mask contract is not installed"}
_LOGICAL_TABLES=frozenset({"DailyBar","StockDailyBar","MinuteBar","StockMinuteBar","BenchmarkIndexDailyBar","StockIncome","StockCashFlow","StockBalance","SizeDaily","TurnoverBaseDaily","IndustryDaily","StockTransaction","EtfDailyBar","FuturesDailyBar"})
_TABLE_ALIASES={"benchmark_index":"BenchmarkIndexDailyBar"}
def _ambiguous_dispatch(name:str)->Callable[...,Any]:
    def _raise(*args:Any,**kwargs:Any):raise LQTPCompatibilityError(f"{name} is recognized but its exact LQTP semantic definition is not present in the supplied manual; execution remains fail-closed until a source-backed definition is provided")
    _raise.__name__=name;return _raise
def _blocked_dispatch(name:str,reason:str,source:bool)->Callable[...,Any]:
    def _raise(*args:Any,**kwargs:Any):
        cls=LQTPDataDependencyError if source else LQTPCompatibilityError
        raise cls(f"{name} is recognized but not executable: {reason}")
    _raise.__name__=name;return _raise

def augment_dsl_allowlist(allow:dict[str,Callable[...,Any]],*,surface:str)->dict[str,Callable[...,Any]]:
    if surface not in {"daily","compat","compat_research","research","all","lqtp"}:return allow
    from cleaned_operators.production_tiers import LQTP_COMPAT_PARSE_CANONICALS
    from cleaned_operators.registry import OperatorRegistry
    from api.source_ref import source_col
    out=dict(allow);out.update({"source_col":source_col,"sma":_sma_dispatch,"safe_log":_safe_log_dispatch,"nullif_zero":_nullif_zero_dispatch,"clean":_clean_dispatch,"momentum":_momentum_dispatch,"market_ret":_market_ret_dispatch,"historical_var":_historical_var_dispatch,"historical_cvar":_historical_cvar_dispatch,"rolling_beta_to_market":_rolling_beta_dispatch,"fp_beta":_rolling_beta_dispatch,"downside_beta":_downside_beta_dispatch,"tail_beta":_tail_beta_dispatch,"residual_momentum_capm":_residual_momentum_dispatch,"coskewness_to_market":_market_three_arg_dispatch("coskewness_to_market"),"idio_vol":_market_three_arg_dispatch("idio_vol"),"idio_skew":_market_three_arg_dispatch("idio_skew"),"real_turnover_rate":_real_turnover_rate_dispatch,"asof":_asof_dispatch,"financial_lag":_financial_lag_dispatch,"lag":_financial_lag_dispatch,"intermediate":_intermediate_dispatch,"minute_at":_minute_at_dispatch,"minute_range":_minute_range_dispatch,"minute_resample":_minute_resample_dispatch,"minute_bar":_minute_bar_dispatch})
    for external,canonical in _EXACT_COMPAT.items():
        if OperatorRegistry.get(canonical) is not None:out.setdefault(external,_factory(canonical))
    for name in _AMBIGUOUS_EXTERNAL_NAMES:out[name]=_ambiguous_dispatch(name)
    for name,reason in _BLOCKED_SOURCE_NAMES.items():out[name]=_blocked_dispatch(name,reason,True)
    for name,reason in _BLOCKED_SEMANTIC_NAMES.items():out[name]=_blocked_dispatch(name,reason,False)
    for canonical in sorted(LQTP_COMPAT_PARSE_CANONICALS):
        if OperatorRegistry.get(canonical) is None:continue
        out.setdefault(canonical,_factory(canonical))
        for alias,target in OperatorRegistry._aliases.items():
            if target==canonical:out.setdefault(alias,_factory(canonical))
    return out

class _SourceAttributeNormalizer(ast.NodeTransformer):
    def visit_Attribute(self,node:ast.Attribute):
        self.generic_visit(node);table=None;params=[]
        if isinstance(node.value,ast.Name) and node.value.id in _LOGICAL_TABLES:table=node.value.id
        elif isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name):
            raw=node.value.func.id;table=_TABLE_ALIASES.get(raw,raw if raw in _LOGICAL_TABLES else None)
            if table is not None:
                if node.value.args:raise LQTPCompatibilityError(f"{raw} DataTable parameters must be named")
                for kw in node.value.keywords:
                    if kw.arg is None or not isinstance(kw.value,ast.Constant):raise LQTPCompatibilityError("DataTable parameters must be scalar literals")
                    params.append((kw.arg,kw.value.value))
        if table is None:return node
        args=[ast.Constant(table),ast.Constant(node.attr)]
        for k,v in params:args.extend([ast.Constant(k),ast.Constant(v)])
        return ast.copy_location(ast.Call(func=ast.Name(id="source_col",ctx=ast.Load()),args=args,keywords=[]),node)
def _normalize_bare_names(source:str)->str:
    tokens=list(tokenize.generate_tokens(io.StringIO(source).readline));pairs=[];bare={"true_range","market_ret"}
    for i,current in enumerate(tokens):
        if current.type==token.NAME and current.string in bare:
            j=i+1
            while j<len(tokens) and tokens[j].type in {tokenize.NL,tokenize.NEWLINE,tokenize.INDENT,tokenize.DEDENT}:j+=1
            if not (j<len(tokens) and tokens[j].string=="("):
                pairs.extend([(token.NAME,"true_range"),(token.OP,"("),(token.NAME,"high"),(token.OP,","),(token.NAME,"low"),(token.OP,","),(token.NAME,"close"),(token.OP,")")] if current.string=="true_range" else [(token.NAME,"market_ret"),(token.OP,"("),(token.OP,")")]);continue
        if current.type!=tokenize.ENDMARKER:pairs.append((current.type,current.string))
    return tokenize.untokenize(pairs)
def normalize_lqtp_formula(text:str)->str:
    source=str(text or "")
    if not source.strip():return source
    source=_normalize_bare_names(source)
    try:
        tree=ast.parse(source,mode="eval");tree=_SourceAttributeNormalizer().visit(tree);ast.fix_missing_locations(tree);return ast.unparse(tree)
    except LQTPCompatibilityError:raise
    except Exception as exc:raise LQTPCompatibilityError(f"failed to normalize LQTP formula: {source}") from exc
