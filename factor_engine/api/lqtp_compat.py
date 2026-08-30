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
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
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
def _momentum_dispatch(x:Any=None,window:Any=None,*,field:Any=None,**kwargs:Any):
    if field is not None:
        x = field
    if isinstance(x, str):
        # LQTP ``momentum(field=close, window=20)`` passes the bare column name
        # as a string — resolve it to a real field reference so the factor
        # actually reads the series (not a string literal).
        from factor_engine.api.columns import field as _field
        x = _field(x, strict=False)
    if x is None or window is None:
        raise LQTPCompatibilityError("momentum requires (x, window) or (field=..., window=...)")
    return _factory("ts_delta")(x,window)
def _source_col(table:str,field:str,**params:Any):
    from factor_engine.api.source_ref import source_col
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
        from factor_engine.api.columns import col
        return _factory("real_turnover_rate")(col("volume"),_source_col("TurnoverBaseDaily","TurnoverBase"))
    if len(args)==2:return _factory("real_turnover_rate")(*args)
    raise LQTPCompatibilityError("real_turnover_rate accepts () or (volume, turnover_base)")
def _asof_dispatch(x:Any):
    from factor_engine.api.source_ref import transform_source_col
    return transform_source_col(x,"financial_asof")
def _financial_lag_dispatch(x:Any,quarters:Any):
    from factor_engine.api.source_ref import transform_source_col
    return transform_source_col(x,"financial_lag",quarters=quarters)
def _intermediate_dispatch(name:Any,version:Any):
    from factor_engine.api.source_ref import intermediate_col
    # Review-8 #469: no int() coercion here — intermediate_col/_strict_int owns
    # the strict integer verdict (True/1.9/"2"/NaN are all rejected).
    return intermediate_col(str(name), version)
def _minute_at_dispatch(field:Any,hhmm:Any):
    from factor_engine.api.source_ref import transform_source_col
    return transform_source_col(field,"minute_at",hhmm=str(hhmm))
def _minute_range_dispatch(field:Any,start:Any,end:Any):
    from factor_engine.api.source_ref import transform_source_col
    return transform_source_col(field,"minute_range",start=str(start),end=str(end))
def _minute_resample_dispatch(field:Any,period:Any):
    from factor_engine.api.source_ref import transform_source_col
    return transform_source_col(field,"minute_resample",period=period)
def _minute_bar_dispatch(field:Any,period:Any,index:Any):
    from factor_engine.api.source_ref import transform_source_col
    return transform_source_col(field,"minute_bar",period=period,index=index)

_EXACT_COMPAT={"decay_linear":"ts_decay_linear","ts_rank_pct":"ts_rank","ts_ewm_mean":"ts_ema","ts_expanding_rank":"expanding_rank","ts_hump_decay":"hump_decay"}
# 2026-08-29 平台财务/统计函数别名（catalog 848 补算因子里 497 个用到）：
# - ttm(x)        → trailing-12M 原始财务列已是 TTM/年报口径：直接用 x 本身
# - quarter(x)    → 单季列：取 x 本身（StockIncome/StockCashFlow 单季口径）
# - yoy(x)        → 同比 = (x - x_lq)/|x_lq|，x_lq = ts_delay(x, 250)（250 交易日≈12个月）
# - nan_to_num(x, v) → where(is_nan(x), v, x)（与平台同义）
# - avg2(a,b)     → (a + b) / 2
# - ts_ewm_cov(x, y, span) → ewm_cov（FE canonical，同 span 语义）
_FINANCIAL_IDENTITY_FUNCS={"ttm":"__lqtp_ttm__","quarter":"__lqtp_quarter__"}
_AMBIGUOUS_EXTERNAL_NAMES=frozenset()
_BLOCKED_SOURCE_NAMES={"l2_sum":"StockTransaction/L2 source contract is not configured","l2_sum_if":"StockTransaction/L2 source contract is not configured","l2_count":"StockTransaction/L2 source contract is not configured","l2_count_if":"StockTransaction/L2 source contract is not configured"}
_BLOCKED_SEMANTIC_NAMES={"group_minmax":"exact LQTP group_minmax contract is not installed","group_quantile_mask":"exact LQTP group_quantile_mask contract is not installed"}
_LOGICAL_TABLES=frozenset({"DailyBar","StockDailyBar","MinuteBar","StockMinuteBar","BenchmarkIndexDailyBar","StockIncome","StockCashFlow","StockBalance","StockIndicator","StockValuationDaily","SizeDaily","TurnoverBaseDaily","IndustryDaily","StockIndustry","StockTransaction","EtfDailyBar","FuturesDailyBar"})
# R55 platform-audit P0: platform intermediate factors are written
# ``FactorIntermediateDaily__param_id_<hex>.Value`` — the ``__param_id_<hex>``
# suffix carries the materialised intermediate's versioned identity.  The
# attribute normalizer strips the suffix into a plain ``FactorIntermediateDaily``
# reference with the param_id carried as the ``name`` parameter, then the
# ``intermediate`` resolver materialises it.
_INTERMEDIATE_TABLE="FactorIntermediateDaily"
_INTERMEDIATE_PARAM_ID_PREFIX="FactorIntermediateDaily__param_id_"
_TABLE_ALIASES={"benchmark_index":"BenchmarkIndexDailyBar"}
def _ambiguous_dispatch(name:str)->Callable[...,Any]:
    def _raise(*args:Any,**kwargs:Any):raise LQTPCompatibilityError(f"{name} is recognized but its exact LQTP semantic definition is not present in the supplied manual; execution remains fail-closed until a source-backed definition is provided")
    _raise.__name__=name;return _raise
def _blocked_dispatch(name:str,reason:str,source:bool)->Callable[...,Any]:
    def _raise(*args:Any,**kwargs:Any):
        cls=LQTPDataDependencyError if source else LQTPCompatibilityError
        raise cls(f"{name} is recognized but not executable: {reason}")
    _raise.__name__=name;return _raise

def _nan_dispatch(*args:Any,**kwargs:Any):
    """LQTP null literal: returns a NaN float (the scalar-broadcast ``where``
    runtime treats it as a constant NaN panel branch)."""
    import math
    return math.nan


def augment_dsl_allowlist(allow:dict[str,Callable[...,Any]],*,surface:str)->dict[str,Callable[...,Any]]:
    if surface not in {"daily","compat","compat_research","research","all","lqtp"}:return allow
    from factor_engine.cleaned_operators.production_tiers import LQTP_COMPAT_PARSE_CANONICALS
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.api.source_ref import source_col
    out=dict(allow);out.update({"source_col":source_col,"sma":_sma_dispatch,"safe_log":_safe_log_dispatch,"nullif_zero":_nullif_zero_dispatch,"clean":_clean_dispatch,"momentum":_momentum_dispatch,"market_ret":_market_ret_dispatch,"historical_var":_historical_var_dispatch,"historical_cvar":_historical_cvar_dispatch,"rolling_beta_to_market":_rolling_beta_dispatch,"fp_beta":_rolling_beta_dispatch,"downside_beta":_downside_beta_dispatch,"tail_beta":_tail_beta_dispatch,"residual_momentum_capm":_residual_momentum_dispatch,"coskewness_to_market":_market_three_arg_dispatch("coskewness_to_market"),"idio_vol":_market_three_arg_dispatch("idio_vol"),"idio_skew":_market_three_arg_dispatch("idio_skew"),"real_turnover_rate":_real_turnover_rate_dispatch,"asof":_asof_dispatch,"financial_lag":_financial_lag_dispatch,"lag":_financial_lag_dispatch,"intermediate":_intermediate_dispatch,"minute_at":_minute_at_dispatch,"minute_range":_minute_range_dispatch,"minute_resample":_minute_resample_dispatch,"minute_bar":_minute_bar_dispatch,"is_nan":_factory("is_nan"),"where":_factory("where"),"ewm_cov":_factory("ewm_cov"),"nan":_nan_dispatch,"null":_nan_dispatch})
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
        # R55 platform-audit P0: ``FactorIntermediateDaily__param_id_<hex>.Value``
        # → table=FactorIntermediateDaily, param name=<param_id>, field=Value.
        if isinstance(node.value,ast.Name) and node.value.id.startswith(_INTERMEDIATE_PARAM_ID_PREFIX):
            raw = node.value.id
            table = _INTERMEDIATE_TABLE
            params.append(("name", raw[len(_INTERMEDIATE_PARAM_ID_PREFIX):]))
        if table is None and isinstance(node.value,ast.Name) and node.value.id in _LOGICAL_TABLES:table=node.value.id
        elif isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name):
            raw=node.value.func.id;table=_TABLE_ALIASES.get(raw,raw if raw in _LOGICAL_TABLES else None)
            if table is not None:
                # positional DataTable parameter: the platform writes
                # ``benchmark_index("000985.SH")`` — the sole positional arg is
                # the index code.  Map it to the named ``index`` parameter the
                # source_col runtime consumes (keeps the flat-call contract).
                if node.value.args and len(node.value.args) == 1 and not node.value.keywords:
                    first = node.value.args[0]
                    if not isinstance(first, ast.Constant):
                        raise LQTPCompatibilityError(f"{raw} DataTable positional parameter must be a scalar literal")
                    params.append(("index", first.value))
                elif node.value.args:
                    raise LQTPCompatibilityError(f"{raw} DataTable parameters must be named")
                for kw in node.value.keywords:
                    if kw.arg is None or not isinstance(kw.value,ast.Constant):raise LQTPCompatibilityError("DataTable parameters must be scalar literals")
                    params.append((kw.arg,kw.value.value))
        if table is None:return node
        args=[ast.Constant(table),ast.Constant(node.attr)]
        for k,v in params:args.extend([ast.Constant(k),ast.Constant(v)])
        return ast.copy_location(ast.Call(func=ast.Name(id="source_col",ctx=ast.Load()),args=args,keywords=[]),node)
def _normalize_bare_names(source:str)->str:
    tokens=list(tokenize.generate_tokens(io.StringIO(source).readline));pairs=[];bare={"true_range","market_ret","overnight_return","intraday_return","daily_return"}
    for i,current in enumerate(tokens):
        if current.type==token.NAME and current.string in bare:
            j=i+1
            while j<len(tokens) and tokens[j].type in {tokenize.NL,tokenize.NEWLINE,tokenize.INDENT,tokenize.DEDENT}:j+=1
            if not (j<len(tokens) and tokens[j].string=="("):
                if current.string=="true_range":
                    pairs.extend([(token.NAME,"true_range"),(token.OP,"("),(token.NAME,"high"),(token.OP,","),(token.NAME,"low"),(token.OP,","),(token.NAME,"close"),(token.OP,")")])
                elif current.string=="market_ret":
                    pairs.extend([(token.NAME,"market_ret"),(token.OP,"("),(token.OP,")")])
                elif current.string=="daily_return":
                    # LQTP 平台 ``daily_return`` == ``ret``（日收益 = Return，bp）。
                    # FE bare ``ret`` 是注册字段（StockDailyBarAdj.Return），直接替换。
                    pairs.extend([(token.NAME,"ret")])
                else:
                    # overnight_return / intraday_return bare → call forms
                    pairs.extend([(token.NAME,current.string),(token.OP,"("),(token.OP,")")])
                continue
        if current.type!=tokenize.ENDMARKER:pairs.append((current.type,current.string))
    return tokenize.untokenize(pairs)


def _rewrite_bare_nan_and_mask(source:str)->str:
    """Rewrite LQTP bare names that are NOT plain columns into AST call forms.

    - ``nan`` / ``null`` (the LQTP null-literals) → ``nan()`` / ``null()`` so
      they bind to the registered ``nan`` constant operator instead of being
      pushed down as physical columns (DuckDB ``Referenced column "nan" not
      found``).
    - ``quality_tradable_mask`` (a platform derived mask: tradable = not
      suspended) → ``quality_tradable_mask()`` so it binds to the registered
      operator with the same platform meaning.
    """
    tokens=list(tokenize.generate_tokens(io.StringIO(source).readline));pairs=[];rewrite={"nan","null","quality_tradable_mask"}
    for i,current in enumerate(tokens):
        if current.type==token.NAME and current.string in rewrite:
            j=i+1
            while j<len(tokens) and tokens[j].type in {tokenize.NL,tokenize.NEWLINE,tokenize.INDENT,tokenize.DEDENT}:j+=1
            if not (j<len(tokens) and tokens[j].string=="("):
                pairs.extend([(token.NAME,current.string),(token.OP,"("),(token.OP,")")])
                continue
        if current.type!=tokenize.ENDMARKER:pairs.append((current.type,current.string))
    return tokenize.untokenize(pairs)


class _PlatformFnRewriter(ast.NodeTransformer):
    """2026-08-29: LQTP 平台财务/统计函数 → FE 等价展开（AST 级）。

    - ttm(x) / quarter(x)：平台 StockIncome/StockCashFlow 等财务列本身
      是累计/TTM 口径，ttm()/quarter() 只是对窗口的声明，FE 侧恒等展开 x。
    - yoy(x)：同比 = (x - delay(x,250)) / |delay(x,250)|（250 交易日≈12个月）。
    - nan_to_num(x, v)：≡ where(is_nan(x), v, x)（nan→v，与平台同义；不用 ifnan/fillna_const——前者属 RESEARCH_CANONICALS 被移出 Factor DSL，后者触发 P1-003 财务零填充守卫）
    - avg2(a, b)：(a + b) / 2；单参 avg2(x) = (x + x_prev)/2（PIT 报告期上一期，
      financial_lag(x,1) 走 lqtp_logical_source_v2 的 fiscal-period 回溯）。
    - ts_ewm_cov(x, y, alpha)：EWM 协方差。FE canonical 是 ewm_cov(x, y, span)；
      平台第三参是 0<alpha<=1 衰减率，按 ts_ema/ewm_std 同族 alpha 语义
      span = round(1/alpha)（0.1→10, 0.15→7, 0.05→20, 0.2→5）。
    - ts_sumac(x, n) → ts_sum(x, n)（JoinQuant 滚动和，与 ts_sum 同语义）。
    - ts_regression_slope_sequence(x, n) → ts_time_slope(x, n)（滚动时间斜率）。
    """
    _IDENTITY: frozenset[str] = frozenset({"ttm", "quarter"})

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self._IDENTITY and node.args:
                return node.args[0]
            if name == "yoy" and len(node.args) == 1:
                x = node.args[0]
                lag = ast.Call(func=ast.Name(id="ts_delay", ctx=ast.Load()),
                               args=[x, ast.Constant(250)], keywords=[])
                return ast.copy_location(ast.Call(
                    func=ast.Name(id="safe_div", ctx=ast.Load()),
                    args=[ast.BinOp(left=ast.copy_location(x, node), op=ast.Sub(),
                                    right=lag),
                          ast.Call(func=ast.Name(id="abs", ctx=ast.Load()),
                                   args=[lag], keywords=[])],
                    keywords=[]), node)
            if name == "nan_to_num" and len(node.args) >= 1:
                # ifnan 在 layer_governance.RESEARCH_CANONICALS（load_all 后被移出
                # Factor DSL → resolve_canonical_strict KeyError）。用 production
                # 面的 where + is_nan 组合表达同一语义（两者都不在 governance
                # 移除名单），语义恒等：nan_to_num(x, v) ≡ where(is_nan(x), v, x)。
                fill_v = node.args[1] if len(node.args) > 1 else ast.Constant(0.0)
                isnan_call = ast.copy_location(ast.Call(
                    func=ast.Name(id="is_nan", ctx=ast.Load()),
                    args=[node.args[0]], keywords=[]), node)
                return ast.copy_location(ast.Call(
                    func=ast.Name(id="where", ctx=ast.Load()),
                    args=[isnan_call, fill_v, node.args[0]], keywords=[]), node)
            if name == "avg2" and len(node.args) == 2:
                return ast.copy_location(ast.BinOp(
                    left=ast.copy_location(node.args[0], node), op=ast.Add(),
                    right=ast.copy_location(node.args[1], node)), node)
            if name == "avg2" and len(node.args) == 1:
                # 单参 avg2(x) = (x + x_prev)/2，x_prev = 上一报告期财务值
                # （financial_lag(x, 1)，PIT 报告期回溯，防泄漏）。
                x = node.args[0]
                lag = ast.Call(func=ast.Name(id="financial_lag", ctx=ast.Load()),
                               args=[ast.copy_location(x, node), ast.Constant(1)], keywords=[])
                return ast.copy_location(ast.BinOp(
                    left=ast.copy_location(x, node), op=ast.Add(), right=lag), node)
            if name == "ts_ewm_cov" and len(node.args) >= 2:
                span_arg: Any
                if len(node.args) >= 3:
                    span = node.args[2]
                    if isinstance(span, ast.Constant) and isinstance(span.value, (int, float)):
                        a = float(span.value)
                        if 0.0 < a < 1.0:
                            # alpha 语义 → ewm_cov span = round(1/alpha)
                            span = ast.Constant(round(1.0 / a))
                else:
                    span = ast.Constant(20)
                return ast.copy_location(ast.Call(
                    func=ast.Name(id="ewm_cov", ctx=ast.Load()),
                    args=[*node.args[:2], span], keywords=list(node.keywords)), node)
            if name == "ts_sumac" and len(node.args) >= 1:
                # JoinQuant ts_sumac(x, n) = 滚动和 = ts_sum(x, n)。
                return ast.copy_location(ast.Call(
                    func=ast.Name(id="ts_sum", ctx=ast.Load()),
                    args=list(node.args), keywords=list(node.keywords)), node)
            if name == "ts_regression_slope_sequence" and len(node.args) >= 1:
                # JoinQuant 滚动时间斜率 = ts_time_slope(x, n)。
                return ast.copy_location(ast.Call(
                    func=ast.Name(id="ts_time_slope", ctx=ast.Load()),
                    args=list(node.args), keywords=list(node.keywords)), node)
            if name == "nan" or name == "null":
                # The LQTP null literal.  ``ast.Constant(float('nan'))`` round-trips
                # through py3.12 ``ast.unparse`` as ``(1e309-1e309)``, which the DSL
                # rejects as a non-finite literal.  Keep the call form ``nan()`` —
                # ``augment_dsl_allowlist`` binds it to ``_nan_dispatch`` which
                # returns a NaN float at parse time; the scalar-broadcast ``where``
                # runtime then treats it as a constant NaN panel branch.
                return node
            if name == "quality_tradable_mask":
                # LQTP 平台 derived mask：可交易 = 未停牌（is_suspend==0）。
                # FE 等价：not(is_suspend)。is_suspend 为 bool/EventBool，
                # not_ 把它转成 1.0/0.0 面板。
                return ast.copy_location(ast.Call(
                    func=ast.Name(id="not_", ctx=ast.Load()),
                    args=[ast.copy_location(ast.Name(id="is_suspend", ctx=ast.Load()), node)],
                    keywords=[]), node)
        return node


def normalize_lqtp_formula(text:str)->str:
    source=str(text or "")
    if not source.strip():return source
    source=_normalize_bare_names(source)
    source=_rewrite_bare_nan_and_mask(source)
    try:
        tree=ast.parse(source,mode="eval")
        tree=_SourceAttributeNormalizer().visit(tree)
        tree=_PlatformFnRewriter().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except LQTPCompatibilityError:raise
    except Exception as exc:raise LQTPCompatibilityError(f"failed to normalize LQTP formula: {source}") from exc
