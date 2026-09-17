"""Supply fiscal period only when every input resolves to one financial table."""
import ast
from functools import lru_cache

_ALLOWED = frozenset({
    "fin_contract_asset_intensity","fin_contract_liability_intensity",
    "fin_deferred_tax_gap","fin_other_earnings_dependence","fin_core_earnings_ratio",
    "fin_goodwill_intensity","fin_impairment_intensity","fin_lease_intensity",
    "fin_noncore_income_ratio","fin_contract_asset_liability_gap",
    "fin_fair_value_income_dependence", "fin_lease_asset_liability_gap",
    "fin_investment_income_dependence", "fin_discontinued_operation_ratio",
    "fin_minority_profit_share", "fin_oci_to_equity",
})
_TABLES=frozenset({"StockBalance","StockIncome","StockCashFlow"})

@lru_cache(maxsize=512)
def _source_table(name, table=None):
    from factor_engine.api.columns import field
    from factor_engine.fields.resolver import resolve_market_field
    try:
        expr=field(name,table=table) if table else field(name)
        return resolve_market_field(expr,"ashare",strict=True).spec.table
    except (KeyError,ValueError,TypeError):
        return None

def _tables(node):
    if isinstance(node,ast.Constant): return set()
    if isinstance(node,ast.Name):
        t=_source_table(node.id)
        return {t} if t in _TABLES else {None}
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
        if node.func.id=="field":
            if len(node.args)!=1 or not isinstance(node.args[0],ast.Constant): return {None}
            kw={k.arg:k.value for k in node.keywords}
            table=kw.get("table")
            if table is not None and not isinstance(table,ast.Constant): return {None}
            t=_source_table(node.args[0].value,table.value if table else None)
            return {t} if t in _TABLES else {None}
        result=set()
        for arg in node.args: result.update(_tables(arg))
        for kw in node.keywords:
            if not isinstance(kw.value,ast.Constant): result.update(_tables(kw.value))
        return result
    return {None}

def migrate_formula(formula:str,logic:str="")->tuple[str,list[str]]:
    if not isinstance(formula,str) or not isinstance(logic,str): raise TypeError("expected strings")
    if len(formula.encode())>65536: raise ValueError("formula exceeds migration input budget")
    try: tree=ast.parse(formula,mode="eval")
    except SyntaxError: return formula,[]
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    lines=formula.splitlines(keepends=True); edits=[];changes=[]
    def off(line,col):
        return sum(map(len,lines[:line-1]))+len(lines[line-1].encode()[:col].decode())
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name): continue
        if not node.func.id.startswith(("fin_", "fiscal_", "report_")): continue
        op=OperatorRegistry.get(node.func.id)
        if op is None: continue
        names=op.metadata.param_names
        if "period_id" not in names: continue
        position=names.index("period_id")
        period=node.args[position] if len(node.args)>position else None
        period_kw=next((k.value for k in node.keywords if k.arg=="period_id"),None)
        if period is not None and period_kw is not None: continue
        period=period if period is not None else period_kw
        if period is None:
            if node.func.id not in _ALLOWED or len(node.args)!=position: continue
        elif not isinstance(period,ast.Name) or period.id not in {"FiscalPeriodId","ReportPeriodEndDate"}:
            continue
        tables=set()
        for arg in node.args:
            if arg is not period: tables.update(_tables(arg))
        for kw in node.keywords:
            if kw.value is not period: tables.update(_tables(kw.value))
        if len(tables)!=1 or None in tables: continue
        table=next(iter(tables))
        value=f"field('report_period_end_date', table='{table}')"
        if period is None:
            end=off(node.end_lineno,node.end_col_offset)-1
            edits.append((end,end,", period_id="+value))
        else:
            edits.append((off(period.lineno,period.col_offset),
                          off(period.end_lineno,period.end_col_offset),value))
        changes.append(f"FIELD_CONTRACT_REPAIR {node.func.id}: all input leaves resolve to {table}; bind its own report_period_end_date, no cross-table period inference")
    for start,end,text in sorted(edits,reverse=True): formula=formula[:start]+text+formula[end:]
    return formula,changes
