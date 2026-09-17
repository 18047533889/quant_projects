"""Preserve undisclosed financial values instead of manufacturing imputed data."""
import ast

def migrate_formula(formula:str,logic:str="")->tuple[str,list[str]]:
    if not isinstance(formula,str) or not isinstance(logic,str): raise TypeError("expected strings")
    if len(formula.encode())>65536: raise ValueError("formula exceeds migration input budget")
    try: tree=ast.parse(formula,mode="eval")
    except SyntaxError: return formula,[]
    lines=formula.splitlines(keepends=True);edits=[];changes=[]
    def off(line,col):
        return sum(map(len,lines[:line-1]))+len(lines[line-1].encode()[:col].decode())
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name):continue
        if node.func.id not in {"cs_impute_mean","cs_impute_median"}:continue
        if len(node.args)!=2 or node.keywords:continue
        if not isinstance(node.args[0],ast.Name) or node.args[0].id not in {"roe","roa","gross_profit_margin","net_profit_margin"}:continue
        field=node.args[0].id
        if not isinstance(node.args[1],ast.Constant) or node.args[1].value!=20:continue
        edits.append((off(node.lineno,node.col_offset),off(node.end_lineno,node.end_col_offset),field))
        changes.append(f"SEMANTIC_REDESIGN {node.func.id}({field},20): retain observed {field} and undisclosed nulls, remove forbidden financial imputation; downstream coverage guard retained")
    for a,b,new in sorted(edits,reverse=True):formula=formula[:a]+new+formula[b:]
    return formula,changes
