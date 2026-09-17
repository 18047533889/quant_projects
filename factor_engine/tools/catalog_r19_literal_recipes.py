"""Quote only registered string choices in old parameter-sketch DSLs."""
import ast
from functools import lru_cache

@lru_cache(maxsize=1)
def _registry():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    load_all()
    return OperatorRegistry

def migrate_formula(formula: str, logic: str = "") -> tuple[str, list[str]]:
    if not isinstance(formula,str) or not isinstance(logic,str):
        raise TypeError("formula and logic must be strings")
    if len(formula.encode()) > 65536: raise ValueError("formula exceeds migration input budget")
    try: tree=ast.parse(formula,mode="eval")
    except SyntaxError: return formula,[]
    registry=_registry()
    edits=[]; changes=[]
    lines=formula.splitlines(keepends=True)
    def offset(line,col):
        return sum(map(len,lines[:line-1]))+len(lines[line-1].encode()[:col].decode())
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name): continue
        from factor_engine.cleaned_operators.tombstones import RemovedOperatorError
        try:
            op=registry.get(node.func.id)
        except RemovedOperatorError:
            # Leave removed primitives intact so compilation records the true failure.
            continue
        if op is None: continue
        metadata=op.metadata
        specs=metadata.param_specs or {}
        aliases=metadata.param_aliases or {}
        for kw in node.keywords:
            spec=specs.get(aliases.get(kw.arg,kw.arg))
            value=kw.value
            if (spec is None or spec.dtype is not str or not spec.choices
                    or not isinstance(value,ast.Name) or value.id not in spec.choices):
                continue
            edits.append((offset(value.lineno,value.col_offset),
                          offset(value.end_lineno,value.end_col_offset),repr(value.id)))
            changes.append(f"EXACT_ENUM_LITERAL {node.func.id}.{kw.arg}: {value.id} -> quoted registered string choice")
    for start,end,text in sorted(edits,reverse=True):
        formula=formula[:start]+text+formula[end:]
    return formula,changes
