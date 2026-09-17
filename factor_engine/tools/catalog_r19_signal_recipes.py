"""Explicit construction of legacy signal shorthand, not fabricated raw data."""
import ast

_SIGNALS = {
    "ret_1d": ("field('ret', table='StockDailyBarAdj')",
               "EXACT_FIELD ret_1d: canonical adjusted one-day decimal return"),
    "rel_volume": ("relative_volume(field('volume', table='StockDailyBarAdj'), 20)",
                   "SEMANTIC_REDESIGN rel_volume: construct 20-day relative volume from observed adjusted-bar volume"),
    "amihud": ("amihud_illiquidity(field('ret', table='StockDailyBarAdj'), field('close', table='StockDailyBarAdj'), field('volume', table='StockDailyBarAdj'), 20)",
               "SEMANTIC_REDESIGN amihud: construct canonical 20-day Amihud illiquidity from observed return/close/volume"),
}

def migrate_formula(formula: str, logic: str = "") -> tuple[str,list[str]]:
    if not isinstance(formula,str) or not isinstance(logic,str):
        raise TypeError("formula and logic must be strings")
    if len(formula.encode())>65536: raise ValueError("formula exceeds migration input budget")
    try: tree=ast.parse(formula,mode="eval")
    except SyntaxError: return formula,[]
    parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    lines=formula.splitlines(keepends=True); edits=[]; changes=[]
    def off(line,col):
        return sum(map(len,lines[:line-1]))+len(lines[line-1].encode()[:col].decode())
    for node in ast.walk(tree):
        if not isinstance(node,ast.Name) or node.id not in _SIGNALS: continue
        parent=parents.get(node)
        if isinstance(parent,ast.Call) and parent.func is node: continue
        if isinstance(parent,ast.Attribute): continue
        new,note=_SIGNALS[node.id]
        edits.append((off(node.lineno,node.col_offset),off(node.end_lineno,node.end_col_offset),new))
        changes.append(note)
    for a,b,new in sorted(edits,reverse=True): formula=formula[:a]+new+formula[b:]
    return formula,changes
