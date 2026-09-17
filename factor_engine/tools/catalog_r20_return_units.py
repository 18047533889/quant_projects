"""Remove obsolete bps conversion only for explicitly bound adjusted-stock returns."""
import ast

def _ratio_return(node):
    if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!="field":
        return False
    return (len(node.args)==1 and isinstance(node.args[0],ast.Constant)
        and node.args[0].value=="ret" and len(node.keywords)==1
        and node.keywords[0].arg=="table" and isinstance(node.keywords[0].value,ast.Constant)
        and node.keywords[0].value.value=="StockDailyBarAdj")

def migrate_formula(formula,logic=""):
    if len(formula.encode())>65536: raise ValueError("formula exceeds input budget")
    try: tree=ast.parse(formula,mode="eval")
    except SyntaxError: return formula,[]
    changes=[]
    class Repair(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if (isinstance(node.func,ast.Name) and node.func.id in {"safe_div_null","safe_div","divide"}
                and len(node.args)==2 and not node.keywords and _ratio_return(node.args[0])
                and isinstance(node.args[1],ast.Constant) and type(node.args[1].value) in {int,float}
                and node.args[1].value==10000):
                changes.append("UNIT_CORRECTION: StockDailyBarAdj.ret is already ratio after field binding (source bps * 0.0001); removed duplicate /10000, preserving the adjusted return field.")
                return node.args[0]
            return node
    tree=Repair().visit(tree)
    return (ast.unparse(ast.fix_missing_locations(tree)),changes) if changes else (formula,[])
