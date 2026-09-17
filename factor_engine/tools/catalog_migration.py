"""Explicit A-share raw-price recipe migration; never evaluate catalog text.

Only source spellings change. Operator names, numeric parameters, fiscal fields,
and economic formulas are not inferred or replaced.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass

_PRICE_FIELDS = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "PreClose": "pre_close", "Vwap": "vwap", "Amount": "amount",
    "Volume": "volume", "Return": "ret", "Factor": "adj_factor",
    "HighLimit": "high_limit", "LowLimit": "low_limit",
}
_RAW_TABLES = frozenset({"StockDailyBar", "DailyBar"})
_ADJ_TABLE = "StockDailyBarAdj"

@dataclass(frozen=True)
class PriceMigration:
    formula: str
    changes: tuple[str, ...]

def migrate_adjusted_price_fields(formula: str, *, market: str) -> PriceMigration:
    """Migrate explicit old A-share daily prices to current adjusted contracts.

    Bare lowercase fields already resolve through the current DSL authority.
    Explicit raw requests in this *opt-in migration* are changed by intent;
    runtime field resolution itself is not changed. This is not a US migration.
    """
    if market != "ashare":
        raise ValueError("adjusted-price catalog migration requires market='ashare'")
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if len(formula) > 65536 or len(formula.encode("utf-8")) > 65536:
        raise ValueError("formula exceeds migration input budget")
    tree = ast.parse(formula, mode="eval")
    changes = []
    def adjusted(name, old):
        canonical = _PRICE_FIELDS.get(name, name)
        changes.append(f"{old} -> {_ADJ_TABLE}.{canonical}")
        return ast.Call(func=ast.Name(id="field", ctx=ast.Load()),
                        args=[ast.Constant(canonical)],
                        keywords=[ast.keyword(arg="table", value=ast.Constant(_ADJ_TABLE))])
    # These exact contracts require exchange-price basis, not adjusted prices.
    from factor_engine.tools.catalog_recipe_migration import _OFFICIAL_LIMIT_PRICE_PARAMETERS
    raw_prices = {"open", "high", "low", "close", "pre_close", "high_limit", "low_limit"}

    def official_leaf(node):
        name = None
        if isinstance(node, ast.Name) and node.id in _PRICE_FIELDS:
            name = node.id
        elif isinstance(node, ast.Attribute):
            owner = node.value
            if isinstance(owner, ast.Call) and not owner.args and not owner.keywords:
                owner = owner.func
            if isinstance(owner, ast.Name) and owner.id in _RAW_TABLES:
                name = node.attr
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "field" and len(node.args) == 1
              and isinstance(node.args[0], ast.Constant)):
            table = next((kw.value for kw in node.keywords if kw.arg == "table"), None)
            if isinstance(table, ast.Constant) and table.value in _RAW_TABLES:
                canonical = _PRICE_FIELDS.get(node.args[0].value, node.args[0].value)
                if canonical in raw_prices:
                    if canonical != node.args[0].value:
                        changes.append(f"{ast.unparse(node)} -> StockDailyBar.{canonical}")
                        node.args[0] = ast.Constant(canonical)
                    return node
        canonical = _PRICE_FIELDS.get(name, name)
        if canonical not in raw_prices:
            return None
        changes.append(f"{ast.unparse(node)} -> StockDailyBar.{canonical} (official price)")
        return ast.copy_location(
            ast.Call(func=ast.Name(id="field", ctx=ast.Load()),
                     args=[ast.Constant(canonical)],
                     keywords=[ast.keyword(arg="table", value=ast.Constant("StockDailyBar"))]),
            node,
        )

    class Rewrite(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in _PRICE_FIELDS:
                return ast.copy_location(adjusted(node.id, node.id), node)
            return node

        def visit_Attribute(self, node):
            owner = node.value
            if isinstance(owner, ast.Call) and not owner.args and not owner.keywords:
                owner = owner.func
            if isinstance(owner, ast.Name) and owner.id in _RAW_TABLES and node.attr in _PRICE_FIELDS:
                return ast.copy_location(adjusted(node.attr, f"{owner.id}.{node.attr}"), node)
            # Qualified non-price data are not changed into raw-name fields.
            return node

        def visit_Call(self, node):
            # Do not interpret a function name as a field name.
            fn = node.func.id if isinstance(node.func, ast.Name) else None
            if fn in {"col", "field"} and len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                name = node.args[0].value
                options = {kw.arg: kw.value for kw in node.keywords}
                table = options.get("table")
                explicit_raw = isinstance(table, ast.Constant) and table.value in _RAW_TABLES
                known = name in _PRICE_FIELDS or name in _PRICE_FIELDS.values()
                # Preserve any extra control flags; don't drop strict/mining intent.
                unqualified_old = table is None and name in _PRICE_FIELDS and None not in options
                if known and (explicit_raw or unqualified_old) and (fn != "col" or not options):
                    canonical = _PRICE_FIELDS.get(name, name)
                    changes.append(f"{ast.unparse(node)} -> {_ADJ_TABLE}.{canonical}")
                    node.func = ast.Name(id="field", ctx=ast.Load())
                    node.args = [ast.Constant(canonical)]
                    node.keywords = [ast.keyword(arg=kw.arg, value=ast.Constant(_ADJ_TABLE))
                                     if kw.arg == "table" else kw for kw in node.keywords]
                    if "table" not in options:
                        node.keywords.append(ast.keyword(arg="table", value=ast.Constant(_ADJ_TABLE)))
                    return node
            official = _OFFICIAL_LIMIT_PRICE_PARAMETERS.get(fn, ())
            def visit_input(value, is_official):
                protected = official_leaf(value) if is_official else None
                return protected if protected is not None else self.visit(value)
            node.args = [visit_input(arg, i < len(official)) for i, arg in enumerate(node.args)]
            node.keywords = [ast.keyword(arg=kw.arg, value=visit_input(kw.value, kw.arg in official))
                             for kw in node.keywords]
            return node
    new_tree = Rewrite().visit(tree)
    ast.fix_missing_locations(new_tree)
    return PriceMigration(ast.unparse(new_tree) if changes else formula, tuple(dict.fromkeys(changes)))
