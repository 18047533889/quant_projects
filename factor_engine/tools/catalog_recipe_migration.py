"""Narrow, source-preserving migrations for catalog recipe spellings.

This module does not evaluate formulas or infer economic meaning.  It rewrites
only reviewed call signatures whose legacy and recipe semantics are identical.
Malformed or ambiguous calls are left untouched.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

_MAX_FORMULA_BYTES = 65_536
_OLD_NAME = "average_volume"
_NEW_NAME = "ts_average_volume"


@dataclass(frozen=True)
class _RecipeSpec:
    new_name: str
    old_parameters: tuple[str, ...]
    new_parameters: tuple[str, ...]
    defaults: tuple[object, ...]


_RECIPE_SPECS = {
    "ROC": _RecipeSpec("rate_of_change", ("price", "window"), ("x", "window"), (10,)),
    "MOM": _RecipeSpec("momentum", ("price", "window"), ("x", "window"), (10,)),
    "RSI": _RecipeSpec("rsi_sma", ("x", "window"), ("close", "window"), (14,)),
    "ATR": _RecipeSpec(
        "atr_sma",
        ("high", "low", "close", "window"),
        ("high", "low", "close", "window"),
        (14,),
    ),
    "BollingerUpper": _RecipeSpec(
        "bollinger_upper",
        ("x", "window", "std_dev"),
        ("x", "window", "width"),
        (20, 2),
    ),
    "BollingerLower": _RecipeSpec(
        "bollinger_lower",
        ("x", "window", "std_dev"),
        ("x", "window", "width"),
        (20, 2),
    ),
    "BollingerBands": _RecipeSpec(
        "bollinger_bands",
        ("x", "window", "std_dev"),
        ("x", "window", "width"),
        (20, 2),
    ),
}

_OFFICIAL_LIMIT_PRICE_PARAMETERS = {
    # Native intraday limit operators consume raw minute OHLC and broadcast
    # official unadjusted daily limits.  These entries also protect the exact
    # semantic-redesign output through the later adjusted-price migration.
    "intra_limit_first_hit_time": (
        "close", "high", "low", "high_limit", "low_limit"
    ),
    "intra_limit_duration": ("close", "high_limit", "low_limit"),
    "intra_limit_reopen_count": ("close", "high_limit", "low_limit"),
    "ashare_limit_open_up_streak": ("open", "high_limit"),
    "ashare_limit_open_down_streak": ("open", "low_limit"),
    "ashare_limit_one_price": ("open", "high", "low", "close", "upper_limit", "lower_limit"),
    "ashare_limit_down_streak": ("close", "low_limit"),
    "ashare_limit_touch_count": ("high", "low", "high_limit", "low_limit"),
    "ashare_limit_down_touch": ("low", "low_limit"),
    "ashare_failed_limit_count": ("high", "low", "close", "high_limit", "low_limit"),
    "ashare_limit_up_touch": ("high", "upper_limit"),
    "ashare_limit_up_streak": ("close", "high_limit"),
    "ashare_limit_failed": ("high", "close", "upper_limit"),
}


def _is_amount_amihud_recipe(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Name)
        and node.func.id == "amihud_illiquidity"
        and len(node.args) == 3
        and not node.keywords
        and not any(isinstance(argument, ast.Starred) for argument in node.args)
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "ret"
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "amount"
        and isinstance(node.args[2], ast.Constant)
        and type(node.args[2].value) is int
    )


def _is_strict_ashare_fiscal_recipe(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Name)
        and node.func.id == "fiscal_standardized_surprise"
        and len(node.args) == 2
        and not node.keywords
        and all(isinstance(argument, ast.Name) for argument in node.args)
        and node.args[1].id == "FiscalPeriodId"
        and node.args[0].id in {"OperatingRevenue_SP", "NetOperateCashFlow_SP"}
    )

_PRIMITIVE_RECIPE_TEMPLATES = {
    "MOM": lambda p: f"ts_delta({p['x']}, {p['window']})",
    "ROC": lambda p: f"multiply(ts_pct({p['x']}, {p['window']}), 100.0)",
    "BollingerUpper": lambda p: (
        f"add(ts_mean({p['x']}, {p['window']}), "
        f"multiply({p['width']}, ts_std({p['x']}, {p['window']})))"
    ),
    "BollingerLower": lambda p: (
        f"subtract(ts_mean({p['x']}, {p['window']}), "
        f"multiply({p['width']}, ts_std({p['x']}, {p['window']})))"
    ),
    # The governed legacy contract names this call BollingerBands but returns
    # one series: the middle band.  std_dev is accepted but does not affect it.
    "BollingerBands": lambda p: f"ts_mean({p['x']}, {p['window']})",
}


@dataclass(frozen=True)
class RecipeMigration:
    """A migrated formula and one audit entry per rewritten call."""

    formula: str
    changes: tuple[str, ...]


_INTRADAY_LIMIT_REDESIGN = {
    "intraday_limit_first_hit_time": (
        "intra_limit_first_hit_time",
        (
            'field("minute_close", table="StockMinuteBar")',
            'field("minute_high", table="StockMinuteBar")',
            'field("minute_low", table="StockMinuteBar")',
            'field("high_limit", table="StockDailyBar")',
            'field("low_limit", table="StockDailyBar")',
            '"up"',
        ),
        (
            "touch=HIGH_FOR_UP/LOW_FOR_DOWN",
            "clock=PHYSICAL_OFFICIAL_SESSION_SLOTS",
            "normalization=slot_id/n_slots",
        ),
    ),
    "intraday_limit_duration": (
        "intra_limit_duration",
        (
            'field("minute_close", table="StockMinuteBar")',
            'field("high_limit", table="StockDailyBar")',
            'field("low_limit", table="StockDailyBar")',
            '"up"',
        ),
        ("touch=CLOSE_ONLY", "clock=REGISTERED_STOCK_MINUTE_BAR_SESSION_GRID"),
    ),
    "intraday_limit_reopen_count": (
        "intra_limit_reopen_count",
        (
            'field("minute_close", table="StockMinuteBar")',
            'field("high_limit", table="StockDailyBar")',
            'field("low_limit", table="StockDailyBar")',
            '"up"',
            '"open"',
        ),
        (
            "touch=CLOSE_ONLY",
            "clock=REGISTERED_STOCK_MINUTE_BAR_SESSION_GRID",
            "transition=open:touch_to_non_touch",
        ),
    ),
}


def redesign_catalog_intraday_limit_formula(
    formula: str, *, enabled: bool = False
) -> RecipeMigration:
    """Opt-in semantic redesign of the exact legacy three-argument calls.

    This is intentionally separate from strict recipe migration: it changes
    touch, clock and transition semantics to the current native canonical
    contract under explicit user authorization.  Ambiguous signatures fail
    closed and the ordinary migration API remains unchanged by default.
    """

    if not enabled:
        return RecipeMigration(formula, ())
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")
    tree = ast.parse(formula, mode="eval")
    lines = formula.splitlines(keepends=True) or [""]
    edits: list[tuple[int, int, str]] = []
    changes: list[str] = []
    expected = ("MinuteClose", "HighLimit", "LowLimit")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        spec = _INTRADAY_LIMIT_REDESIGN.get(node.func.id)
        if spec is None or node.keywords or len(node.args) != 3:
            continue
        if not all(
            isinstance(arg, ast.Name) and arg.id == name
            for arg, name in zip(node.args, expected)
        ):
            continue
        canonical, arguments, decisions = spec
        start = _character_offset(lines, node.lineno, node.col_offset)
        end = _character_offset(lines, node.end_lineno, node.end_col_offset)
        edits.append((start, end, f"{canonical}({', '.join(arguments)})"))
        changes.append(
            "SEMANTIC_REDESIGN "
            f"{node.func.id}(MinuteClose,HighLimit,LowLimit) -> {canonical}; "
            "price_basis=RAW_OFFICIAL; side=up; " + "; ".join(decisions)
        )
    migrated = formula
    for start, end, replacement in sorted(edits, reverse=True):
        migrated = migrated[:start] + replacement + migrated[end:]
    return RecipeMigration(migrated, tuple(changes))


def _character_offset(lines: list[str], lineno: int, byte_col: int) -> int:
    """Convert an AST UTF-8 byte column into a string character offset."""

    if lineno < 1 or lineno > len(lines):
        raise ValueError("AST location is outside the formula")
    line = lines[lineno - 1]
    prefix = line.encode("utf-8")[:byte_col]
    try:
        column = len(prefix.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("AST location splits a UTF-8 code point") from exc
    return sum(len(item) for item in lines[: lineno - 1]) + column


def migrate_average_volume_calls(formula: str) -> RecipeMigration:
    """Rename only bare ``average_volume`` call targets.

    Parameters, keywords, whitespace, strings, field identifiers and attribute
    calls are never rewritten. Invalid Python-expression syntax fails closed
    with ``SyntaxError``. In particular, a legacy five-OHLCV call keeps all
    five arguments and therefore remains invalid against the two-parameter
    ``ts_average_volume(volume, window)`` contract.
    """

    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")

    tree = ast.parse(formula, mode="eval")
    spans: list[tuple[int, int]] = []
    lines = formula.splitlines(keepends=True) or [""]

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Name) or function.id != _OLD_NAME:
            continue
        if function.end_lineno is None or function.end_col_offset is None:
            raise ValueError("AST call target has no complete source location")
        start = _character_offset(lines, function.lineno, function.col_offset)
        end = _character_offset(lines, function.end_lineno, function.end_col_offset)
        if formula[start:end] != _OLD_NAME:
            raise ValueError("AST/source disagreement for average_volume call target")
        spans.append((start, end))

    migrated = formula
    for start, end in sorted(spans, reverse=True):
        migrated = migrated[:start] + _NEW_NAME + migrated[end:]
    return RecipeMigration(
        formula=migrated,
        changes=tuple(f"{_OLD_NAME} -> {_NEW_NAME}" for _ in spans),
    )


def migrate_catalog_recipe_formula(formula: str) -> RecipeMigration:
    """Apply the complete reviewed catalog-recipe migration set.

    Order matters: the R19 operator-recipe pass runs first because it is the
    only pass that recognises the *legacy* generic OHLCV call shape
    (``Indicator(open, high, low, close, volume)`` -- five positional bare
    ``field(...)`` leaves).  Every indicator it rewrites is replaced by a
    signature with a different arity, so the positional recipe specs below
    (which match the short ``Indicator(price, window)`` spellings) cannot
    re-fire on the R19 output and the composition stays idempotent.

    Confining the repair to this chain is the point: while the R19 pass lived
    in its own module and was never imported here, the R57 compile chain
    silently shipped 72 catalog rows still carrying the un-runnable legacy
    call, with an empty ``migration_changes`` audit trail.
    """

    from factor_engine.tools.catalog_r19_operator_recipes import migrate_formula as _migrate_r19

    r19_formula, r19_changes = _migrate_r19(formula)
    initial = migrate_average_volume_calls(r19_formula)
    source = initial.formula
    tree = ast.parse(source, mode="eval")
    lines = source.splitlines(keepends=True) or [""]
    edits: list[tuple[int, int, str]] = []
    changes: list[str] = [*r19_changes, *initial.changes]

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        old_name = node.func.id
        if _is_strict_ashare_fiscal_recipe(node):
            table, field_name = {
                "OperatingRevenue_SP": ("StockIncome", "operating_revenue"),
                "NetOperateCashFlow_SP": ("StockCashFlow", "operating_cash_flow"),
            }[node.args[0].id]
            period = f'field("report_period_end_date", table="{table}")'
            cumulative = f'field("{field_name}", table="{table}")'
            replacement = (
                "fiscal_standardized_surprise("
                f"fin_quarter_from_cumulative({cumulative}, {period}, "
                f"ashare_fiscal_quarter_from_period_end({period})), {period})"
            )
            start = _character_offset(lines, node.lineno, node.col_offset)
            end = _character_offset(lines, node.end_lineno, node.end_col_offset)
            edits.append((start, end, replacement))
            changes.append(
                f"{node.args[0].id}/FiscalPeriodId -> strict {table} single-period recipe"
            )
            continue
        if _is_amount_amihud_recipe(node):
            start = _character_offset(lines, node.func.lineno, node.func.col_offset)
            end = _character_offset(
                lines, node.func.end_lineno, node.func.end_col_offset
            )
            edits.append((start, end, "price_impact"))
            changes.append(
                "amihud_illiquidity(ret, amount, window) -> "
                "price_impact(ret, amount, window)"
            )
        if old_name in _OFFICIAL_LIMIT_PRICE_PARAMETERS:
            # Exchange-limit predicates require official, unadjusted prices.
            # Only rewrite direct catalog price leaves, never infer an arbitrary
            # expression's basis or change prices elsewhere in the factor.
            price_parameters = _OFFICIAL_LIMIT_PRICE_PARAMETERS[old_name]
            leaves = [
                *node.args[:len(price_parameters)],
                *(kw.value for kw in node.keywords if kw.arg in price_parameters),
            ]
            for leaf in leaves:
                if isinstance(leaf, ast.Name) and leaf.id in {
                    "open", "high", "low", "close", "pre_close",
                    "high_limit", "low_limit",
                }:
                    start = _character_offset(lines, leaf.lineno, leaf.col_offset)
                    end = _character_offset(lines, leaf.end_lineno, leaf.end_col_offset)
                    replacement = f'field("{leaf.id}", table="StockDailyBar")'
                    edits.append((start, end, replacement))
                    changes.append(f"{old_name}.{leaf.id} -> StockDailyBar official raw price")
        if old_name == "ts_abdi_ranaldo_spread":
            malformed_ohlc = (
                len(node.args) == 5
                and not node.keywords
                and not any(isinstance(arg, ast.Starred) for arg in node.args)
                and all(
                    isinstance(argument, ast.Name) and argument.id == expected
                    for argument, expected in zip(node.args[:4], ("open", "high", "low", "close"))
                )
                and isinstance(node.args[4], ast.Constant)
                and type(node.args[4].value) is int
            )
            if malformed_ohlc:
                start = _character_offset(lines, node.lineno, node.col_offset)
                end = _character_offset(lines, node.end_lineno, node.end_col_offset)
                window_start = _character_offset(
                    lines, node.args[4].lineno, node.args[4].col_offset
                )
                window_end = _character_offset(
                    lines, node.args[4].end_lineno, node.args[4].end_col_offset
                )
                window = source[window_start:window_end]
                edits.append(
                    (start, end, f"ts_abdi_ranaldo_spread(close, high, low, {window})")
                )
                changes.append(
                    "ts_abdi_ranaldo_spread malformed OHLC signature -> close/high/low signature"
                )
            # The governed kernel fixes correction="monthly". Remove only the
            # exact legacy spelling of that default, never a different policy,
            # duplicate binding, unpacking, or unknown trailing arguments.
            policy = None
            if len(node.args) == 5 and not node.keywords:
                policy = node.args[4]
            elif (
                len(node.args) == 4 and len(node.keywords) == 1
                and node.keywords[0].arg == "correction"
            ):
                policy = node.keywords[0].value
            if (
                isinstance(policy, ast.Constant) and policy.value == "monthly"
                and not any(isinstance(arg, ast.Starred) for arg in node.args)
            ):
                previous = node.args[3]
                start = _character_offset(lines, previous.end_lineno, previous.end_col_offset)
                end = _character_offset(lines, policy.end_lineno, policy.end_col_offset)
                edits.append((start, end, ""))
                changes.append("ts_abdi_ranaldo_spread explicit monthly -> fixed monthly default")
        if old_name == "ts_spectral_entropy":
            # Opt in only when the catalog spelling itself proves that the
            # immediate input is the governed turnover activity field.  Do not
            # infer a semantic kind for arbitrary expressions or attributes.
            direct_input: ast.expr | None = None
            x_keywords = [keyword for keyword in node.keywords if keyword.arg == "x"]
            if len(node.args) >= 1 and not x_keywords:
                direct_input = node.args[0]
            elif not node.args and len(x_keywords) == 1:
                direct_input = x_keywords[0].value
            if (
                isinstance(direct_input, ast.Name)
                and direct_input.id in {"turnover_ratio", "TurnoverRatio"}
                and not any(isinstance(argument, ast.Starred) for argument in node.args)
                and all(keyword.arg is not None for keyword in node.keywords)
            ):
                start = _character_offset(lines, node.func.lineno, node.func.col_offset)
                end = _character_offset(lines, node.func.end_lineno, node.func.end_col_offset)
                if source[start:end] != old_name:
                    raise ValueError("AST/source disagreement for activity spectral entropy")
                edits.append((start, end, "ts_activity_spectral_entropy"))
                changes.append(
                    "ts_spectral_entropy(turnover activity) -> ts_activity_spectral_entropy"
                )
        if old_name == "ts_hill_tail_index":
            # The catalog uses min_tail for the documented minimum tail sample
            # count. Never hide duplicate canonical+legacy bindings.
            names = [keyword.arg for keyword in node.keywords]
            if names.count("min_tail") == 1 and "min_tail_count" not in names and None not in names and len(node.args) < 5:
                keyword = next(k for k in node.keywords if k.arg == "min_tail")
                start = _character_offset(lines, keyword.lineno, keyword.col_offset)
                if source[start:start + len("min_tail")] != "min_tail":
                    raise ValueError("AST/source disagreement for tail keyword")
                edits.append((start, start + len("min_tail"), "min_tail_count"))
                changes.append("ts_hill_tail_index.min_tail -> min_tail_count")
        if old_name == "ts_markov_persistence":
            # markov_dynamics documents min_periods/min_count as the same
            # transition-count floor; min_history is a distinct maturity gate.
            # Keep collisions for the validator instead of hiding a duplicate.
            names = [keyword.arg for keyword in node.keywords]
            if (
                names.count("min_periods") == 1
                and "min_count" not in names
                and None not in names
                and len(node.args) < 5
            ):
                keyword = next(k for k in node.keywords if k.arg == "min_periods")
                start = _character_offset(lines, keyword.lineno, keyword.col_offset)
                if source[start : start + len("min_periods")] != "min_periods":
                    raise ValueError("AST/source disagreement for Markov keyword")
                edits.append((start, start + len("min_periods"), "min_count"))
                changes.append("ts_markov_persistence.min_periods -> min_count")
        boolean_slot = {
            "ts_regression_tstat": ("add_intercept", 4),
            "ts_regression_intercept": ("add_intercept", 4),
            "cs_bucket": ("ascending", 2),
        }.get(old_name)
        if boolean_slot is not None:
            # Reviewed legacy kernels applied bool(value); only literal integer
            # 0/1 have an exact spelling migration. Preserve ambiguous bindings.
            parameter, position = boolean_slot
            names = [keyword.arg for keyword in node.keywords]
            target: ast.expr | None = None
            if len(node.args) == position + 1 and parameter not in names and None not in names:
                target = node.args[position]
            elif len(node.args) <= position and names.count(parameter) == 1 and None not in names:
                target = next(
                    keyword.value for keyword in node.keywords
                    if keyword.arg == parameter
                )
            if (
                isinstance(target, ast.Constant)
                and type(target.value) is int
                and target.value in {0, 1}
            ):
                start = _character_offset(lines, target.lineno, target.col_offset)
                end = _character_offset(lines, target.end_lineno, target.end_col_offset)
                literal = source[start:end]
                if literal in {"0", "1"}:
                    replacement = "True" if target.value == 1 else "False"
                    edits.append((start, end, replacement))
                    changes.append(f"{old_name}.{parameter} {literal} -> {replacement}")
        financial_arity = {
            "current_ratio": 2,
            "debt_to_equity": 2,
            "operating_margin": 2,
            "quick_ratio": 3,
        }
        if (
            old_name in financial_arity
            and len(node.args) == financial_arity[old_name]
            and not node.keywords
            and all(isinstance(argument, ast.Name) for argument in node.args)
        ):
            func_start = _character_offset(lines, node.func.lineno, node.func.col_offset)
            if old_name != "quick_ratio":
                edits.append((func_start, func_start + len(old_name), "fin_ratio"))
            else:
                call_end = _character_offset(lines, node.end_lineno, node.end_col_offset)
                arguments = []
                for argument in node.args:
                    start = _character_offset(lines, argument.lineno, argument.col_offset)
                    end = _character_offset(lines, argument.end_lineno, argument.end_col_offset)
                    arguments.append(source[start:end])
                replacement = (
                    f"fin_ratio(subtract({arguments[0]}, {arguments[1]}), {arguments[2]})"
                )
                edits.append((func_start, call_end, replacement))
            changes.append(f"{old_name} -> reviewed fundamental ratio recipe")
        spec = _RECIPE_SPECS.get(old_name)
        if spec is None or any(isinstance(arg, ast.Starred) for arg in node.args):
            continue
        if len(node.args) > len(spec.old_parameters):
            continue
        bound = set(spec.old_parameters[: len(node.args)])
        keyword_nodes: dict[str, ast.keyword] = {}
        valid = True
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg not in spec.old_parameters:
                valid = False
                break
            if keyword.arg in bound:
                valid = False
                break
            bound.add(keyword.arg)
            keyword_nodes[keyword.arg] = keyword
        required_count = len(spec.old_parameters) - len(spec.defaults)
        if not valid or any(name not in bound for name in spec.old_parameters[:required_count]):
            continue

        primitive_template = _PRIMITIVE_RECIPE_TEMPLATES.get(old_name)
        if primitive_template is not None:
            # Whole-call expansion is safe only when it cannot overlap another
            # reviewed legacy recipe edit.  Nested recipe calls remain unchanged
            # for a later explicit migration rather than risking crossed spans.
            descendants = [
                child for child in ast.walk(node)
                if child is not node
                and isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and (
                    child.func.id in _RECIPE_SPECS
                    or child.func.id == "ts_abdi_ranaldo_spread"
                    or child.func.id in _OFFICIAL_LIMIT_PRICE_PARAMETERS
                    or _is_amount_amihud_recipe(child)
                    or _is_strict_ashare_fiscal_recipe(child)
                )
            ]
            if descendants:
                continue
            values: dict[str, str] = {}
            for index, argument in enumerate(node.args):
                parameter = spec.new_parameters[index]
                start = _character_offset(lines, argument.lineno, argument.col_offset)
                end = _character_offset(lines, argument.end_lineno, argument.end_col_offset)
                values[parameter] = source[start:end]
            for old_parameter, keyword in keyword_nodes.items():
                parameter = spec.new_parameters[spec.old_parameters.index(old_parameter)]
                start = _character_offset(lines, keyword.value.lineno, keyword.value.col_offset)
                end = _character_offset(lines, keyword.value.end_lineno, keyword.value.end_col_offset)
                values[parameter] = source[start:end]
            for index in range(required_count, len(spec.old_parameters)):
                parameter = spec.new_parameters[index]
                if parameter not in values:
                    values[parameter] = repr(spec.defaults[index - required_count])
            call_start = _character_offset(lines, node.lineno, node.col_offset)
            call_end = _character_offset(lines, node.end_lineno, node.end_col_offset)
            edits.append((call_start, call_end, primitive_template(values)))
            changes.append(f"{old_name} -> reviewed primitive recipe")
            continue

        func_start = _character_offset(lines, node.func.lineno, node.func.col_offset)
        func_end = _character_offset(lines, node.func.end_lineno, node.func.end_col_offset)
        edits.append((func_start, func_end, spec.new_name))

        for old_parameter, keyword in keyword_nodes.items():
            new_parameter = spec.new_parameters[spec.old_parameters.index(old_parameter)]
            if new_parameter == old_parameter:
                continue
            name_start = _character_offset(lines, keyword.lineno, keyword.col_offset)
            if name_start < 0 or source[name_start : name_start + len(old_parameter)] != old_parameter:
                raise ValueError("AST/source disagreement for recipe keyword")
            edits.append((name_start, name_start + len(old_parameter), new_parameter))

        missing_optional = []
        for index in range(required_count, len(spec.old_parameters)):
            if spec.old_parameters[index] not in bound:
                default = spec.defaults[index - required_count]
                missing_optional.append(f"{spec.new_parameters[index]}={default!r}")
        if missing_optional:
            # Insert immediately after the last argument, before any existing
            # trailing comma/comment. Appending before ')' would produce ',,'.
            last = max([*node.args, *node.keywords], key=lambda item: (item.end_lineno, item.end_col_offset))
            insert_at = _character_offset(lines, last.end_lineno, last.end_col_offset)
            edits.append((insert_at, insert_at, ", " + ", ".join(missing_optional)))
        changes.append(f"{old_name} -> {spec.new_name}")

    migrated = source
    for start, end, replacement in sorted(edits, reverse=True):
        migrated = migrated[:start] + replacement + migrated[end:]
    return RecipeMigration(migrated, tuple(changes))
