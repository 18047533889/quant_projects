"""R19 recipes for reviewed bare intermediate-signal placeholders."""
from __future__ import annotations

import ast

_MAX_FORMULA_BYTES = 65_536
_DEFINITIONS = {
    "atr_pct": (
        "atr_pct(high, low, close, window=20)",
        "SEMANTIC_DEFINITION atr_pct := atr_pct(high, low, close, window=20) "
        "using adjusted daily OHLC and a fixed 20-session horizon",
    ),
    "overnight_ret": (
        "overnight_return(open, pre_close)",
        "SEMANTIC_DEFINITION overnight_ret := overnight_return(open, pre_close) "
        "using adjusted daily open and prior close",
    ),
    "intraday_ret": (
        "open_close_return(open, close)",
        "SEMANTIC_DEFINITION intraday_ret := open_close_return(open, close) "
        "using adjusted daily open and close",
    ),
    "vwap_dist": (
        "vwap_distance_pct(close, volume, window=20)",
        "SEMANTIC_DEFINITION vwap_dist := vwap_distance_pct(close, volume, window=20) "
        "using adjusted close, daily volume, and a fixed 20-session horizon",
    ),
    "rolling_vwap": (
        "rolling_vwap(close, volume, window=20)",
        "SEMANTIC_DEFINITION rolling_vwap placeholder := rolling_vwap(close, volume, "
        "window=20) using adjusted close and daily volume",
    ),
    "volume_shock": (
        "gt(volume_shock(volume, window=20), 2.0)",
        "SEMANTIC_DEFINITION volume_shock event := gt(volume_shock(volume, window=20), "
        "2.0), a formal MaskBool for volume z-score above two",
    ),
    "turnover_shock": (
        "gt(turnover_shock(turnover_ratio, window=20), 2.0)",
        "SEMANTIC_DEFINITION turnover_shock event := "
        "gt(turnover_shock(turnover_ratio, window=20), 2.0), a formal MaskBool for "
        "turnover-ratio z-score above two",
    ),
}

_EVENT_DEFINITIONS = {
    "limit_up_touch": (
        "ashare_limit_up_touch(field('high', table='StockDailyBar'), "
        "field('high_limit', table='StockDailyBar'), 0.005)",
        "SEMANTIC_DEFINITION limit_up_touch event := raw intraday high touches the "
        "official raw upper limit within CNY 0.005",
    ),
    "limit_down_touch": (
        "ashare_limit_down_touch(field('low', table='StockDailyBar'), "
        "field('low_limit', table='StockDailyBar'), 0.005)",
        "SEMANTIC_DEFINITION limit_down_touch event := raw intraday low touches the "
        "official raw lower limit within CNY 0.005",
    ),
    "failed_limit": (
        "ashare_limit_failed(field('high', table='StockDailyBar'), "
        "field('close', table='StockDailyBar'), "
        "field('high_limit', table='StockDailyBar'), 0.005)",
        "SEMANTIC_DEFINITION failed_limit event := raw high touches the official raw "
        "upper limit but raw close does not hold it, with CNY 0.005 tolerance",
    ),
    "gap_up": (
        "gt(overnight_return(open, pre_close), 0.0)",
        "SEMANTIC_DEFINITION gap_up event := adjusted overnight_return(open, pre_close) "
        "> 0.0; zero is the explicit threshold, so any strictly positive gap qualifies",
    ),
    "gap_down": (
        "lt(overnight_return(open, pre_close), 0.0)",
        "SEMANTIC_DEFINITION gap_down event := adjusted overnight_return(open, pre_close) "
        "< 0.0; zero is the explicit threshold, so any strictly negative gap qualifies",
    ),
    "suspension_resume": (
        "and_(eq(is_suspend, 0.0), eq(delay(is_suspend, 1), 1.0))",
        "SEMANTIC_DEFINITION suspension_resume event := is_suspend is false today and "
        "was true on the immediately prior trading row; uses lagged state only",
    ),
    "st_transition": (
        "ne(is_st, delay(is_st, 1))",
        "SEMANTIC_DEFINITION st_transition event := current is_st differs from its "
        "immediately prior trading-row value; uses no future state",
    ),
}

_EVENT_INPUT_OPERATORS = frozenset({
    "event_historical_response_mean", "event_historical_response_sign_balance",
    "event_response_peak_lag", "event_response_decay_rate",
    "event_response_dispersion", "event_response_reversal_strength",
    "event_hawkes_branching_ratio_proxy",
})

_MALFORMED_LIMIT_CALLS = {
    "ashare_limit_up_touch": (
        (("open", "StockDailyBar"), ("close", "StockDailyBar"),
         ("high_limit", "StockDailyBarAdj"), ("low_limit", "StockDailyBarAdj")),
        _EVENT_DEFINITIONS["limit_up_touch"][0],
        "SEMANTIC_REPAIR malformed OHLC limit-up template := formal raw high, raw "
        "official upper limit, and CNY 0.005 tolerance",
    ),
    "ashare_limit_down_touch": (
        (("open", "StockDailyBar"), ("close", "StockDailyBar"),
         ("high_limit", "StockDailyBarAdj"), ("low_limit", "StockDailyBarAdj"),
         ("volume", "StockDailyBarAdj")),
        _EVENT_DEFINITIONS["limit_down_touch"][0],
        "SEMANTIC_REPAIR malformed OHLCV limit-down template := formal raw low, raw "
        "official lower limit, and CNY 0.005 tolerance",
    ),
    "ashare_limit_failed": (
        (("open", "StockDailyBar"), ("close", "StockDailyBar"),
         ("high_limit", "StockDailyBar"), ("low_limit", "StockDailyBarAdj"),
         ("volume", "StockDailyBarAdj")),
        _EVENT_DEFINITIONS["failed_limit"][0],
        "SEMANTIC_REPAIR malformed OHLCV failed-limit template := formal raw high, "
        "raw close, raw official upper limit, and CNY 0.005 tolerance",
    ),
}


def _field_ref(node: ast.AST) -> tuple[str, str] | None:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "field" and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str) and len(node.keywords) == 1
            and node.keywords[0].arg == "table"
            and isinstance(node.keywords[0].value, ast.Constant)
            and isinstance(node.keywords[0].value.value, str)):
        return None
    return node.args[0].value, node.keywords[0].value.value


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_col]
    return sum(len(line) for line in lines[: lineno - 1]) + len(prefix.decode("utf-8"))


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST node has no complete source location")
    return (_offset(lines, node.lineno, node.col_offset),
            _offset(lines, node.end_lineno, node.end_col_offset))


def migrate_formula(formula: str, logic: str = "") -> tuple[str, list[str]]:
    """Expand only exact bare placeholders into audited, formal expressions."""
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(logic, str):
        raise TypeError("logic must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return formula, []

    function_nodes = {
        id(node.func) for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    lines = formula.splitlines(keepends=True) or [""]
    edits: list[tuple[int, int, str]] = []
    changed_names: list[str] = []
    direct_changes: list[str] = []
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            continue
        repair = _MALFORMED_LIMIT_CALLS.get(call.func.id)
        if repair is None or call.keywords:
            continue
        expected, replacement, change = repair
        if tuple(_field_ref(arg) for arg in call.args) != expected:
            continue
        start, end = _span(lines, call)
        edits.append((start, end, replacement))
        if change not in direct_changes:
            direct_changes.append(change)
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id in _EVENT_INPUT_OPERATORS
        ):
            continue
        event_nodes = [kw.value for kw in call.keywords if kw.arg == "event"]
        # Positional event slots are intentionally not guessed here: the reviewed
        # R19 shapes use event= explicitly, which makes the boolean role certain.
        for event_node in event_nodes:
            if not isinstance(event_node, ast.Name):
                continue
            definition = _EVENT_DEFINITIONS.get(event_node.id)
            if definition is None:
                continue
            start, end = _span(lines, event_node)
            edits.append((start, end, definition[0]))
            if event_node.id not in changed_names:
                changed_names.append(event_node.id)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or id(node) in function_nodes:
            continue
        definition = _DEFINITIONS.get(node.id)
        if definition is None:
            continue
        start, end = _span(lines, node)
        edits.append((start, end, definition[0]))
        if node.id not in changed_names:
            changed_names.append(node.id)

    migrated = formula
    for start, end, replacement in sorted(edits, reverse=True):
        migrated = migrated[:start] + replacement + migrated[end:]
    return migrated, direct_changes + [
        (_DEFINITIONS.get(name) or _EVENT_DEFINITIONS[name])[1]
        for name in changed_names
    ]
