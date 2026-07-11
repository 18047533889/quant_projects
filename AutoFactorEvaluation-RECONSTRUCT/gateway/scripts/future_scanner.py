"""
未来函数拦截器模块

负责:
- 检测因子表达式中是否包含未来函数
- 检测时间偏移中的正向引用（如 t+n）
- 检测特定模式的未来数据泄露
"""

import re
from typing import List, Tuple, Optional, Set, Dict, Any
from enum import Enum

from .config import GatewayConfig
from .exceptions import FutureFunctionError


class FutureFunctionSeverity(Enum):
    """未来函数严重程度"""
    CRITICAL = "critical"
    WARNING = "warning"


class FutureScanResult:
    """未来函数扫描结果"""

    def __init__(self, has_future: bool, violation_detail: str = "",
                 severity: FutureFunctionSeverity = FutureFunctionSeverity.CRITICAL,
                 matched_patterns: List[str] = None):
        self.has_future = has_future
        self.violation_detail = violation_detail
        self.severity = severity
        self.matched_patterns = matched_patterns or []

    def __repr__(self):
        return f"FutureScanResult(has_future={self.has_future}, detail='{self.violation_detail[:50]}...')"


class FutureFunctionScanner:
    """
    未来函数扫描器

    检测因子表达式中的未来函数和数据泄露
    """

    # 未来函数黑名单（函数名）
    DEFAULT_FUTURE_FUNCTIONS = {
        "REFX", "PEEK", "FUTURE", "LEAD", "SHIFT_FORWARD",
        "FWD_LOOK", "TS_LEAD", "NEXT", "FORWARD", "LOOKAHEAD",
        "FUTURE_VALUE", "AHEAD", "FORWARD_LOOKING"
    }

    # 正向偏移正则模式（修复版）
    POSITIVE_OFFSET_PATTERNS = [
        # 匹配 t+1, t+n, ts+1 等模式
        (re.compile(r'\b(t|ts|time|date)\s*\+\s*(\d+)\b', re.IGNORECASE), "Time offset +N"),


        # 匹配 shift(x, -1) 或 shift(x, -n) 表示向前偏移（未来数据）
        (re.compile(r'shift\s*\(\s*[^,]+,\s*-\s*\d+\s*\)', re.IGNORECASE), "Shift with negative offset (future data)"),

        # 匹配 lag(x, -1) 表示向前偏移
        (re.compile(r'lag\s*\(\s*[^,]+,\s*-\s*\d+\s*\)', re.IGNORECASE), "Lag with negative offset (future data)"),

        # 匹配任何函数参数中的负整数（可能是偏移量）
        (re.compile(r'\(\s*[^,]+,\s*-\s*[1-9]\d*\s*\)', re.IGNORECASE), "Function call with negative parameter"),

        # 匹配 lead 函数
        (re.compile(r'\blead\s*\(', re.IGNORECASE), "LEAD function"),

        # 匹配 future 关键字
        (re.compile(r'\bfuture\b', re.IGNORECASE), "FUTURE keyword"),

        # 匹配 _next_ 或 _forward_ 模式
        (re.compile(r'_next_|_forward_|_future_', re.IGNORECASE), "Future suffix pattern"),
    ]

    # 危险模式（需要更严格的检查）
    DANGEROUS_PATTERNS = [
        (re.compile(r'price\s*\(\s*\+\s*1\s*\)', re.IGNORECASE), "Price +1 offset"),
        (re.compile(r'close\s*\(\s*\+\s*\d+\s*\)', re.IGNORECASE), "Close +N offset"),
        (re.compile(r'high\s*\(\s*\+\s*\d+\s*\)', re.IGNORECASE), "High +N offset"),
        (re.compile(r'low\s*\(\s*\+\s*\d+\s*\)', re.IGNORECASE), "Low +N offset"),
    ]

    def __init__(self, config: GatewayConfig, use_ast_parser: bool = False):
        self.config = config
        self.use_ast_parser = use_ast_parser

        self.future_functions = self.DEFAULT_FUTURE_FUNCTIONS.copy()
        if hasattr(config, 'future_blacklist') and config.future_blacklist:
            self.future_functions.update(config.future_blacklist)

        self.regex_patterns = []
        if hasattr(config, 'future_regex_patterns') and config.future_regex_patterns:
            for pattern_info in config.future_regex_patterns:
                if isinstance(pattern_info, dict):
                    pattern = pattern_info.get("pattern", "")
                    desc = pattern_info.get("description", "")
                    if pattern:
                        try:
                            self.regex_patterns.append((re.compile(pattern, re.IGNORECASE), desc))
                        except re.error:
                            pass

        if not self.regex_patterns:
            self.regex_patterns = self.POSITIVE_OFFSET_PATTERNS.copy()

        self.ast_parser = None
        if use_ast_parser:
            try:
                pass
            except ImportError:
                self.use_ast_parser = False

    def scan(self, expr: str) -> Tuple[bool, str]:
        """扫描表达式中的未来函数"""
        if not expr or not expr.strip():
            return False, ""

        violations = []

        # ========== 添加这些专用检测 ==========
        # 检测 shift 负偏移 (直接检测，不依赖正则列表)
        shift_negative = re.search(r'shift\s*\([^)]*,\s*-\s*\d+', expr, re.IGNORECASE)
        if shift_negative:
            violations.append("shift() with negative offset (future data)")

        # 检测 lag 负偏移
        lag_negative = re.search(r'lag\s*\([^)]*,\s*-\s*\d+', expr, re.IGNORECASE)
        if lag_negative:
            violations.append("lag() with negative offset (future data)")

        # 检测 _next_ 等后缀
        if re.search(r'_next_|_forward_|_future_', expr, re.IGNORECASE):
            violations.append("Future suffix pattern (_next_, _forward_)")
        # ====================================

        # 1. 检查未来函数名
        function_violations = self._scan_future_functions(expr)
        violations.extend(function_violations)

        # 2. 检查正则模式
        pattern_violations = self._scan_regex_patterns(expr)
        violations.extend(pattern_violations)

        if violations:
            detail = "; ".join(violations[:5])
            if len(violations) > 5:
                detail += f" (and {len(violations) - 5} more)"
            return True, detail

        return False, ""

    def _scan_future_functions(self, expr: str) -> List[str]:
        """扫描未来函数名"""
        violations = []

        for func in self.future_functions:
            pattern = re.compile(r'\b' + re.escape(func) + r'\s*\(', re.IGNORECASE)
            if pattern.search(expr):
                violations.append(f"Future function: {func}()")

        return violations

    def _scan_regex_patterns(self, expr: str) -> List[str]:
        """使用正则模式扫描"""
        violations = []

        for pattern, description in self.regex_patterns:
            if pattern.search(expr):
                violations.append(f"Pattern matched: {description}")

        return violations

    def scan_strict(self, expr: str) -> FutureScanResult:
        """严格扫描模式"""
        has_future, detail = self.scan(expr)

        dangerous_matches = []
        for pattern, description in self.DANGEROUS_PATTERNS:
            if pattern.search(expr):
                dangerous_matches.append(description)

        severity = FutureFunctionSeverity.CRITICAL
        if dangerous_matches:
            detail = f"{detail}; Dangerous patterns: {', '.join(dangerous_matches)}" if detail else f"Dangerous patterns: {', '.join(dangerous_matches)}"

        return FutureScanResult(
            has_future=has_future,
            violation_detail=detail,
            severity=severity,
            matched_patterns=dangerous_matches
        )

    def is_safe(self, expr: str) -> bool:
        """快速检查表达式是否安全"""
        has_future, _ = self.scan(expr)
        return not has_future

    def get_future_functions_list(self) -> Set[str]:
        """获取未来函数黑名单"""
        return self.future_functions.copy()

    def explain_violation(self, expr: str) -> str:
        """解释违规"""
        has_future, detail = self.scan(expr)

        if not has_future:
            return "No future function detected. Expression is safe."

        lines = [
            "=" * 50,
            "Future Function Violation Report",
            "=" * 50,
            f"Expression: {expr[:100]}{'...' if len(expr) > 100 else ''}",
            "",
            f"Violation: {detail}",
            "",
            "Explanation:",
            "  - Future functions use data that would not be available in real-time trading",
            "  - This creates look-ahead bias and invalidates backtest results",
            "  - Please rewrite the expression using only historical/lagged data",
            "",
            "Common fixes:",
            "  - Replace LEAD(x, n) with LAG(x, n)",
            "  - Replace shift(x, -n) with shift(x, +n) or lag(x, n)",
            "  - Check for t+n references and change to t-n",
            "=" * 50
        ]

        return "\n".join(lines)


def quick_check(expr: str, config: Optional[GatewayConfig] = None) -> bool:
    """快速检查表达式是否安全"""
    if config is None:
        from .config import GatewayConfig
        config = GatewayConfig()
    scanner = FutureFunctionScanner(config)
    return scanner.is_safe(expr)


def explain_future_function(expr: str, config: Optional[GatewayConfig] = None) -> str:
    """解释未来函数问题"""
    if config is None:
        from .config import GatewayConfig
        config = GatewayConfig()
    scanner = FutureFunctionScanner(config)
    return scanner.explain_violation(expr)