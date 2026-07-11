"""
复杂度评估器模块

负责:
- 计算因子表达式的复杂度得分
- 基于算子权重累加 + 嵌套深度惩罚
- 判断是否超过预算阈值
- 提供详细的复杂度分析报告

支持两种模式:
- 正则匹配模式（快速，无需 AST）
- AST 解析模式（精确，需要 factor_engine）
"""

import re
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field

from .config import GatewayConfig


@dataclass
class ComplexityReport:
    """复杂度评估报告"""
    total_score: float  # 总复杂度得分
    operator_weight_sum: float  # 算子权重和
    depth_penalty: float  # 深度惩罚
    max_depth: int  # 最大嵌套深度
    operator_breakdown: Dict[str, float]  # 各算子权重明细
    is_within_budget: bool  # 是否在预算内
    threshold: float  # 预算阈值

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_score": self.total_score,
            "operator_weight_sum": self.operator_weight_sum,
            "depth_penalty": self.depth_penalty,
            "max_depth": self.max_depth,
            "operator_breakdown": self.operator_breakdown,
            "is_within_budget": self.is_within_budget,
            "threshold": self.threshold,
        }

    def summary(self) -> str:
        """返回摘要字符串"""
        status = "✅ PASS" if self.is_within_budget else "❌ FAIL"
        return f"Complexity: {self.total_score:.2f} / {self.threshold} ({status})"


class ComplexityEvaluator:
    """
    复杂度评估器

    基于算子权重和嵌套深度计算因子表达式的计算复杂度
    """

    # 默认权重（当配置中没有时使用）
    DEFAULT_WEIGHTS = {
        # 基础运算
        "ADD": 0.5, "SUB": 0.5, "MUL": 0.8, "DIV": 0.8,
        "POWER": 1.2, "SQRT": 1.0, "LOG": 1.0, "EXP": 1.0,
        "NEG": 0.3, "ABS": 0.5, "SIGN": 0.5,

        # 比较运算
        "GT": 0.3, "LT": 0.3, "GE": 0.3, "LE": 0.3, "EQ": 0.3, "NE": 0.3,
        "AND": 0.3, "OR": 0.3, "NOT": 0.2,

        # 聚合函数
        "SUM": 1.0, "MEAN": 1.0, "MEDIAN": 1.5, "MODE": 1.5,
        "MAX": 0.8, "MIN": 0.8, "STD": 2.0, "VAR": 2.0,

        # 时序函数
        "MA": 1.0, "SMA": 1.2, "EMA": 1.5, "WMA": 1.3,
        "CORR": 3.0, "COVAR": 2.5, "RANK": 1.5,
        "DELTA": 0.5, "CHANGE": 0.5, "LAG": 0.5,

        # 条件函数
        "IF": 0.5, "IIF": 0.5, "CASE": 1.0,

        # 复杂函数
        "REGRESSION": 5.0, "NN": 10.0, "GBDT": 8.0,
        "PCA": 4.0, "ICA": 4.0,
    }

    # 算子匹配正则（匹配函数调用模式）
    OPERATOR_PATTERN = re.compile(r'\b([A-Z][A-Z0-9_]*)\s*\(', re.IGNORECASE)

    # 括号嵌套深度检测
    BRACKET_DEPTH_PATTERN = re.compile(r'[()]')

    def __init__(self, config: GatewayConfig, use_ast_parser: bool = False):
        """
        初始化复杂度评估器

        Args:
            config: 网关配置对象
            use_ast_parser: 是否使用 AST 解析器（更精确）
        """
        self.config = config
        self.use_ast_parser = use_ast_parser
        self.threshold = config.max_complexity

        # 加载权重配置
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if hasattr(config, 'complexity_weights') and config.complexity_weights:
            self.weights.update(config.complexity_weights)

        # 深度惩罚配置
        self.depth_penalty_enabled = getattr(config, 'depth_penalty_enabled', True)
        self.depth_penalty_per_level = getattr(config, 'depth_penalty_per_level', 0.5)

        # AST 解析器（可选）
        self.ast_parser = None
        if use_ast_parser:
            try:
                # 尝试导入 factor_engine 的解析器
                # from factor_engine.parser import parse_expression
                # self.ast_parser = parse_expression
                pass
            except ImportError:
                self.use_ast_parser = False

    def evaluate(self, expr: str) -> Tuple[float, bool, ComplexityReport]:
        """
        评估表达式复杂度

        Args:
            expr: 因子表达式字符串

        Returns:
            (complexity_score, is_within_budget, report)
        """
        if not expr or not expr.strip():
            return 0.0, True, ComplexityReport(
                total_score=0.0,
                operator_weight_sum=0.0,
                depth_penalty=0.0,
                max_depth=0,
                operator_breakdown={},
                is_within_budget=True,
                threshold=self.threshold
            )

        # 1. 提取所有算子并计算权重和
        operator_breakdown = self._calculate_operator_weights(expr)
        operator_weight_sum = sum(operator_breakdown.values())

        # 2. 计算最大嵌套深度
        max_depth = self._calculate_max_depth(expr)

        # 3. 计算深度惩罚
        depth_penalty = self._calculate_depth_penalty(max_depth) if self.depth_penalty_enabled else 0.0

        # 4. 计算总分
        total_score = operator_weight_sum + depth_penalty

        # 5. 判断是否在预算内
        is_within_budget = total_score <= self.threshold

        # 6. 生成报告
        report = ComplexityReport(
            total_score=total_score,
            operator_weight_sum=operator_weight_sum,
            depth_penalty=depth_penalty,
            max_depth=max_depth,
            operator_breakdown=operator_breakdown,
            is_within_budget=is_within_budget,
            threshold=self.threshold
        )

        return total_score, is_within_budget, report

    def _calculate_operator_weights(self, expr: str) -> Dict[str, float]:
        """
        计算表达式中所有算子的权重

        Args:
            expr: 表达式字符串

        Returns:
            算子 -> 权重 的字典
        """
        operator_weights = {}

        # 提取所有算子
        matches = self.OPERATOR_PATTERN.findall(expr)

        for op in matches:
            op_upper = op.upper()
            weight = self.weights.get(op_upper, 1.0)  # 默认权重 1.0
            operator_weights[op_upper] = operator_weights.get(op_upper, 0.0) + weight

        return operator_weights

    def _calculate_max_depth(self, expr: str) -> int:
        """
        计算表达式的最大嵌套深度

        通过括号匹配计算嵌套层级

        Args:
            expr: 表达式字符串

        Returns:
            最大嵌套深度
        """
        max_depth = 0
        current_depth = 0

        for char in expr:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth = max(0, current_depth - 1)

        return max_depth

    def _calculate_depth_penalty(self, depth: int) -> float:
        """
        计算深度惩罚

        Args:
            depth: 最大嵌套深度

        Returns:
            深度惩罚值
        """
        if depth <= 1:
            return 0.0
        # 深度惩罚 = (深度 - 1) * 每层惩罚
        return (depth - 1) * self.depth_penalty_per_level

    def evaluate_simple(self, expr: str) -> Tuple[float, bool]:
        """
        简化版评估（仅返回分数和是否通过）

        Args:
            expr: 表达式字符串

        Returns:
            (complexity_score, is_within_budget)
        """
        score, within, _ = self.evaluate(expr)
        return score, within

    def get_operator_weight(self, operator: str) -> float:
        """
        获取指定算子的权重

        Args:
            operator: 算子名称

        Returns:
            权重值
        """
        return self.weights.get(operator.upper(), 1.0)

    def explain_complexity(self, expr: str) -> str:
        """
        生成复杂度分析的详细说明

        Args:
            expr: 表达式字符串

        Returns:
            可读的分析报告
        """
        score, within, report = self.evaluate(expr)

        lines = [
            "=" * 50,
            "Complexity Analysis Report",
            "=" * 50,
            f"Expression: {expr[:80]}{'...' if len(expr) > 80 else ''}",
            "",
            f"Total Score: {score:.2f}",
            f"Threshold:   {self.threshold:.2f}",
            f"Status:      {'✅ WITHIN BUDGET' if within else '❌ BUDGET EXCEEDED'}",
            "",
            f"Operator Weight Sum: {report.operator_weight_sum:.2f}",
            f"Max Nesting Depth:   {report.max_depth}",
            f"Depth Penalty:       {report.depth_penalty:.2f}",
            "",
            "Operator Breakdown:",
        ]

        for op, weight in sorted(report.operator_breakdown.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  - {op}: {weight:.2f}")

        if len(report.operator_breakdown) > 10:
            lines.append(f"  ... and {len(report.operator_breakdown) - 10} more")

        lines.append("=" * 50)

        return "\n".join(lines)

    def get_complexity_level(self, score: float) -> str:
        """
        根据分数返回复杂度等级

        Args:
            score: 复杂度得分

        Returns:
            等级描述
        """
        if score < 5:
            return "Low"
        elif score < 15:
            return "Medium"
        elif score < 30:
            return "High"
        else:
            return "Extreme"

    def suggest_optimization(self, expr: str) -> List[str]:
        """
        提供复杂度优化建议

        Args:
            expr: 表达式字符串

        Returns:
            优化建议列表
        """
        suggestions = []
        score, within, report = self.evaluate(expr)

        if within:
            suggestions.append("Expression complexity is within budget.")
            return suggestions

        # 找出最耗时的算子
        expensive_ops = sorted(
            report.operator_breakdown.items(),
            key=lambda x: -x[1]
        )[:3]

        if expensive_ops:
            suggestions.append(f"Most expensive operators: {', '.join([f'{op}({w:.1f})' for op, w in expensive_ops])}")

        # 嵌套深度建议
        if report.max_depth > 5:
            suggestions.append(
                f"High nesting depth ({report.max_depth}). Consider breaking down into intermediate variables.")

        # 算子替换建议
        for op, weight in expensive_ops:
            if op in ["CORR", "COVAR"] and weight > 2.5:
                suggestions.append(f"Consider replacing {op} with simpler correlation approximation if possible.")
            elif op in ["REGRESSION", "NN", "GBDT"]:
                suggestions.append(f"Consider using pre-computed features instead of computing {op} online.")

        if not suggestions:
            suggestions.append("Consider simplifying the expression by reducing the number of operators.")

        return suggestions


# 便捷函数
def calculate_complexity(expr: str, config: Optional[GatewayConfig] = None) -> float:
    """
    快速计算表达式复杂度

    Args:
        expr: 因子表达式
        config: 配置对象（可选）

    Returns:
        复杂度得分
    """
    if config is None:
        from .config import GatewayConfig
        config = GatewayConfig()

    evaluator = ComplexityEvaluator(config)
    score, _ = evaluator.evaluate_simple(expr)
    return score


def is_within_budget(expr: str, config: Optional[GatewayConfig] = None) -> bool:
    """
    检查表达式是否在预算内

    Args:
        expr: 因子表达式
        config: 配置对象（可选）

    Returns:
        True 表示在预算内
    """
    if config is None:
        from .config import GatewayConfig
        config = GatewayConfig()

    evaluator = ComplexityEvaluator(config)
    _, within = evaluator.evaluate_simple(expr)
    return within


def get_complexity_report(expr: str, config: Optional[GatewayConfig] = None) -> ComplexityReport:
    """
    获取完整的复杂度报告

    Args:
        expr: 因子表达式
        config: 配置对象（可选）

    Returns:
        ComplexityReport 对象
    """
    if config is None:
        from .config import GatewayConfig
        config = GatewayConfig()

    evaluator = ComplexityEvaluator(config)
    _, _, report = evaluator.evaluate(expr)
    return report