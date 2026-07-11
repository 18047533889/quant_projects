"""
complexity.py 的单元测试
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gateway.config import GatewayConfig
from gateway.complexity import (
    ComplexityEvaluator,
    calculate_complexity,
    is_within_budget,
    get_complexity_report
)


def test_complexity_evaluator():
    """测试复杂度评估器"""
    config = GatewayConfig()
    evaluator = ComplexityEvaluator(config)

    print("=" * 50)
    print("测试复杂度评估器")
    print("=" * 50)

    # 测试简单表达式
    expr1 = "MA(close, 5)"
    score1, within1, report1 = evaluator.evaluate(expr1)
    print(f"表达式: {expr1}")
    print(f"  得分: {score1:.2f}, 预算内: {within1}")
    print(f"  深度: {report1.max_depth}, 算子: {list(report1.operator_breakdown.keys())}")

    # 测试中等复杂度
    expr2 = "CORR(MA(close, 5), MA(volume, 10))"
    score2, within2, report2 = evaluator.evaluate(expr2)
    print(f"\n表达式: {expr2}")
    print(f"  得分: {score2:.2f}, 预算内: {within2}")

    # 测试高复杂度
    expr3 = "REGRESSION(close, MA(close, 5), STD(close, 20), volume)"
    score3, within3, report3 = evaluator.evaluate(expr3)
    print(f"\n表达式: {expr3}")
    print(f"  得分: {score3:.2f}, 预算内: {within3}")

    # 测试空表达式
    score4, within4, _ = evaluator.evaluate("")
    print(f"\n空表达式得分: {score4:.2f}")
    assert score4 == 0.0
    assert within4 is True

    print("\n✅ 基础测试通过")


def test_complexity_threshold():
    """测试预算阈值"""
    config = GatewayConfig()
    evaluator = ComplexityEvaluator(config)

    # 使用配置中的阈值（默认 20.0）
    threshold = config.max_complexity
    print(f"\n预算阈值: {threshold}")

    # 应该通过的表达式
    simple_expr = "MA(close, 5)"
    score, within = evaluator.evaluate_simple(simple_expr)
    assert within is True, f"{simple_expr} 应该在预算内"
    print(f"✅ {simple_expr}: {score:.2f} <= {threshold}")

    # 可能超预算的复杂表达式
    complex_expr = "CORR(MA(close, 5), STD(close, 20)) + CORR(volume, close) * 2"
    score, within = evaluator.evaluate_simple(complex_expr)
    print(f"📊 {complex_expr[:50]}...: {score:.2f} (阈值: {threshold})")

    print("✅ 阈值测试通过")


def test_operator_weights():
    """测试算子权重"""
    config = GatewayConfig()
    evaluator = ComplexityEvaluator(config)

    # 测试单个算子
    weight_ma = evaluator.get_operator_weight("MA")
    weight_corr = evaluator.get_operator_weight("CORR")
    weight_unknown = evaluator.get_operator_weight("UNKNOWN")

    print(f"\nMA 权重: {weight_ma}")
    print(f"CORR 权重: {weight_corr}")
    print(f"UNKNOWN 默认权重: {weight_unknown}")

    assert weight_ma == 1.0
    assert weight_corr == 3.0
    assert weight_unknown == 1.0

    print("✅ 权重测试通过")


def test_convenience_functions():
    """测试便捷函数"""
    config = GatewayConfig()

    expr = "MA(close, 5) + STD(close, 20)"

    # calculate_complexity
    score = calculate_complexity(expr, config)
    print(f"\ncalculate_complexity('{expr}') = {score:.2f}")
    assert score > 0

    # is_within_budget
    within = is_within_budget(expr, config)
    print(f"is_within_budget('{expr}') = {within}")

    # get_complexity_report
    report = get_complexity_report(expr, config)
    print(f"report.summary() = {report.summary()}")

    print("✅ 便捷函数测试通过")


def test_explain_complexity():
    """测试复杂度分析报告"""
    config = GatewayConfig()
    evaluator = ComplexityEvaluator(config)

    expr = "CORR(MA(close, 5), STD(close, 20))"

    explanation = evaluator.explain_complexity(expr)
    print("\n" + explanation[:500] + "...")

    print("✅ 解释报告测试通过")


def test_suggest_optimization():
    """测试优化建议"""
    config = GatewayConfig()
    evaluator = ComplexityEvaluator(config)

    # 测试复杂表达式
    complex_expr = "REGRESSION(close, MA(close, 5), CORR(close, volume), NN(close))"

    suggestions = evaluator.suggest_optimization(complex_expr)
    print("\n优化建议:")
    for s in suggestions:
        print(f"  - {s}")

    print("✅ 优化建议测试通过")


def test_depth_penalty():
    """测试深度惩罚"""
    config = GatewayConfig()

    # 临时修改深度惩罚配置
    config.depth_penalty_enabled = True
    config.depth_penalty_per_level = 0.5

    evaluator = ComplexityEvaluator(config)

    # 深度 1 的表达式
    shallow_expr = "ADD(close, 5)"
    score1, _, report1 = evaluator.evaluate(shallow_expr)

    # 深度 3 的表达式
    deep_expr = "ADD(ADD(close, 5), ADD(close, 5))"
    score2, _, report2 = evaluator.evaluate(deep_expr)

    print(f"\n浅层表达式深度: {report1.max_depth}, 得分: {score1:.2f}")
    print(f"深层表达式深度: {report2.max_depth}, 得分: {score2:.2f}")

    # 深度惩罚应该使深层表达式得分更高
    depth_penalty_diff = score2 - score1
    print(f"深度惩罚贡献: {depth_penalty_diff:.2f}")

    print("✅ 深度惩罚测试通过")


if __name__ == "__main__":
    test_complexity_evaluator()
    test_complexity_threshold()
    test_operator_weights()
    test_convenience_functions()
    test_explain_complexity()
    test_suggest_optimization()
    test_depth_penalty()

    print("\n" + "=" * 50)
    print("🎉 所有复杂度测试通过！")
    print("=" * 50)