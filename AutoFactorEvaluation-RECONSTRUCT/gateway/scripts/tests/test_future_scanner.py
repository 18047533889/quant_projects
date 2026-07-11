import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gateway.config import GatewayConfig
from gateway.future_scanner import FutureFunctionScanner, quick_check, explain_future_function


def test_future_scanner():
    config = GatewayConfig()
    scanner = FutureFunctionScanner(config)

    print("=" * 50)
    print("测试未来函数扫描器")
    print("=" * 50)

    # 测试安全表达式
    safe_expressions = [
        "MA(close, 5)",
        "STD(close, 20)",
        "CORR(return_1d, return_2d)",
        "IF(close > open, close, open)",
        "RANK(volume)",
    ]

    for expr in safe_expressions:
        has_future, detail = scanner.scan(expr)
        assert has_future is False, f"应该安全: {expr}"
        print(f"✅ 安全: {expr[:40]}...")

    # 测试不安全表达式
    unsafe_expressions = [
        ("REFX(close, 1)", "REFX"),
        ("t+1", "t+1"),
        ("LEAD(close, 5)", "LEAD"),
        ("shift(close, -1)", "shift负偏移"),
        ("price(+1)", "price+1"),
        ("FUTURE_VALUE(close)", "FUTURE_VALUE"),
        ("close _next_", "_next_模式"),
    ]

    for expr, expected in unsafe_expressions:
        has_future, detail = scanner.scan(expr)
        assert has_future is True, f"应该检测到未来函数: {expr}"
        print(f"✅ 检测到: {expr[:30]}... -> {detail[:50]}...")

    # 测试 explain_violation
    print("\n" + "=" * 50)
    print("违规说明示例:")
    print("=" * 50)
    explanation = scanner.explain_violation("REFX(close, 1)")
    print(explanation[:500])

    print("\n🎉 所有测试通过！")


def test_quick_check():
    print("\n" + "=" * 50)
    print("测试 quick_check 便捷函数")
    print("=" * 50)

    assert quick_check("MA(close, 5)") is True
    assert quick_check("REFX(close, 1)") is False

    print("✅ quick_check 测试通过")


if __name__ == "__main__":
    test_future_scanner()
    test_quick_check()