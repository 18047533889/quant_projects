# test_io.py
from gateway.io_utils import generate_run_id, generate_report_md
from gateway.models import GatewayResult, GatewayLabel
from datetime import datetime, timezone

run_id = generate_run_id()
print(f"✅ 生成 run_id: {run_id}")

# 创建一个模拟的 GatewayResult 对象
mock_result = GatewayResult(
    label=GatewayLabel.PASS,
    reason="",
    run_id=run_id,
    checked_at=datetime.now(timezone.utc).isoformat(),
    is_legal=True,
    has_future=False,
    complexity_score=15.5,
    is_within_budget=True,
    is_duplicate=False,
    historical_report_id=None,
    error_message=None
)

# 测试 report 生成
test_md = generate_report_md(
    candidate_id="test_001",
    result=mock_result,  # 传入真实对象，不是 None
    candidate_dict={"Expr": "MA(close, 5)", "Config": {}}
)
print("✅ generate_report_md 可用")
print("\n生成的报告预览:")
print(test_md[:500])