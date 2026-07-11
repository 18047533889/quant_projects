import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gateway.config import GatewayConfig
from gateway.validator import StaticValidator


def test_validator():
    config = GatewayConfig()
    validator = StaticValidator(config)

    # 测试有效的候选因子
    valid_candidate = {
        "schema_version": "disk.v1",
        "candidate_id": "cand_20260528_a1b2c3d4",
        "Expr": "MA(close, 5)",
        "Config": {"window": 5},
        "BornTimestamp": "2026-05-28T00:00:00Z",
        "BasicInfo": {"type": "equity"}
    }

    is_valid, error, warnings = validator.validate_all(valid_candidate)
    print(f"有效候选: is_valid={is_valid}, error='{error}'")
    assert is_valid is True

    # 测试无效的候选因子（缺少字段）
    invalid_candidate = {
        "schema_version": "disk.v1",
        "candidate_id": "test_001"
    }

    is_valid, error, warnings = validator.validate_all(invalid_candidate)
    print(f"无效候选: is_valid={is_valid}, error='{error}'")
    assert is_valid is False
    assert "Missing required fields" in error

    # 测试不支持的算子
    bad_operator_candidate = {
        "schema_version": "disk.v1",
        "candidate_id": "cand_20260528_a1b2c3d4",
        "Expr": "UNKNOWN_FUNC(close, 5)",
        "Config": {},
        "BornTimestamp": "2026-05-28T00:00:00Z",
        "BasicInfo": {}
    }

    is_valid, error, warnings = validator.validate_all(bad_operator_candidate)
    print(f"无效算子: is_valid={is_valid}, error='{error}'")
    assert is_valid is False
    assert "UNKNOWN_FUNC" in error

    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    test_validator()