# 评价脚本：输入 YAML（或路径），输出评分（JSON 到 stdout，含 score、passed）
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SCORE_THRESHOLD
from config.yaml_schema import validate_yaml_dict


def _score_yaml(data: dict) -> tuple[float, list[str]]:
    """简易可解释评分：结构 70 分 + 表达式复杂度 20 分 + 数据源完整度 10 分。"""
    issues: list[str] = []
    errors = validate_yaml_dict(data)
    if errors:
        issues.extend(errors)
        # 结构错误时只给低分
        score = max(0.0, 0.3 - 0.03 * len(errors))
        return score, issues

    score = 0.7
    expr = str(data.get("factor", {}).get("expr", ""))
    ds = data.get("data_source", {})

    # 表达式复杂度（最多 +0.2）
    func_count = expr.count("(")
    if func_count >= 3:
        score += 0.2
    elif func_count == 2:
        score += 0.15
    elif func_count == 1:
        score += 0.1
    else:
        score += 0.05
        issues.append("表达式较简单，建议增加时序或截面算子组合")

    # 数据源附加信息（最多 +0.1）
    bonus = 0.0
    if ds.get("timestamp_unit") == "ns":
        bonus += 0.05
    if ds.get("start_date") or ds.get("end_date"):
        bonus += 0.05
    score += bonus
    if bonus < 0.1:
        issues.append("建议补充 timestamp_unit/start_date/end_date 以增强可复现性")

    return min(score, 1.0), issues


def main() -> None:
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not yaml_path or not Path(yaml_path).exists():
        out = {"score": 0.0, "passed": False, "detail": "YAML 文件不存在"}
    else:
        try:
            import yaml  # type: ignore
        except ImportError:
            out = {"score": 0.0, "passed": False, "detail": "缺少 pyyaml 依赖"}
            print(json.dumps(out, ensure_ascii=False))
            return

        try:
            data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
            score, issues = _score_yaml(data if isinstance(data, dict) else {})
            passed = score >= SCORE_THRESHOLD
            detail = "通过" if passed else "未通过"
            out = {
                "score": round(score, 4),
                "passed": passed,
                "detail": detail,
                "threshold": SCORE_THRESHOLD,
                "issues": issues,
            }
        except Exception as e:
            out = {"score": 0.0, "passed": False, "detail": f"评估异常: {e}"}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
