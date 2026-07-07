# 第6层：调用评价脚本、返回评分与达标判断
import json
import subprocess
from pathlib import Path

from config.settings import PROJECT_ROOT, SCRIPTS_DIR, EVAL_SCRIPT_NAME, SCORE_THRESHOLD


def run_eval_script(yaml_path: str) -> dict:
    """执行评价脚本，传入 YAML 路径，返回包含 score、passed、detail 等的结果。"""
    script = SCRIPTS_DIR / EVAL_SCRIPT_NAME
    if not script.exists():
        return {"score": 0.0, "passed": False, "detail": "评价脚本不存在", "stdout": "", "stderr": ""}
    try:
        out = subprocess.run(
            ["python", str(script), yaml_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        stdout, stderr = out.stdout or "", out.stderr or ""
        try:
            data = json.loads(stdout.strip() or "{}")
        except json.JSONDecodeError:
            data = {"score": 0.0, "passed": False, "detail": stdout or stderr or "无输出"}
        if "passed" not in data:
            data["passed"] = float(data.get("score", 0)) >= SCORE_THRESHOLD
        data.setdefault("stdout", stdout)
        data.setdefault("stderr", stderr)
        return data
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "passed": False, "detail": "评价脚本超时"}
    except Exception as e:
        return {"score": 0.0, "passed": False, "detail": str(e)}


def is_passed(result: dict) -> bool:
    """根据评价脚本返回判断是否达标。"""
    return bool(result.get("passed", False))
