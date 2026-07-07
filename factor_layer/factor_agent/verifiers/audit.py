# 版本与评分留档、可回滚、可审计
from pathlib import Path
from typing import Any

from config.settings import OUTPUT_DIR, PROJECT_ROOT


def log_final(yaml_path: str | None, result: Any, passed: bool) -> None:
    """达标或达到最大轮次时写审计记录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "audit_log.txt"
    line = f"yaml_path={yaml_path} passed={passed} result={result}\n"
    log_file.write_text(log_file.read_text(encoding="utf-8") + line, encoding="utf-8")


def log_final_json(yaml_path: str | None, result: dict, passed: bool) -> None:
    """同上，以 JSON 行追加。"""
    import json
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "audit_log.jsonl"
    rec = {"yaml_path": yaml_path, "passed": passed, "result": result}
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
