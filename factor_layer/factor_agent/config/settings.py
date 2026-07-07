# 全局配置：路径、API、阈值等
from pathlib import Path

# 项目根（factor_agent 目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 第1层：常驻上下文
CLAUDE_MD_PATH = PROJECT_ROOT / "docs" / "CLAUDE.md"

# 评价脚本与工作目录
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
EVAL_SCRIPT_NAME = "配置文件评价脚本.py"
OUTPUT_DIR = PROJECT_ROOT / "output"  # 每版 YAML 与评分结果可落此目录

# Agent 循环
MAX_ROUNDS = 10
SCORE_THRESHOLD = 0.7  # 第6层达标线：结构与表达式需达到基础可用水平

# API（由调用方或环境变量注入）
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8000

# 子 Agent（第5层）：独立上下文、禁止递归
SUBAGENT_MAX_ITERATIONS = 30
SUBAGENT_SYSTEM = "你是子智能体，负责完成父智能体下发的具体任务。不可再生成子智能体，仅可使用当前提供的工具。完成后用自然语言回复结论。"
