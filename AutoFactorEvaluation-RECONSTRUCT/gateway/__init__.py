"""
Gateway（极速网关）— 因子评估流水线第一道关口。

重构版本 — 7 步审查流水线:
  Step 1: 解析 manifest.json
  Step 2: YAML 配置化（tiny_run=True）
  Step 3: 去重检测（缓存命中）
  Step 4: 因子引擎小体量试运行
  Step 5: 未来函数拦截（DeepSeek）
  Step 6: 复杂度评分
  Step 7: 数据质量检测

使用示例:
    from gateway.scripts.initialization import init_gateway
    from gateway.scripts.gateway_core import run_gateway, write_afv_json

    init = init_gateway()
    output = run_gateway("manifest.json", output_path="/tmp/gw_out", cache_root="/tmp/gw_cache")
    afv_path = write_afv_json(output, "/tmp/gw_out")
"""

from .scripts.gateway_core import run_gateway, StepResult
from utils.deepseek_client import deepseek_chat, get_api_key
