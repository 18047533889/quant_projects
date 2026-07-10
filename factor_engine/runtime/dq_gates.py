"""兼容 shim：将 ``runtime.dq_gates`` 重定向至 ``runtime.quality.dq_gates``。

历史代码可能 ``from runtime.dq_gates import ...``；本模块保持向后兼容，
实际实现位于 ``runtime/quality/dq_gates.py``（因子产出轻量 DQ 门禁）。
"""
import sys
import runtime.quality.dq_gates as _mod
sys.modules[__name__] = _mod
