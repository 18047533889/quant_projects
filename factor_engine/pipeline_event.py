"""兼容 shim：``runtime.pipeline.event`` 的完整别名。"""
import sys

import factor_engine.runtime.pipeline.event as _event

sys.modules[__name__] = _event
