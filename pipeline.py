"""兼容 shim：``runtime.pipeline.batch`` 的完整别名。"""
import sys

import factor_engine.runtime.pipeline.batch as _batch

sys.modules[__name__] = _batch
