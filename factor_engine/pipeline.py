"""兼容 shim：``runtime.pipeline.batch`` 的完整别名。"""
import sys

import runtime.pipeline.batch as _batch

sys.modules[__name__] = _batch
