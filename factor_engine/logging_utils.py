"""兼容 shim：``util.logging_utils`` 的完整别名。"""
import sys

import util.logging_utils as _mod

sys.modules[__name__] = _mod
