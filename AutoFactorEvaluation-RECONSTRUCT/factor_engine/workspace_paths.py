"""兼容 shim：``util.workspace_paths`` 的完整别名。"""
import sys

import util.workspace_paths as _mod

sys.modules[__name__] = _mod
