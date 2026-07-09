"""兼容 shim：``data_access.clickhouse.panel`` 的完整别名。"""
import sys

import data_access.clickhouse.panel as _mod

sys.modules[__name__] = _mod
