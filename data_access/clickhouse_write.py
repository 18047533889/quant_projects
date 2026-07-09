"""兼容 shim：``data_access.clickhouse.write`` 的完整别名。"""
import sys

import data_access.clickhouse.write as _mod

sys.modules[__name__] = _mod
