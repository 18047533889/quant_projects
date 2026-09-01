# -*- coding: utf-8 -*-
"""无状态公式身份层（stateless factor identity layer）。

``factor_engine.identity`` 是一个**只读、无副作用**的公式身份计算包：

- 不对现有算子 / backend / planner 做任何修改（仅新增本包文件）；
- 输入是 DSL 公式文本，输出是 :class:`FactorIdentity`（冻结 dataclass）；
- 身份基于 AST（``factor_engine.api.dsl_parser.parse_expr``）而非原始字符串，
  因此 ``close`` 与 ``AdjClose``、``ts_mean`` 与 ``TS_MEAN`` 等跨 Miner
  写法天然收敛到同一身份；
- 稳定序列化基于 ``factor_engine.expr.canonical``（``canonical_expression``），
  不另造平行体系；
- 哈希一律 SHA-256 + 命名空间 ``factor_identity_v1``（禁 ``hash()`` / BLAKE3）。

公共入口：:

    from factor_engine.identity import get_factor_identity

    ident = get_factor_identity("ts_mean(close, 20)")
    ident.signal_equivalence_id   # 同信号（含全局取负）公式共享
    ident.orientation             # +1 / -1
"""

from __future__ import annotations

from .models import FactorIdentity, FactorIdentityError, IdentityLookupResult
from .api import get_factor_identity

__all__ = [
    "FactorIdentity",
    "FactorIdentityError",
    "IdentityLookupResult",
    "get_factor_identity",
]

__version__ = "0.1.0"
