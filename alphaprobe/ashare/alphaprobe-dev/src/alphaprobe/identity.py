"""FactorIdentity 工厂（任务书 §75.1 / Phase 4）：身份构造统一接 dedup。

trainer/pool/exporter 替换字符串 identity 的统一入口：
``from_formula(formula, orientation) -> contracts.FactorIdentity``。

identity 规则（与 dedup 层严格一致）：
- ``factor_id``：canonical_ast_hash 前缀（稳定、可追溯）；
- ``canonical_formula``：dedup.canonicalize_dsl 文本化简；
- ``canonical_ast_hash``：sha256(canonical_formula)；
- ``signal_equivalence_id``：sign-invariant（f / -f / 0-f / (-1)*f 同 id）；
- ``parameter_family_id``：窗口参数打码（ts_mean(x,19/20/21) 同族）；
- ``orientation``：Train-only（§10），默认 +1。

不修改 factor_engine/、data_access/；纯 AlphaPROBE 侧组合 dedup + contracts。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from alphaprobe.contracts import FactorIdentity
from alphaprobe.dedup import (
    canonical_ast_hash,
    canonicalize_dsl,
    parameter_family_key,
    signal_equivalence_id,
)

if TYPE_CHECKING:
    from alphaprobe.integration.factor_engine_adapter import CanonicalFactor

# canonical_ast_hash 前缀长度（factor_id 短标识）
FACTOR_ID_PREFIX = 12


def factor_id_from_hash(canonical_hash: str) -> str:
    """factor_id = canonical_ast_hash 前 12 位（稳定且跨轮一致）。"""
    return str(canonical_hash)[:FACTOR_ID_PREFIX]


class FactorIdentityFactory:
    """§75.1 统一构造器：把 contracts.FactorIdentity 的构造接 dedup。"""

    def __init__(self, adapter: Any | None = None) -> None:
        self._adapter = adapter

    def from_formula(
        self,
        formula: str,
        orientation: int = 1,
        *,
        factor_id: str | None = None,
        canonical: "CanonicalFactor | None" = None,
    ) -> FactorIdentity:
        """从 DSL 公式构造 FactorIdentity。

        Parameters
        ----------
        formula : str
            factor_engine DSL 公式文本。
        orientation : int
            Train-only 方向（§10）：+1 或 -1；默认 +1。
        factor_id : str, optional
            覆盖 factor_id（默认 canonical_ast_hash[:12]）。
        canonical : CanonicalFactor, optional
            已 canonicalize 的结果（避免重复解析）。
        """
        text = str(formula or "").strip()
        if not text:
            raise ValueError("empty formula")

        if canonical is not None:
            canonical_formula = canonical.canonical_formula or canonicalize_dsl(text)
            ast_hash = canonical.canonical_ast_hash or canonical_ast_hash(canonical_formula)
            signal_id = canonical.signal_equivalence_id or signal_equivalence_id(canonical_formula)
            family_id = canonical.parameter_family_id or parameter_family_key(canonical_formula)
        else:
            canonical_formula = canonicalize_dsl(text)
            ast_hash = canonical_ast_hash(canonical_formula)
            signal_id = signal_equivalence_id(canonical_formula)
            family_id = parameter_family_key(canonical_formula)

        return FactorIdentity(
            factor_id=factor_id or factor_id_from_hash(ast_hash),
            canonical_formula=canonical_formula,
            canonical_ast_hash=ast_hash,
            signal_equivalence_id=signal_id,
            parameter_family_id=family_id,
            orientation=int(orientation),
        )

    def from_canonical(
        self,
        canonical: "CanonicalFactor",
        orientation: int = 1,
        *,
        factor_id: str | None = None,
    ) -> FactorIdentity:
        """从 adapter.canonicalize 产物构造 identity（避免二次解析）。"""
        return self.from_formula(
            canonical.formula,
            orientation=orientation,
            factor_id=factor_id,
            canonical=canonical,
        )

    def from_dedup(self, formula: str, orientation: int = 1) -> FactorIdentity:
        """纯 dedup 路径（不依赖 FE；测试/降级用）。"""
        return self.from_formula(formula, orientation=orientation)


# 模块级便捷函数（trainer/pool/exporter 直接调用）
def from_formula(
    formula: str,
    orientation: int = 1,
    *,
    factor_id: str | None = None,
) -> FactorIdentity:
    return FactorIdentityFactory().from_formula(formula, orientation=orientation, factor_id=factor_id)


__all__ = [
    "FACTOR_ID_PREFIX",
    "FactorIdentityFactory",
    "factor_id_from_hash",
    "from_formula",
]
