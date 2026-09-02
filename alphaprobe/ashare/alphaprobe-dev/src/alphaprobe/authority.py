"""权威接线层（V2-G §28 / §30 / §36）：FE 唯一 formula 权威 + modeling 契约。

本模块是 AlphaPROBE 挖掘链上「谁说了算」的单一接线点：

- :func:`ensure_authority_available`：把 ``quant_projects`` 引导进 ``sys.path``
  （``fe_bridge.paths.ensure_factor_engine_importable``），使 ``DedupClient``
  的 FE-first 降级链第一条（``factor_engine.identity.get_factor_identity``）
  真正可达。FE 不可用时**不吞异常**（fail-closed），由调用方决定降级。
- :func:`build_identity_view`：FE 权威身份（canonical_formula / canonical_ast_hash /
  signal_equivalence_id / parameter_family_id）。正则文本化简不再出现在活跃
  主链上——全部委托 FE AST（``factor_engine.identity``）。
- :func:`vwap_20d_label_contract` / :func:`validate_label_contract_20d`：
  把 modeling 库的 ``LabelContract`` / ``EmbargoSpec`` / ``validate_overlap``
  接入挖掘/评估链，声明 vwap→vwap 20 日 label 的 horizon=20、embargo≥20。
- :func:`assert_authority_available`：CI / runner 冒烟用，FE 缺失即 raise。

不修改 factor_engine/、data_access/、modeling/；只在 AlphaPROBE 侧接线。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FactorIdentityAuthorityError",
    "ensure_authority_available",
    "assert_authority_available",
    "build_identity_view",
    "vwap_20d_label_contract",
    "validate_label_contract_20d",
]


class FactorIdentityAuthorityError(RuntimeError):
    """FE 身份权威不可用（fail-closed，调用方据此判断降级级别）。"""


# ---------------------------------------------------------------------------
# FE 引导（幂等；DedupClient 内部裸 import factor_engine.identity 前先引导）
# ---------------------------------------------------------------------------


def ensure_authority_available() -> None:
    """把 quant_projects 引导进 sys.path，使 factor_engine.identity 可达。

    fail-closed：FE 不可用抛 :class:`FactorIdentityAuthorityError`，
    绝不 ``except Exception: pass``。调用方（runner / pipeline / continuous）
    捕获后明确选择降级语义。
    """
    from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable

    try:
        ensure_factor_engine_importable()
        from factor_engine.identity import get_factor_identity as _  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - fail-closed，统一包装
        raise FactorIdentityAuthorityError(
            f"factor_engine.identity unavailable: {type(exc).__name__}: {exc}"
        ) from exc


def assert_authority_available() -> None:
    """CI / runner 冒烟入口：FE 身份权威缺失即 raise（fail-closed）。"""
    ensure_authority_available()


# ---------------------------------------------------------------------------
# 权威身份（FE AST 优先；正则文本化简不再出现在活跃主链）
# ---------------------------------------------------------------------------


def build_identity_view(
    formula: str,
    provider: Any | None = None,
) -> dict[str, Any]:
    """构造 FE 权威身份视图（canonical_formula / canonical_ast_hash /
    signal_equivalence_id / parameter_family_id / orientation）。

    Parameters
    ----------
    formula : str
        factor_engine DSL 公式文本。
    provider : optional
        身份提供者（默认 ``factor_engine.identity.get_factor_identity``）。
        注入用（测试 / 降级策略显式替换）。

    Raises
    ------
    FactorIdentityAuthorityError
        FE 身份不可用 / 公式无法解析（fail-closed，不吞）。
    """
    text = str(formula or "").strip()
    if not text:
        raise FactorIdentityAuthorityError("empty formula for identity")

    ensure_authority_available()
    if provider is None:
        from factor_engine.identity import get_factor_identity

        run = get_factor_identity
    elif callable(provider):
        run = provider
    elif hasattr(provider, "get_factor_identity"):
        run = provider.get_factor_identity
    else:
        raise FactorIdentityAuthorityError(f"invalid identity provider: {provider!r}")

    result = run(text)
    if result is None:
        raise FactorIdentityAuthorityError(f"FE identity returned None for {text!r}")

    # FE 返回值：冻结 dataclass（有属性）或 dict（兼容）。
    if isinstance(result, dict):
        canonical_dsl = result.get("canonical_dsl") or result.get("canonical_formula") or text
        ast_hash = result.get("canonical_ast_hash") or result.get("ast_hash") or ""
        signal_id = result.get("signal_equivalence_id") or result.get("signal_id") or ""
        family_id = result.get("parameter_family_id")
        orientation = result.get("orientation", 1)
    else:
        canonical_dsl = str(
            getattr(result, "canonical_dsl", "") or getattr(result, "canonical_formula", "") or text
        )
        ast_hash = str(getattr(result, "canonical_ast_hash", "") or getattr(result, "ast_hash", "") or "")
        signal_id = str(
            getattr(result, "signal_equivalence_id", "") or getattr(result, "signal_id", "") or ""
        )
        family_id = getattr(result, "parameter_family_id", None)
        orientation = int(getattr(result, "orientation", 1) or 1)

    if not signal_id or not ast_hash:
        raise FactorIdentityAuthorityError(
            f"FE identity incomplete for {text!r} (signal={bool(signal_id)}, ast={bool(ast_hash)})"
        )

    return {
        "formula": text,
        "canonical_formula": canonical_dsl,
        "canonical_ast_hash": ast_hash,
        "signal_equivalence_id": signal_id,
        "parameter_family_id": str(family_id) if family_id else None,
        "orientation": orientation,
    }


# ---------------------------------------------------------------------------
# Modeling 契约：vwap→vwap 20 日 label（§36-§38）
# ---------------------------------------------------------------------------


def vwap_20d_label_contract() -> Any:
    """返回 modeling 的 vwap→vwap 20 日 ``LabelContract``（authoritative）。

    ``horizon_bars=20``、``overlapping=True``（日频每日锚定 → label 区间重叠）、
    ``return_basis='vwap_to_vwap'``、``embargo_bars=20``。embargo≥horizon 保证
    边界前 label 已成熟（modeling ``EmbargoSpec.validate_against_label`` 校验）。
    """
    from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable

    ensure_factor_engine_importable()
    from modeling.contracts import LabelContract

    return LabelContract(
        label_name="vwap_to_vwap_20d",
        origin_time="t",
        availability_time_rule="label matured at t+20 close",
        horizon_bars=20,
        overlapping=True,  # 日频每日锚定 → derive_overlap(stride=1)=True
        return_basis="vwap_to_vwap",
        entry_price_basis="VWAP_t",
        exit_price_basis="VWAP_{t+20}",
        embargo_bars=20,
    )


def validate_label_contract_20d() -> None:
    """校验 vwap→vwap 20 日 label 契约（fail-closed：不满足即 raise）。

    - ``modeling.contracts.validate_overlap``：overlap 标志须与 horizon/stride 自洽；
    - ``EmbargoSpec(20).validate_against_label``：embargo ≥ horizon（≥20）。
    """
    from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable

    ensure_factor_engine_importable()
    from modeling.contracts import EmbargoSpec, validate_overlap

    contract = vwap_20d_label_contract()
    # 日频单步（stride=1）：horizon=20 > 1 → overlapping 必须为 True。
    validate_overlap(contract, stride_bars=1)
    embargo_spec = EmbargoSpec(
        days=20,
        rationale="vwap 20d label: embargo >= horizon so labels are mature before boundary",
        applies_to="both",
    )
    violations = embargo_spec.validate_against_label(contract)
    if violations:
        raise FactorIdentityAuthorityError(
            "LabelContract embargo insufficient: " + "; ".join(violations)
        )
