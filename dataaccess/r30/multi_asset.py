"""data_access.r30.multi_asset —— R30-P2-002 多资产 Instrument Contract。

铺接口:核心 identity / grain 模型**不得假设所有 instrument 都是永久股票
ticker**。为未来期货 / 期权铺底:

  - ``DerivativeContractSpec``:衍生品合约规格(期货 / 期权);
  - ``asset_class_grain(asset_class)``:资产类别 → 数据粒度键。

``InstrumentIdentity`` 位于 ``data_access.r30.specs``(其他 agent 建设,可能尚未
就绪)—— 互转辅助一律**懒加载 + try/except**,缺失时抛清晰错误;签名变化时用
``inspect.signature`` 过滤构造参数,防御性降级。
"""
from __future__ import annotations

import inspect
from typing import Any

__all__ = ["DerivativeContractSpec", "asset_class_grain"]


def _import_specs_module() -> Any | None:
    """懒加载 ``data_access.r30.specs`` 模块(未建成返回 None)。"""
    try:
        import importlib

        return importlib.import_module("data_access.r30.specs")
    except Exception:  # noqa: BLE001 —— 模块可能尚未建成
        return None


def _identity_classes():
    """返回 ``(InstrumentIdentity 类, InstrumentType 枚举)``,缺失则为 None。"""
    mod = _import_specs_module()
    if mod is None:
        return None, None
    return getattr(mod, "InstrumentIdentity", None), getattr(mod, "InstrumentType", None)


def _only_accepted_kwargs(cls: Any, kwargs: dict) -> dict:
    """只保留构造器接受的 kwargs(签名变化不炸)。"""
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return kwargs
    if "kwargs" in params:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


class DerivativeContractSpec:
    """衍生品合约规格。

    - 期货: ``underlying + contract_id + expiry(+ multiplier)``,无 strike / option_type;
    - 期权: 另含 ``strike`` 与 ``option_type``(``"call"`` / ``"put"``)。

    ``contract_id`` 在期货 / 期权里是**可交易合约标识**(例如 AAPLZ26C260),
    不是永久股票 ticker;``instrument_namespace`` 区分不同交易所 / 体系。
    """

    def __init__(
        self,
        underlying: str,
        contract_id: str,
        expiry: str | None = None,
        multiplier: float | None = None,
        strike: float | None = None,
        option_type: str | None = None,
        instrument_namespace: str = "default",
    ) -> None:
        self.underlying = underlying
        self.contract_id = contract_id
        self.expiry = expiry
        self.multiplier = multiplier
        self.strike = strike
        self.option_type = option_type
        self.instrument_namespace = instrument_namespace

    # ------------------------------------------------------------------
    def is_futures(self) -> bool:
        """期货:无期权类型(无 strike / option_type)。"""
        return self.option_type is None

    def is_options(self) -> bool:
        """期权:显式给出 option_type(call/put)。"""
        return self.option_type is not None

    # ------------------------------------------------------------------
    # InstrumentIdentity 互转(懒 import 防御性;specs 未建成 → 清晰报错)
    # ------------------------------------------------------------------
    def to_instrument_identity(self) -> Any:
        """构造 ``specs.InstrumentIdentity``(懒加载;模块未建 → RuntimeError)。"""
        ident_cls, type_enum = _identity_classes()
        if ident_cls is None:
            raise RuntimeError(
                "data_access.r30.specs 尚未建成;InstrumentIdentity 互转暂不可用"
            )
        instrument_type = None
        if type_enum is not None:
            instrument_type = (
                type_enum.OPTION if self.is_options() else type_enum.FUTURE
            )
        kwargs: dict = {
            "symbol": self.contract_id,
            "security_id": self.contract_id,
            "market": self.instrument_namespace,
        }
        if instrument_type is not None:
            kwargs["instrument_type"] = instrument_type
        return ident_cls(**_only_accepted_kwargs(ident_cls, kwargs))

    @classmethod
    def from_instrument_identity(cls, ident: Any) -> "DerivativeContractSpec":
        """从 ``specs.InstrumentIdentity`` 反向构造(懒加载;模块未建 → 报错)。"""
        ident_cls, _ = _identity_classes()
        if ident_cls is None:
            raise RuntimeError(
                "data_access.r30.specs 尚未建成;InstrumentIdentity 互转暂不可用"
            )
        symbol = getattr(ident, "symbol", None) or getattr(ident, "contract_id", None)
        security_id = getattr(ident, "security_id", None) or symbol
        market = (
            getattr(ident, "market", None)
            or getattr(ident, "namespace", None)
            or "default"
        )
        instrument_type = getattr(ident, "instrument_type", None)
        option_type = None
        if instrument_type is not None:
            type_name = str(getattr(instrument_type, "name", instrument_type)).upper()
            if "OPTION" in type_name:
                option_type = getattr(ident, "option_type", None) or "call"
        return cls(
            underlying=getattr(ident, "underlying", None) or symbol,
            contract_id=str(security_id or symbol),
            expiry=getattr(ident, "expiry", None),
            multiplier=getattr(ident, "multiplier", None),
            strike=getattr(ident, "strike", None),
            option_type=option_type,
            instrument_namespace=str(market),
        )


def asset_class_grain(asset_class: str) -> tuple[str, str]:
    """资产类别 → 数据粒度键 ``(time_dim, instrument_dim)``。

      - EQUITY → ``('trade_date', 'instrument')``;
      - FUTURE → ``('trade_date', 'contract_id')``;
      - OPTION → ``('trade_date', 'contract_id')``。

    未知类别 fail-closed 抛 ``ValueError``(不静默猜粒度)。
    """
    cls = str(asset_class or "").strip().upper()
    if cls in {"EQUITY", "STOCK", "EQUITY_STOCK"}:
        return ("trade_date", "instrument")
    if cls in {"FUTURE", "FUTURES"}:
        return ("trade_date", "contract_id")
    if cls in {"OPTION", "OPTIONS"}:
        return ("trade_date", "contract_id")
    raise ValueError(f"不支持的 asset_class: {asset_class!r} (支持 EQUITY/FUTURE/OPTION)")
