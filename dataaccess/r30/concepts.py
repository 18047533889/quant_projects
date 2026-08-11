"""data_access.r30.concepts —— R30-P0-012：Canonical ConceptId + 一等 UnitType。

R30 全维度成熟度层是 **additive** 的：不修改任何既有 store/read/runtime 文件。
本模块把 R24 semantic_catalog 里的 ``SemanticField.logical_name``（自由字符串）
与 ``UnitSpec``（dimension/currency/scale 三字段）升级成**一等对象**：

    - ``ConceptId``：规范化概念标识（``price.close`` / ``financial.net_income``），
      提供 family / subpath / descendant / logical-name 往返；compile-time 用它做
      语义契约的统一 key。
    - ``UnitType``：带单位代数的类型系统（Return/Ratio/Price/Money/Shares/Volume/
      Count/Days）。核心不变式：**Money[CNY] + Money[USD] 在没有 FX contract 时
      compile-time 直接 reject**（``check_addable`` / ``check_multiply``）。
    - ``MarketContext``：市场级稳定上下文（market_id/currency/calendar 等），
      ``ASHARE`` / ``US`` 预设常量。

原始数据本身不改，只升级 semantic metadata / compiler 层：向既有 API 的映射
（``UnitType.from_unit_spec`` / ``to_unit_spec``）放在本层，不触碰 semantic_catalog。

维护人：quant 基础平台组    最后更新：2026-08-11
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar


class UnitError(ValueError):
    """单位代数 / 概念标识非法时的异常。

    继承 ``ValueError``（非 DataAccessError 层级），保持本层自包含——
    R30 概念对象不依赖 core.exceptions，避免耦合既有包。
    """


# ---------------------------------------------------------------------------
# Canonical ConceptId（R30-P0-012）
# ---------------------------------------------------------------------------

_CONCEPT_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


class ConceptId:
    """规范化概念标识。

    形如 ``price.close`` / ``price.vwap`` / ``return.close_to_close`` /
    ``valuation.market_cap`` / ``financial.revenue`` /
    ``financial.net_income.consolidated`` / ``financial.total_assets`` /
    ``fundamental.roe``。

    两种构造等价：

        ConceptId("price.close")          # 单参数点分字符串
        ConceptId("price", "close")       # 多段参数

    构造即校验；非法标识抛 ``UnitError``。``from_logical_name`` 是宽松入口
    （非法返回 None 不抛）。

    与既有 ``SemanticField.logical_name`` 的关系：本类型是**一等对象**，
    ``to_logical_name()`` 向下输出点连接字符串，向上可再经 catalog 消歧。
    """

    __slots__ = ("_segments",)

    def __init__(self, *parts: str) -> None:
        if len(parts) == 1 and isinstance(parts[0], str) and "." in parts[0]:
            segments = tuple(parts[0].split("."))
        else:
            segments = tuple(str(p) for p in parts)
        text = ".".join(segments)
        if not _CONCEPT_RE.fullmatch(text):
            raise UnitError(
                f"非法 ConceptId {text!r}：需形如 'price.close'，"
                "每段 ^[a-z][a-z0-9_]*$ 且至少一段含点号"
            )
        self._segments = segments

    # ---- 访问器 ----

    @property
    def family(self) -> str:
        """首段，如 ``price.close`` → ``price``。"""
        return self._segments[0]

    @property
    def subpath(self) -> tuple[str, ...]:
        """段元组，如 ``financial.net_income`` → (\"financial\", \"net_income\")。"""
        return self._segments

    def is_descendant_of(self, other: "ConceptId | str") -> bool:
        """self 是否是 other 的严格后代（other 段是 self 段的前缀）。

        ``other`` 可以是 ``ConceptId`` 或字符串；无点号的裸字符串按 family 前缀
        处理（``ConceptId(\"price.close\").is_descendant_of(\"price\")`` → True）。
        """
        if isinstance(other, ConceptId):
            osub = other.subpath
        elif isinstance(other, str):
            osub = tuple(other.split(".")) if "." in other else (other,)
        else:
            return False
        if len(self._segments) <= len(osub):
            return False
        return self._segments[: len(osub)] == osub

    def to_logical_name(self) -> str:
        """点连接输出（向下映射到既有 logical_name 字符串）。"""
        return ".".join(self._segments)

    def is_valid(self) -> bool:
        """实例方法：构造已保证合法，恒 True；保留给对已构造对象的语义查询。"""
        return True

    @staticmethod
    def is_valid_name(text: str) -> bool:
        """宽松校验一个字符串是否合法概念标识（不构造对象）。"""
        return bool(_CONCEPT_RE.fullmatch(str(text).strip()))

    @classmethod
    def from_logical_name(cls, name: Any) -> "ConceptId | None":
        """从点连接字符串构造；非法返回 None（不抛）。

        与严格构造函数不同——数据来自 catalog 外部输入时用本入口做 fail-open
        探测，合法才升级为一等对象。
        """
        if not isinstance(name, str):
            return None
        text = name.strip()
        if not _CONCEPT_RE.fullmatch(text):
            return None
        return cls(text)

    # ---- dunder ----

    def __str__(self) -> str:
        return self.to_logical_name()

    def __repr__(self) -> str:
        return f"ConceptId({self.to_logical_name()!r})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ConceptId):
            return NotImplemented
        return self._segments == other._segments

    def __hash__(self) -> int:
        return hash(self._segments)


# ---------------------------------------------------------------------------
# 一等 UnitType（R30-P0-012）
# ---------------------------------------------------------------------------

# kind → 规范显示名（__str__ / parse 的 canonical form）
_KIND_CANONICAL: dict[str, str] = {
    "return": "Return",
    "ratio": "Ratio",
    "price": "Price",
    "money": "Money",
    "shares": "Shares",
    "volume": "Volume",
    "count": "Count",
    "days": "Days",
}
_VALID_KINDS = frozenset(_KIND_CANONICAL)
# 需要显式币种的 kind（"currency 类型"）
_CURRENCY_KINDS = frozenset({"money", "price"})


@dataclass(frozen=True)
class UnitType:
    """一等单位类型：``kind`` + 可选 ``currency``。

    kind ∈ {return, ratio, price, money, shares, volume, count, days}；
    ``price`` / ``money`` 携带币种（``Money[CNY]`` / ``Price[USD]``）。

    单位代数（compile-time 检查的核心）：

        - ``can_add``：同 kind 且（无量纲类型，或币种相同）→ 可加；
          ``Money[CNY] + Money[USD]`` 不可加（无 FX contract）。
        - ``can_multiply`` / ``multiply_result``：核心规则 + 保守失败——
          Return×Price→Price、Ratio×Ratio→Ratio、Money/Shares→Price、
          Ratio 缩放任意度量单位；未覆盖组合抛 ``UnitError``。
        - ``requires_fx``：判断两个同 kind 币种类型跨币种相加是否需要 FX。
    """

    kind: str
    currency: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise UnitError(f"未知单位类型 {self.kind!r}（应为 {sorted(_VALID_KINDS)}）")
        if self.currency is not None:
            # 币种统一大写（ISO 4217 风格），保证 Money[cny] == Money[CNY]
            object.__setattr__(self, "currency", str(self.currency).upper())

    # ---- 类方法构造 ----

    @classmethod
    def return_(cls) -> "UnitType":
        return cls("return")

    @classmethod
    def ratio(cls) -> "UnitType":
        return cls("ratio")

    @classmethod
    def price(cls, currency: str) -> "UnitType":
        return cls("price", currency)

    @classmethod
    def money(cls, currency: str) -> "UnitType":
        return cls("money", currency)

    @classmethod
    def shares(cls) -> "UnitType":
        return cls("shares")

    @classmethod
    def volume(cls) -> "UnitType":
        return cls("volume")

    @classmethod
    def count(cls) -> "UnitType":
        return cls("count")

    @classmethod
    def days(cls) -> "UnitType":
        return cls("days")

    # ---- 字符串解析 / 输出 ----

    @classmethod
    def parse(cls, expr: str) -> "UnitType":
        """解析 ``"Money[CNY]"`` / ``"Price[USD]"`` / ``"Return"`` / ``"Ratio"``。

        大小写不敏感（``money[cny]`` == ``Money[CNY]``）；非法表达式抛 ``UnitError``。
        """
        text = str(expr).strip()
        m = re.fullmatch(r"([A-Za-z]+)(?:\[([A-Za-z0-9_]+)\])?", text)
        if not m:
            raise UnitError(f"无法解析单位表达式 {expr!r}（应为 Money[CNY] / Return / ...）")
        kind = m.group(1).lower()
        if kind not in _VALID_KINDS:
            raise UnitError(f"无法解析单位表达式 {expr!r}：未知 kind {kind!r}")
        currency = m.group(2)
        return cls(kind, currency)

    def __str__(self) -> str:
        if self.currency is not None:
            return f"{_KIND_CANONICAL[self.kind]}[{self.currency}]"
        return _KIND_CANONICAL[self.kind]

    def __repr__(self) -> str:
        return f"UnitType({str(self)})"

    # ---- 加法代数 ----

    def can_add(self, other: "UnitType") -> bool:
        """同 kind 且（无量纲类型，或币种相同）→ 可加。

        ``Money[CNY] + Money[USD]`` → False（无 FX）。
        """
        if not isinstance(other, UnitType):
            return False
        if self.kind != other.kind:
            return False
        if self.kind in _CURRENCY_KINDS:
            return self.currency == other.currency
        return True

    def add_result(self, other: "UnitType") -> "UnitType":
        """同 kind 的可加结果单位；kind 不同抛 ``UnitError``。

        币种取 self 优先（相同币种时无歧义；不同币种仅当调用方已持有 FX 时由
        ``check_addable(..., fx_contract=True)`` 调用）。
        """
        if not isinstance(other, UnitType) or self.kind != other.kind:
            raise UnitError(f"{self} + {other} 单位类型不同，不可相加")
        return UnitType(self.kind, self.currency or other.currency)

    def requires_fx(self, other: "UnitType") -> bool:
        """两个同 kind 币种类型且币种不同 → 需要 FX 才能统一。

        无量纲类型 / 不同 kind / 有任一币种未声明 → False（后者是语义不完备，
        由调用方按「不可加」处理，不当作需要 FX）。
        """
        if not isinstance(other, UnitType):
            return False
        if self.kind != other.kind or self.kind not in _CURRENCY_KINDS:
            return False
        return bool(
            self.currency and other.currency and self.currency != other.currency
        )

    # ---- 乘法代数 ----

    def can_multiply(self, other: "UnitType") -> bool:
        try:
            self.multiply_result(other)
            return True
        except UnitError:
            return False

    def multiply_result(self, other: "UnitType") -> "UnitType":
        """单位乘法结果（核心规则 + 保守失败）。

        规则表（不追求完备）：
            - Return × Price  → Price（Return 是无量纲比例，等价恒等）
            - Ratio × Ratio   → Ratio
            - Money / Shares  → Price（Money × Shares⁻¹；乘法器里 Shares 视为倒数）
            - Price × Shares  → Money
            - Ratio 缩放任意度量单位（Ratio × X → X，X × Ratio → X）
        未覆盖组合抛 ``UnitError``。
        """
        if not isinstance(other, UnitType):
            raise UnitError(f"不能与 {other!r} 做单位乘法")
        a, b = self.kind, other.kind
        if a == "return" and b == "price":
            return UnitType("price", other.currency)
        if a == "price" and b == "return":
            return UnitType("price", self.currency)
        if a == "ratio" and b == "ratio":
            return UnitType("ratio")
        if a == "money" and b == "shares":
            return UnitType("price", self.currency)
        if a == "shares" and b == "money":
            return UnitType("price", other.currency)
        if a == "price" and b == "shares":
            return UnitType("money", self.currency)
        if a == "shares" and b == "price":
            return UnitType("money", other.currency)
        # 无量纲 Ratio 缩放任意度量单位
        if b == "ratio":
            return UnitType(a, self.currency)
        if a == "ratio":
            return UnitType(b, other.currency)
        raise UnitError(f"不支持的单位乘法: {self} × {other}")

    # ---- 与既有 UnitSpec（semantic_catalog）的映射 ----

    # UnitType.kind → UnitSpec.dimension（best-effort；UnitSpec 没有 volume/days，
    # 保留原样字符串——UnitSpec.dimension 是 str，不强制枚举）。
    _DIMENSION_MAP: ClassVar[dict[str, str]] = {
        "return": "ratio",  # 回报本质是价格比，dimension 用 ratio
        "ratio": "ratio",
        "price": "price",
        "money": "money",
        "shares": "shares",
        "volume": "volume",
        "count": "count",
        "days": "days",
    }

    def to_unit_spec(self) -> Any:
        """向下映射到 ``data_access.read.semantic_catalog.UnitSpec``。

        惰性 import（防御性）：本层不依赖 semantic_catalog 的加载；映射仅在显式
        调用时发生。``cross_market_comparable`` 对币种类型且已声明币种置 False——
        与 R24 T-X03 的跨市场拒绝语义一致。
        """
        from data_access.read.semantic_catalog import UnitSpec

        dimension = self._DIMENSION_MAP.get(self.kind, self.kind)
        return UnitSpec(
            dimension=dimension,
            currency=self.currency,
            scale=None,
            source_unit=str(self),
            canonical_unit=str(self),
            cross_market_comparable=(
                self.kind not in _CURRENCY_KINDS or self.currency is None
            ),
            requires_fx=False,
        )

    @classmethod
    def from_unit_spec(cls, spec: Any) -> "UnitType":
        """从 ``UnitSpec`` 向上映射回一等 ``UnitType``（best-effort）。

        防御性访问：用 ``getattr`` 取 dimension/currency；未知 dimension 回退
        Ratio（无量纲，保守不报错——调用方做 compile-time 检查时再 fail-closed）。
        """
        dimension = getattr(spec, "dimension", None)
        currency = getattr(spec, "currency", None)
        if dimension == "money":
            return cls.money(currency) if currency else cls("money")
        if dimension == "price":
            return cls.price(currency) if currency else cls("price")
        if dimension == "shares":
            return cls.shares()
        if dimension == "count":
            return cls.count()
        if dimension == "volume":
            return cls.volume()
        if dimension == "days":
            return cls.days()
        if dimension in ("ratio", "return"):
            return cls.ratio()
        return cls.ratio()


# ---------------------------------------------------------------------------
# 全局加法/乘法检查器（compile-time）
# ---------------------------------------------------------------------------


def check_addable(a: UnitType, b: UnitType, fx_contract: bool = False) -> UnitType:
    """可加返回结果单位；不可加且无 FX → ``UnitError``。

    - ``Money[CNY] + Money[CNY]`` → Money[CNY]
    - ``Money[CNY] + Money[USD]`` 无 fx_contract → UnitError（需要 FX contract）
    - ``Money[CNY] + Money[USD]`` 有 fx_contract → Money[CNY]（以 self 币种为基准）
    - 不同 kind（Ratio + Money）→ UnitError（单位不兼容，与 FX 无关）
    """
    if a.can_add(b):
        return a.add_result(b)
    if a.requires_fx(b):
        if not fx_contract:
            raise UnitError(f"{a} + {b} 需要 FX contract")
        return a.add_result(b)
    raise UnitError(f"{a} + {b} 不可相加：单位不兼容")


def check_multiply(a: UnitType, b: UnitType) -> UnitType:
    """可乘返回结果单位；不可乘抛 ``UnitError``（乘法不涉及 FX）。"""
    return a.multiply_result(b)


# ---------------------------------------------------------------------------
# MarketContext（R30-P1-005 的市场级稳定上下文）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketContext:
    """市场级稳定上下文：compile-time 确定市场 → 币种/日历/命名空间。"""

    market_id: str
    currency: str | None = None
    calendar_id: str | None = None
    instrument_namespace: str | None = None


ASHARE = MarketContext(
    market_id="ashare",
    currency="CNY",
    calendar_id="ashare",
    instrument_namespace="ashare",
)
"""A股市场预设（CNY / ashare 日历）。"""

US = MarketContext(
    market_id="us",
    currency="USD",
    calendar_id="us",
    instrument_namespace="us",
)
"""美股市场预设（USD / us 日历）。"""


__all__ = [
    "ASHARE",
    "ConceptId",
    "MarketContext",
    "UnitError",
    "UnitType",
    "US",
    "check_addable",
    "check_multiply",
]
