"""Instrument identity must be namespaced by market.

``instrument = "A"`` is not globally unique: ``ashare:000001.SZ`` (Ping An Bank)
and ``us:AAPL`` are different instruments.  Every cache / factor-lake / lineage
key uses ``(market, instrument, trade_date)`` so same-named tickers never collide.
"""
from __future__ import annotations

from dataclasses import dataclass


# A-share 6 位数字代码的交易所推断（#226）。
def _infer_ashare_exchange(code: str) -> str:
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "SH"
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return "SZ"
    if code.startswith(("430", "83", "87", "88", "92")):
        return "BJ"
    return "SH"


class InstrumentNormalizer:
    """R40 #226：market-specific instrument 规范化工厂。

    旧的 ``InstrumentKey.__post_init__`` 只做 strip+lower，不做市场级 canonical
    form。本类把同一证券的各种拼写折叠成 canonical：

    * A 股：``000001.SZ`` / ``SZ000001`` / ``000001`` -> ``000001.SZ``
      （``sh600000`` -> ``600000.SH``，交易所按代码段推断）；
    * 美股：``aapl`` -> ``AAPL``（大写 canonical，保留 ``BRK.B`` 点分格式）。

    ``canonical_security_id`` / ``display_ticker`` / ``provider_symbol`` 分别给
    出不同消费方的规范化形式。``for_market(market)`` 是工厂入口。
    """

    def __init__(self, market: str, exchange: str = "") -> None:
        self.market = str(market or "").strip().lower()
        self.exchange = str(exchange or "").strip().upper()

    @classmethod
    def for_market(cls, market: str, exchange: str = "") -> "InstrumentNormalizer":
        key = str(market).strip().lower()
        if key in {"a_share", "cn", "china"}:
            key = "ashare"
        if key not in {"ashare", "us"}:
            raise ValueError(f"no instrument normalizer for market {market!r}")
        return cls(market=key, exchange=exchange)

    def canonical_security_id(self, instrument: str) -> str:
        """同一证券的规范代码（A 股带交易所后缀，美股大写）。"""
        s = str(instrument or "").strip()
        if not s:
            return s
        if self.market == "us":
            return s.upper()
        # A 股
        low = s.lower()
        if low.startswith(("sh", "sz", "bj")):
            prefix, code = low[:2].upper(), s[2:]
            return f"{code}.{prefix}"
        if s.endswith((".sh", ".sz", ".bj", ".SH", ".SZ", ".BJ")):
            code, suffix = s.rsplit(".", 1)
            return f"{code}.{suffix.upper()}"
        if s.isdigit() and len(s) == 6:
            return f"{s}.{_infer_ashare_exchange(s)}"
        return s

    def display_ticker(self, instrument: str) -> str:
        """面向展示的 ticker（A 股无后缀，美股原样）。"""
        s = str(instrument or "").strip()
        if self.market == "us":
            return s.upper()
        canonical = self.canonical_security_id(s)
        return canonical.split(".")[0] if canonical.endswith((".SH", ".SZ", ".BJ")) else canonical

    def provider_symbol(self, instrument: str) -> str:
        """数据源侧的 symbol（通常无交易所后缀）。"""
        s = str(instrument or "").strip()
        if self.market == "us":
            return s.upper()
        canonical = self.canonical_security_id(s)
        return canonical.split(".")[0]

    def normalize(self, instrument: str) -> str:
        return self.canonical_security_id(instrument)


@dataclass(frozen=True, order=True)
class InstrumentKey:
    """A market-namespaced instrument identity."""

    market: str
    instrument: str

    def __post_init__(self) -> None:
        if not isinstance(self.market, str) or not self.market.strip():
            raise ValueError(f"market must be a non-empty string, got {self.market!r}")
        if not isinstance(self.instrument, str) or not self.instrument.strip():
            raise ValueError(
                f"instrument must be a non-empty string, got {self.instrument!r}"
            )
        object.__setattr__(self, "market", self.market.strip().lower())
        object.__setattr__(self, "instrument", self.instrument.strip())

    def __str__(self) -> str:
        return f"{self.market}:{self.instrument}"

    @classmethod
    def parse(cls, qualified: str) -> "InstrumentKey":
        if ":" not in str(qualified):
            raise ValueError(
                f"instrument key must be 'market:instrument', got {qualified!r}"
            )
        market, instrument = str(qualified).split(":", 1)
        return cls(market=market, instrument=instrument)

    @classmethod
    def normalized(cls, market: str, instrument: str, exchange: str = "") -> "InstrumentKey":
        """R40 #226：用 market-specific canonical form 构造 InstrumentKey。"""
        norm = InstrumentNormalizer.for_market(market, exchange)
        return cls(market=norm.market, instrument=norm.canonical_security_id(instrument))

    def canonicalized(self) -> "InstrumentKey":
        """返回本 key 的 market-normalized canonical 形式。"""
        try:
            return InstrumentKey.normalized(self.market, self.instrument)
        except ValueError:
            return self

    def as_tuple(self) -> tuple[str, str]:
        return (self.market, self.instrument)


def qualified_instrument(market: str, instrument: str) -> str:
    """Return the canonical ``market:instrument`` spelling."""
    return str(InstrumentKey(market=market, instrument=instrument))


__all__ = ["InstrumentKey", "InstrumentNormalizer", "qualified_instrument"]

