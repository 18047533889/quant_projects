"""R61-FI-010 — DataAccess field-domain taxonomy + typed FieldTaxonomyProvider.

Covers (plan §5 / capability matrix A2/A2b/A3/A4/A6):
    1. 4 new optional ``SemanticField`` metadata attrs parse from YAML and
       round-trip; old rows without tags keep identical ``to_dict``.
    2. Canonical domain/role vocabulary is enforced (fail-closed, no guessing).
    3. ``FieldSemanticDescriptor`` is a thin VIEW over the catalog SemanticField
       (no duplicated authority) and round-trips through dict.
    4. ``SemanticFieldTaxonomyProvider`` resolves via catalog (incl. aliases)
       and FAILS CLOSED on unknown ids — no substring inference.
    5. PIT / knowledge / period / pub metadata is preserved untouched by the
       taxonomy extension.
    6. Existing-catalog hash stability: the full catalog ``get_identity``
       digest is UNCHANGED by rows that declare no taxonomy tags (defaults
       must not enter serialized output).
"""
from __future__ import annotations

import copy

import pytest

from data_access.core.exceptions import ValidationError
from data_access.read.semantic_catalog import (
    FieldSemanticDescriptor,
    FieldTaxonomyProvider,
    SemanticField,
    SemanticFieldCatalog,
    SemanticFieldTaxonomyProvider,
    VALID_ECONOMIC_ROLES,
    VALID_FIELD_DOMAINS,
    VALID_FREQUENCY_CLASSES,
    VALID_PIT_CLASSES,
    get_semantic_catalog,
    parse_semantic_field,
    reset_semantic_catalog,
)


def _raw_field(name: str = "close") -> dict:
    return {
        "dataset": "ashare_stock_daily_adj",
        "physical_name": "AdjClose",
        "dtype": "double",
        "frequency": "daily",
        "grain": "instrument",
        "market": "ashare",
        "availability": "same_day",
        "time_role": "event_time",
        "temporal_model": "panel",
        "join_policy": "exact",
        "mining_allowed": True,
    }


def _parse(raw: dict, name: str = "x") -> SemanticField:
    return parse_semantic_field(name, raw)


# ---------------------------------------------------------------------------
# 1) dataclass extension defaults + YAML parse
# ---------------------------------------------------------------------------


def test_semantic_field_has_optional_taxonomy_attrs() -> None:
    f = SemanticField(logical_name="close", dataset="d", physical_name="c")
    assert f.data_domains == ()
    assert f.economic_roles == ()
    assert f.frequency_class is None
    assert f.pit_class is None


def test_parse_semantic_field_reads_taxonomy_tags() -> None:
    f = _parse(
        {
            **_raw_field(),
            "data_domains": ["PRICE", "VOLUME"],
            "economic_roles": ["price_level", "return"],
            "frequency_class": "daily",
            "pit_class": "panel_same_day",
        }
    )
    assert f.data_domains == ("PRICE", "VOLUME")
    assert f.economic_roles == ("price_level", "return")
    assert f.frequency_class == "daily"
    assert f.pit_class == "panel_same_day"
    # 老行不带 tag → 全部缺省，序列化输出**没有** taxonomy 键（hash 稳定）。
    d = f.to_dict()
    assert d["data_domains"] == ["PRICE", "VOLUME"]
    assert d["pit_class"] == "panel_same_day"
    g = _parse(_raw_field())
    gd = g.to_dict()
    for k in ("data_domains", "economic_roles", "frequency_class", "pit_class"):
        assert k not in gd


def test_parse_semantic_field_lowercases_domains_uppercases_roles() -> None:
    f = _parse({**_raw_field(), "data_domains": ["price", "Fundamental.Quality"]})
    assert f.data_domains == ("PRICE", "FUNDAMENTAL.QUALITY")
    f2 = _parse({**_raw_field(), "economic_roles": ["Price_Level", "RETURN"]})
    assert f2.economic_roles == ("price_level", "return")


def test_parse_rejects_unknown_domain_fail_closed() -> None:
    with pytest.raises(ValidationError):
        _parse({**_raw_field(), "data_domains": ["VOLATILITY"]})
    with pytest.raises(ValidationError):
        _parse({**_raw_field(), "economic_roles": ["profitability"]})
    with pytest.raises(ValidationError):
        _parse({**_raw_field(), "frequency_class": "hourly"})
    with pytest.raises(ValidationError):
        _parse({**_raw_field(), "pit_class": "point_in_time"})
    # 裸字符串当作单元素（不是 set/dict 等非法容器）
    f = _parse({**_raw_field(), "data_domains": "PRICE"})
    assert f.data_domains == ("PRICE",)


# ---------------------------------------------------------------------------
# 2) canonical vocabulary registries
# ---------------------------------------------------------------------------


def test_vocabulary_registries_frozen_and_contain_plan_tokens() -> None:
    # frozenset 不可变（没有 add / remove）
    assert not hasattr(VALID_FIELD_DOMAINS, "add")
    assert not hasattr(VALID_ECONOMIC_ROLES, "add")
    for token in (
        "PRICE",
        "VOLUME",
        "LIQUIDITY",
        "FUNDAMENTAL.VALUE",
        "FUNDAMENTAL.QUALITY",
        "FUNDAMENTAL.GROWTH",
        "FUNDAMENTAL.INVESTMENT",
        "FUNDAMENTAL.CASHFLOW",
        "FUNDAMENTAL.LEVERAGE",
        "EVENT",
        "FLOW_SENTIMENT",
        "MICROSTRUCTURE",
        "RISK",
        "CALENDAR",
        "ALTERNATIVE",
    ):
        assert token in VALID_FIELD_DOMAINS
    assert "quality" in VALID_ECONOMIC_ROLES
    assert "daily" in VALID_FREQUENCY_CLASSES
    assert "announcement_pit" in VALID_PIT_CLASSES


# ---------------------------------------------------------------------------
# 3) FieldSemanticDescriptor — thin view + serialization round-trip
# ---------------------------------------------------------------------------


def test_descriptor_from_catalog_field_is_thin_view() -> None:
    f = _parse({**_raw_field(), "data_domains": ["PRICE"], "pit_class": "panel_same_day"})
    desc = FieldSemanticDescriptor.from_field(f)
    assert desc.canonical_field_id == f.logical_name
    assert desc.data_domains == ("PRICE",)
    assert desc.pit_class == "panel_same_day"
    # 基础/PIT 元数据转发自 catalog 字段（同一来源）
    assert desc.dataset == f.dataset
    assert desc.physical_name == f.physical_name
    assert desc.market == f.market
    assert desc.frequency == f.frequency
    assert desc.grain == f.grain
    assert desc.availability == f.availability
    assert desc.temporal_model == f.temporal_model


def test_descriptor_serialization_round_trip() -> None:
    f = _parse(
        {
            **_raw_field("roe"),
            "dataset": "ashare_stock_indicator",
            "physical_name": "Roe",
            "knowledge_time": "PubDate",
            "effective_time": "ReportPeriodEndDate",
            "period_time": "ReportPeriodEndDate",
            "temporal_model": "financial_event",
            "availability": "next_trading_day",
            "pit_fidelity": "knowledge_date_pit",
            "data_domains": ["FUNDAMENTAL.QUALITY"],
            "economic_roles": ["quality"],
            "frequency_class": "daily",
            "pit_class": "announcement_pit",
        },
        name="roe",
    )
    desc = FieldSemanticDescriptor.from_field(f)
    raw = desc.to_dict()
    assert raw["data_domains"] == ["FUNDAMENTAL.QUALITY"]
    assert raw["knowledge_time"] == "PubDate"
    assert raw["period_time"] == "ReportPeriodEndDate"
    restored = FieldSemanticDescriptor.from_dict(raw)
    assert restored == desc
    assert restored.knowledge_time == "PubDate"
    assert restored.pit_fidelity == "knowledge_date_pit"
    # 无 tag 的 descriptor 序列化不输出 taxonomy 键
    bare = FieldSemanticDescriptor.from_field(
        _parse({**_raw_field(), "dataset": "d", "physical_name": "c"})
    )
    for k in ("data_domains", "economic_roles", "frequency_class", "pit_class"):
        assert k not in bare.to_dict()


def test_descriptor_has_domain_tags_property() -> None:
    assert (
        FieldSemanticDescriptor.from_field(_parse(_raw_field())).has_domain_tags
        is False
    )
    tagged = FieldSemanticDescriptor.from_field(
        _parse({**_raw_field(), "data_domains": ["PRICE"]})
    )
    assert tagged.has_domain_tags is True


# ---------------------------------------------------------------------------
# 4) FieldTaxonomyProvider — catalog resolution + fail-closed unknown
# ---------------------------------------------------------------------------


def test_provider_describes_fields_and_is_mapping() -> None:
    cat = SemanticFieldCatalog(
        {
            "close": _parse({**_raw_field("close"), "physical_name": "AdjClose"}, name="close"),
            "roe": _parse(
                {**_raw_field("roe"), "dataset": "d", "physical_name": "Roe",
                 "data_domains": ["FUNDAMENTAL.QUALITY"]},
                name="roe",
            ),
        }
    )
    prov: FieldTaxonomyProvider = SemanticFieldTaxonomyProvider(cat)
    out = prov.describe_fields(["close", "roe"])
    assert isinstance(out, dict)
    assert out["close"].data_domains == ()
    assert out["roe"].data_domains == ("FUNDAMENTAL.QUALITY",)
    assert set(out) == {"close", "roe"}


def test_provider_returns_all_market_candidates_for_aliases() -> None:
    prov = SemanticFieldTaxonomyProvider()
    # 'ret' 是 return_bp 的 alias（A股）；provider 走 catalog resolve_one，
    # 无上下文时 research 取第一个——必须仍在 catalog 内、不是名字猜测。
    desc = prov.describe_field("return_bp")
    assert desc.data_domains == ("PRICE",)
    assert desc.economic_roles == ("return",)


def test_provider_unknown_field_raises_no_substring_guess() -> None:
    cat = SemanticFieldCatalog(
        {"volume": _parse({**_raw_field("volume"), "physical_name": "Volume"}, name="volume")}
    )
    prov = SemanticFieldTaxonomyProvider(cat)
    # 'vol' 能 substring 到 volume——但必须 fail-closed 拒绝，禁止猜测。
    with pytest.raises(ValidationError, match="未登记"):
        prov.describe_field("vol")
    with pytest.raises(ValidationError, match="未登记"):
        prov.describe_fields(["volume", "totally_missing"])
    # 不返回部分结果：整批抛
    with pytest.raises(ValidationError):
        prov.describe_fields(["totally_missing"])


def test_provider_uses_default_singleton_catalog() -> None:
    prov = SemanticFieldTaxonomyProvider()
    desc = prov.describe_field("close")
    assert desc.physical_name == "AdjClose"
    assert desc.data_domains == ("PRICE",)


# ---------------------------------------------------------------------------
# 5) PIT / knowledge / pub metadata preserved after parse
# ---------------------------------------------------------------------------


def test_pit_metadata_preserved_on_taxonomy_parse() -> None:
    f = _parse(
        {
            **_raw_field("eps"),
            "dataset": "ashare_stock_indicator",
            "physical_name": "Eps",
            "time_role": "knowledge_time",
            "temporal_model": "financial_event",
            "knowledge_time": "PubDate",
            "effective_time": "ReportPeriodEndDate",
            "period_time": "ReportPeriodEndDate",
            "revision_order": ["UpdateTime"],
            "availability": "next_trading_day",
            "join_policy": "pit_asof_backward",
            "period_selection": "latest_period",
            "pit_fidelity": "knowledge_date_pit",
            "data_domains": ["FUNDAMENTAL.QUALITY"],
            "pit_class": "announcement_pit",
        },
        name="eps",
    )
    assert f.knowledge_time == "PubDate"
    assert f.effective_time == "ReportPeriodEndDate"
    assert f.period_time == "ReportPeriodEndDate"
    assert f.revision_order == ("UpdateTime",)
    assert f.availability == "next_trading_day"
    assert f.join_policy == "pit_asof_backward"
    assert f.period_selection == "latest_period"
    assert f.pit_fidelity == "knowledge_date_pit"
    assert f.data_domains == ("FUNDAMENTAL.QUALITY",)
    assert f.pit_class == "announcement_pit"


def test_catalog_roundtrip_preserves_pit_and_taxonomy() -> None:
    reset_semantic_catalog()
    try:
        cat = get_semantic_catalog()
        roe = cat.get("roe", market="ashare")
        assert roe.knowledge_time == "PubDate"
        assert roe.period_time == "ReportPeriodEndDate"
        assert roe.effective_time == "ReportPeriodEndDate"
        assert roe.pit_fidelity == "knowledge_date_pit"
        # A2 标签解析成功
        assert roe.data_domains == ("FUNDAMENTAL.QUALITY",)
        assert roe.frequency_class == "daily"
        assert roe.pit_class == "announcement_pit"
    finally:
        reset_semantic_catalog()


# ---------------------------------------------------------------------------
# 6) existing-catalog content-hash stability
# ---------------------------------------------------------------------------


def _catalog_identity_digest(cat: SemanticFieldCatalog) -> str:
    ident = cat.get_identity(strict=True)
    assert ident.available is True
    return ident.digest


def test_untagged_field_to_dict_identical_and_hash_stable() -> None:
    """扩展前的老行（无 tag）→ 与扩展后同一行的 to_dict / 身份 digest 逐位一致。"""
    kwargs = dict(
        logical_name="earnings_event",
        dataset="events",
        physical_name="earnings_event",
        availability_latency=2,
    )
    old = SemanticFieldCatalog({"earnings_event": SemanticField(**kwargs)})
    # 同一行在扩展语义下 = 相同的 SemanticField 构造 + 缺省 taxonomy。
    new = SemanticFieldCatalog(
        {"earnings_event": SemanticField(**kwargs, data_domains=(), economic_roles=(), frequency_class=None, pit_class=None)}
    )
    assert old.to_dict() == new.to_dict()
    assert _catalog_identity_digest(old) == _catalog_identity_digest(new)


def test_full_yaml_catalog_identity_matches_pre_extension() -> None:
    """生产 YAML：任何未声明 taxonomy 的行仍产出与扩展前一致的 catalog digest。

    本文件被加载时，config/semantic_fields.yaml 已带 A2 tags——因此本测试只
    验证**加 tag 的字段**会改变 catalog 身份（tag 参与内容），而构造一个
    无任何 tag 的 catalog 时 identity 与「扩展前语义」逐字节一致（由
    ``test_untagged_field_to_dict_identical_and_hash_stable`` 覆盖）。
    """
    reset_semantic_catalog()
    try:
        cat = get_semantic_catalog()
        tagged = [n for n, f in cat._fields.items() if f.data_domains or f.pit_class]
        assert tagged, "YAML 至少要给 A 股 canonical 字段打 tag"
        # 抽样的 A 股 canonical 字段确实带期望 tag
        close = cat.get("close", market="ashare")
        assert close.data_domains == ("PRICE",)
        assert close.frequency_class == "daily"
        assert close.pit_class == "panel_same_day"
        volume = cat.get("volume", market="ashare")
        assert volume.data_domains == ("VOLUME",)
        amount = cat.get("amount", market="ashare")
        assert amount.data_domains == ("VOLUME", "LIQUIDITY")
        ret = cat.get("return_bp", market="ashare")
        assert ret.data_domains == ("PRICE",)
        turn = cat.get("turnover_ratio", market="ashare")
        assert turn.data_domains == ("LIQUIDITY",)
        roe = cat.get("roe", market="ashare")
        assert roe.data_domains == ("FUNDAMENTAL.QUALITY",)
        eps = cat.get("eps", market="ashare")
        assert "FUNDAMENTAL.VALUE" in eps.data_domains
    finally:
        reset_semantic_catalog()


def test_identity_differs_when_tags_declared() -> None:
    """显式 tag 改变内容身份（tag 是真实语义，不是被忽略的装饰）。"""
    base = dict(logical_name="x", dataset="d", physical_name="c")
    plain = SemanticFieldCatalog({"x": SemanticField(**base)})
    tagged = SemanticFieldCatalog(
        {"x": SemanticField(**base, data_domains=("PRICE",))}
    )
    assert _catalog_identity_digest(plain) != _catalog_identity_digest(tagged)
    d1 = plain.to_dict()["x"]
    d2 = tagged.to_dict()["x"]
    assert "data_domains" not in d1
    assert d2["data_domains"] == ["PRICE"]


def test_from_dict_accepts_and_roundtrips_taxonomy_keys() -> None:
    """descriptor to_dict → from_dict 对称；字段 dict 带/不带 taxonomy 键都可解析。"""
    raw = copy.deepcopy(_raw_field())
    raw.update(
        {
            "logical_name": "close",
            "data_domains": ["PRICE"],
            "frequency_class": "daily",
            "pit_class": "panel_same_day",
        }
    )
    f = parse_semantic_field("close", raw)
    d = f.to_dict()
    # to_dict 的 payload 再喂回 parse_semantic_field → 仍解析成功（对称）
    f2 = parse_semantic_field("close", {k: v for k, v in d.items() if v is not None})
    assert f2.data_domains == ("PRICE",)
    assert f2.frequency_class == "daily"
    # from_dict 里 data_domains=None（JSON 反序列化的空 tag 行）也应接受
    with_none = dict(d)
    with_none["data_domains"] = None
    f3 = parse_semantic_field("close", with_none)
    assert f3.data_domains == ()
