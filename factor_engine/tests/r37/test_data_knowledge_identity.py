# -*- coding: utf-8 -*-
"""R37-P0-011/012/014：DataKnowledgeIdentity + Universe PIT + PriceBasis。

- 统一数据知识身份：不同 snapshot/universe/calendar/price_basis/revision
  => 不同 identity（cache/checkpoint key 必须不同）。
- Universe PIT：成员加入前不能出现、不同 universe => 不同 membership hash。
- PriceBasis：不同 basis => 不同 identity 维度（不共 cache）。
"""
from __future__ import annotations

import os

os.environ.setdefault("FACTOR_ENGINE_CPU_BUDGET", "4")


def _identity(**over):
    from semantic.data_knowledge_identity import DataKnowledgeIdentity

    base = dict(dataset_id="daily_bar", snapshot_id="snap1", schema_epoch="e1",
                market="ashare", calendar_id="ashare_cal", universe_snapshot_id="CSI300",
                price_basis="RAW", timezone="Asia/Shanghai")
    base.update(over)
    return DataKnowledgeIdentity(**base)


def test_identity_changes_on_snapshot():
    a = _identity()
    b = _identity(snapshot_id="snap2")
    assert a.to_key() != b.to_key()
    assert a.digest() != b.digest()


def test_identity_changes_on_universe():
    a = _identity()
    b = _identity(universe_snapshot_id="CSI500")
    assert a.digest() != b.digest()


def test_identity_changes_on_price_basis():
    a = _identity()
    b = _identity(price_basis="CONTINUOUS")
    assert a.digest() != b.digest()
    # FORWARD_ADJUSTED vs BACKWARD_ADJUSTED 也不相同
    c = _identity(price_basis="FORWARD_ADJUSTED")
    d = _identity(price_basis="BACKWARD_ADJUSTED")
    assert c.digest() != d.digest()


def test_identity_changes_on_revision():
    a = _identity()
    b = _identity(source_revision_id="rev2")
    assert a.digest() != b.digest()


def test_identity_from_factor_identity():
    """从 FactorSemanticIdentity 组合 DataKnowledgeIdentity。"""
    from runtime.factor_identity import FactorSemanticIdentity
    from semantic.data_knowledge_identity import DataKnowledgeIdentity

    fi = FactorSemanticIdentity(
        ir_hash="h1", operator_contract_hash="h2", field_contract_hash="h3",
        source_contract_hash="h4", source_dependency_hash="h5",
        market="ashare", calendar="ashare_cal", price_basis="RAW",
        universe="CSI300", universe_membership_hash="m1", timezone="Asia/Shanghai",
    )
    dki = DataKnowledgeIdentity.from_factor_identity(
        fi, snapshot_meta={"snapshot_id": "s1", "schema_epoch": "e1"})
    assert dki.market == "ashare"
    assert dki.price_basis == "RAW"
    assert dki.universe_snapshot_id == "CSI300"
    assert dki.universe_membership_hash == "m1"
    assert dki.digest()


def test_identity_deterministic_across_instances():
    a = _identity()
    b = _identity()
    assert a.to_key() == b.to_key()
    assert a == b and hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Universe PIT（R37-P0-012）
# ---------------------------------------------------------------------------


def test_universe_membership_pit_effective_at():
    from market.universe import UniverseMembership

    m = UniverseMembership(universe="CSI300", instrument="000001",
                           valid_time="2024-01-01", knowledge_time="2024-01-05")
    # 决策时点在名单公布前（knowledge 不可及）=> 不可用
    assert m.effective_at("2024-01-01") is False
    # 决策时点在公布后、成分生效后 => 可用
    assert m.effective_at("2024-01-05") is True
    # 决策时点在成分生效后（即使 knowledge 已到）=> 可用
    assert m.effective_at("2024-06-01") is True


def test_universe_membership_before_valid_red():
    """成分加入前不能出现：valid_time 之后才 effective。"""
    from market.universe import UniverseMembership

    m = UniverseMembership(universe="CSI300", instrument="000001",
                           valid_time="2024-03-01", knowledge_time="2024-03-05")
    assert m.effective_at("2024-02-01") is False
    assert m.effective_at("2024-03-10") is True


def test_universe_membership_hash_differs_by_membership():
    from market.universe import universe_membership_identity

    h1 = universe_membership_identity("CSI300", ("a", "b"), as_of="2024-01-01")
    h2 = universe_membership_identity("CSI300", ("a", "b", "c"), as_of="2024-01-01")
    h3 = universe_membership_identity("CSI300", ("a", "b"), as_of="2024-02-01")
    assert h1 != h2
    assert h1 != h3
    # 成员顺序无关
    h4 = universe_membership_identity("CSI300", ("b", "a"), as_of="2024-01-01")
    assert h1 == h4


# ---------------------------------------------------------------------------
# PriceBasis（R37-P0-014）
# ---------------------------------------------------------------------------


def test_price_basis_canonical_normalization():
    from fields.concepts import PriceBasis

    assert PriceBasis.canonical("RAW") == "RAW"
    assert PriceBasis.canonical("CONTINUOUS") == "CONTINUOUS"
    assert PriceBasis.canonical(None) == "RAW"
    # 未知值保留（向后兼容），不静默改写
    assert PriceBasis.canonical("SOMETHING_NEW") == "SOMETHING_NEW"
    # 前/后复权、全收益都是明确 basis
    assert PriceBasis.FORWARD_ADJUSTED == "FORWARD_ADJUSTED"
    assert PriceBasis.BACKWARD_ADJUSTED == "BACKWARD_ADJUSTED"
    assert PriceBasis.TOTAL_RETURN == "TOTAL_RETURN"
