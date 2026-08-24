# -*- coding: utf-8 -*-
"""R30-P2-001..004 —— TrainingDatasetSpec / 多资产 / Distributed / API Surface。

纯 Python 验收(不依赖真实 store / snapshot 模块):
    - TrainingDatasetSpec.validate 合法/非法(purge<0、空 features/labels、重叠段);
    - assemble 用 fake store(防御:有 read_joined 委托、缺失/抛错降级);
    - DerivativeContractSpec 期货/期权判定 + asset_class_grain;
    - distributed 本地 provider:锁互斥(两线程)、lease token、metadata KV;
    - api_surface preferred / compat。
"""
from __future__ import annotations

import threading

import pytest

from data_access.r30 import api_surface, distributed as dist
from data_access.r30.multi_asset import DerivativeContractSpec, asset_class_grain
from data_access.r30.training import TrainingDatasetSpec


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeStore:
    """fake store:可选 read_joined(返回 sentinel 或抛错)。"""

    def __init__(self, sentinel: object = None, raise_on_join: bool = False) -> None:
        self.sentinel = sentinel
        self.raise_on_join = raise_on_join
        self.calls: list[dict] = []

    def read_joined(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_on_join:
            raise RuntimeError("fake read_joined boom")
        return self.sentinel


class _FakeSnapshot:
    """带 digest() 的假 ExperimentDataSnapshot(替代真实模块)。"""

    def __init__(self, value: str = "snap-abc") -> None:
        self._value = value

    def digest(self) -> str:
        return self._value


# ---------------------------------------------------------------------------
# TrainingDatasetSpec.validate
# ---------------------------------------------------------------------------


def test_training_spec_validate_ok():
    spec = TrainingDatasetSpec(
        features=("f1", "f2"),
        labels=("y",),
        universe="equity_daily",
        time_range=("2020-01-01", "2020-12-31"),
        train_valid_test=(
            ("2020-01-01", "2020-09-30"),
            ("2020-10-01", "2020-11-30"),
            ("2020-12-01", "2020-12-31"),
        ),
        purge=1,
        embargo=2,
    )
    assert spec.validate() == []


def test_training_spec_validate_purge_embargo_negative():
    spec = TrainingDatasetSpec(features=("f",), labels=("y",), universe="u", purge=-1, embargo=-2)
    errors = spec.validate()
    assert any("purge" in e for e in errors)
    assert any("embargo" in e for e in errors)


def test_training_spec_validate_empty_features_labels():
    spec = TrainingDatasetSpec(features=(), labels=("y",), universe="u")
    errors = spec.validate()
    assert any("features" in e for e in errors)

    spec2 = TrainingDatasetSpec(features=("f",), labels=(), universe="u")
    errors2 = spec2.validate()
    assert any("labels" in e for e in errors2)


def test_training_spec_validate_overlapping_segments():
    spec = TrainingDatasetSpec(
        features=("f",),
        labels=("y",),
        universe="u",
        train_valid_test=(
            ("2020-01-01", "2020-06-30"),
            ("2020-06-01", "2020-11-30"),  # 与上一段重叠
            ("2020-12-01", "2020-12-31"),
        ),
    )
    errors = spec.validate()
    assert any("重叠" in e for e in errors)


def test_training_spec_validate_wrong_segment_count():
    spec = TrainingDatasetSpec(
        features=("f",),
        labels=("y",),
        universe="u",
        train_valid_test=(("2020-01-01", "2020-06-30"), ("2020-07-01", "2020-12-31")),
    )
    errors = spec.validate()
    assert any("3 段" in e for e in errors)


def test_training_spec_to_dict_includes_snapshot_digest():
    spec = TrainingDatasetSpec(
        features=("f1", "f2"),
        labels=("y",),
        universe="u",
        train_valid_test=(
            ("2020-01-01", "2020-09-30"),
            ("2020-10-01", "2020-11-30"),
            ("2020-12-01", "2020-12-31"),
        ),
        experiment_snapshot=_FakeSnapshot("snap-xyz"),
    )
    d = spec.to_dict()
    assert d["features"] == ["f1", "f2"]
    assert d["labels"] == ["y"]
    assert d["train_valid_test"][0] == ["2020-01-01", "2020-09-30"]
    assert d["experiment_snapshot_digest"] == "snap-xyz"
    assert d["purge"] == 0


# ---------------------------------------------------------------------------
# TrainingDatasetSpec.assemble(fake store 防御)
# ---------------------------------------------------------------------------


def test_training_spec_assemble_delegates_to_read_joined():
    sentinel = object()
    fake = _FakeStore(sentinel=sentinel)
    spec = TrainingDatasetSpec(features=("f1",), labels=("y",), universe="u")
    out = spec.assemble(fake)
    assert out is sentinel
    assert fake.calls and fake.calls[0]["anchor"] == "u"
    assert fake.calls[0]["universe"] == "u"


def test_training_spec_assemble_store_without_read_joined():
    spec = TrainingDatasetSpec(features=("f1",), labels=("y",), universe="u")
    out = spec.assemble(object())  # 无 read_joined → 描述性 dict,不抛
    assert isinstance(out, dict)
    assert out["assembled"] is False


def test_training_spec_assemble_delegate_error_degrades():
    fake = _FakeStore(raise_on_join=True)
    spec = TrainingDatasetSpec(features=("f1",), labels=("y",), universe="u")
    out = spec.assemble(fake)
    assert isinstance(out, dict)
    assert out["assembled"] is False
    assert "delegate_error" in out


# ---------------------------------------------------------------------------
# DerivativeContractSpec / asset_class_grain
# ---------------------------------------------------------------------------


def test_derivative_contract_futures_vs_options():
    fut = DerivativeContractSpec(
        underlying="AAPL",
        contract_id="AAPLZ26",
        expiry="2026-12-18",
        multiplier=100.0,
    )
    assert fut.is_futures() is True
    assert fut.is_options() is False

    opt = DerivativeContractSpec(
        underlying="AAPL",
        contract_id="AAPLZ26C260",
        expiry="2026-12-18",
        multiplier=100.0,
        strike=260.0,
        option_type="call",
    )
    assert opt.is_options() is True
    assert opt.is_futures() is False
    assert opt.option_type == "call"


def test_asset_class_grain_mapping():
    assert asset_class_grain("EQUITY") == ("trade_date", "instrument")
    assert asset_class_grain("FUTURE") == ("trade_date", "contract_id")
    assert asset_class_grain("OPTION") == ("trade_date", "contract_id")
    # 大小写不敏感
    assert asset_class_grain("future") == ("trade_date", "contract_id")


def test_asset_class_grain_unknown_fail_closed():
    with pytest.raises(ValueError):
        asset_class_grain("BOND")


def test_derivative_contract_instrument_identity_roundtrip():
    # specs.py 已建成 → 真实互转应 round-trip 保字段。
    opt = DerivativeContractSpec(
        underlying="AAPL",
        contract_id="AAPLZ26C260",
        expiry="2026-12-18",
        multiplier=100.0,
        strike=260.0,
        option_type="call",
        instrument_namespace="us",
    )
    ident = opt.to_instrument_identity()
    assert ident.symbol == "AAPLZ26C260"
    assert ident.market == "us"
    back = DerivativeContractSpec.from_instrument_identity(ident)
    assert back.contract_id == "AAPLZ26C260"
    assert back.instrument_namespace == "us"
    assert back.is_options() is True

    fut = DerivativeContractSpec(underlying="AAPL", contract_id="AAPLZ26")
    back_fut = DerivativeContractSpec.from_instrument_identity(
        fut.to_instrument_identity()
    )
    assert back_fut.is_futures() is True


def test_derivative_contract_instrument_identity_defensive_when_specs_missing(monkeypatch):
    # specs 模块未建成(或 import 失败)→ 懒加载失败,互转必须清晰报错。
    import data_access.r30.multi_asset as ma

    monkeypatch.setattr(ma, "_import_specs_module", lambda: None)
    fut = DerivativeContractSpec(underlying="AAPL", contract_id="AAPLZ26")
    with pytest.raises(RuntimeError):
        fut.to_instrument_identity()
    with pytest.raises(RuntimeError):
        DerivativeContractSpec.from_instrument_identity(object())


# ---------------------------------------------------------------------------
# distributed 本地 provider
# ---------------------------------------------------------------------------


def test_distributed_lock_mutual_exclusion_two_threads():
    provider = dist.LocalLockProvider()
    assert provider.acquire("shared") is True

    results: list = []

    def try_acquire():
        results.append(provider.acquire("shared", timeout=0.0))

    t = threading.Thread(target=try_acquire)
    t.start()
    t.join()
    assert results == [False]  # 主线程持有 → 另一线程立即失败

    assert provider.release("shared") is True

    def acquire_after_release():
        results.append(provider.acquire("shared", timeout=1.0))
        provider.release("shared")

    t2 = threading.Thread(target=acquire_after_release)
    t2.start()
    t2.join()
    assert results == [False, True]


def test_distributed_lock_release_unheld_is_false():
    provider = dist.LocalLockProvider()
    assert provider.acquire("k") is True
    assert provider.release("k") is True
    assert provider.release("k") is False  # 已释放再释放 → False 不抛


def test_distributed_lease_acquire_renew_release():
    provider = dist.LocalLeaseProvider()
    token = provider.acquire("lease-a", ttl=5.0)
    assert token is not None
    assert provider.acquire("lease-a", ttl=5.0) is None  # 已被持有
    assert provider.renew("lease-a", token, ttl=5.0) is True
    assert provider.renew("lease-a", "wrong-token", ttl=5.0) is False
    assert provider.release("lease-a", token) is True
    assert provider.acquire("lease-a", ttl=5.0) is not None  # 释放后可再占


def test_distributed_metadata_store_kv():
    store = dist.LocalMetadataStore()
    store.put("a/1", 1)
    store.put("a/2", 2)
    store.put("b/1", 3)
    assert store.get("a/1") == 1
    assert store.get("missing") is None
    assert store.list("a/") == ["a/1", "a/2"]
    assert store.delete("a/1") is True
    assert store.get("a/1") is None
    assert store.delete("a/1") is False


def test_distributed_get_default_providers():
    providers = dist.get_default_providers()
    assert set(providers) == {"lease", "lock", "metadata"}
    assert isinstance(providers["lease"], dist.LocalLeaseProvider)
    assert isinstance(providers["lock"], dist.LocalLockProvider)
    assert isinstance(providers["metadata"], dist.LocalMetadataStore)
    # 满足 Protocol(结构匹配)
    assert isinstance(providers["lease"], dist.DistributedLeaseProvider)
    assert isinstance(providers["lock"], dist.DistributedLockProvider)
    assert isinstance(providers["metadata"], dist.DistributedMetadataStore)


# ---------------------------------------------------------------------------
# api_surface
# ---------------------------------------------------------------------------


def test_api_surface_preferred():
    assert api_surface.PREFERRED_PUBLIC_API == (
        "read",
        "scan",
        "session",
        "plan",
        "write",
        "publish",
    )
    assert api_surface.is_preferred("read") is True
    assert api_surface.is_preferred("publish") is True
    assert api_surface.is_preferred("read_arrow") is False
    assert api_surface.is_preferred("nope") is False


def test_api_surface_compat_aliases():
    aliases = api_surface.compat_aliases("read")
    assert "read_arrow" in aliases
    assert "read_auto" in aliases
    assert "read_frame" in aliases
    assert "read_uri" in aliases
    assert "read_joined" in aliases
    assert "read_factors" in aliases
    # 别名 → 归属 canonical
    assert api_surface.compat_aliases("read_arrow") == ("read",)
    assert api_surface.compat_aliases("read_factors") == ("read",)
    # 无别名的 canonical / 未知名称
    assert api_surface.compat_aliases("write") == ()
    assert api_surface.compat_aliases("nope") == ()


def test_api_surface_public_surface_dict():
    surf = api_surface.public_api_surface()
    assert set(surf["canonical_entries"]) == set(api_surface.PREFERRED_PUBLIC_API)
    assert surf["preferred_public_api"] == list(api_surface.PREFERRED_PUBLIC_API)
    assert surf["compat_aliases"]["read_arrow"] == "read"
    assert "read_joined" in surf["aliases_by_canonical"]["read"]
