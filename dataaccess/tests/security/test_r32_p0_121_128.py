"""R32-P0-121..128 聚焦测试：Symlink TOCTOU、production lock fail-closed、
price basis fail-closed、路径 canonicalization。

覆盖项：
    P0-121  symlink 解析原子化（assert_no_symlink_escape / canonicalize）
    P0-122  production mutation lock fail-closed（无锁根 → 拒绝）
    P0-123  price basis 未知值 fail-closed（不回落 RAW）
    P0-124  canonicalize_strict（resolve(strict=True)，不存在即拒）
    P0-125  因子湖 discover 拒绝 symlink 逃逸
    P0-126  _clear_dir 单次 lstat，不跟随 symlink
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from data_access.core.exceptions import DataError, ValidationError


# ============================================================================
# R32-P0-121: Symlink TOCTOU 原子化解析
# ============================================================================


def test_symlink_toctou_atomic_resolution(tmp_path):
    """symlink 指向沙箱外 → 解析后比较，逃逸被拒（无 check/use 窗口）。"""
    from data_access.registry.paths import assert_no_symlink_escape

    root = tmp_path / "allowed_root"
    root.mkdir()
    outside = tmp_path / "outside_secret"
    outside.mkdir()
    (outside / "secret.parquet").write_text("stolen")

    # 沙箱内的普通文件 → 放行，返回 realpath
    inner = root / "data.parquet"
    inner.write_text("ok")
    resolved = assert_no_symlink_escape(inner, root)
    assert resolved == Path(os.path.realpath(str(inner)))

    # 沙箱内放一个指向沙箱外的 symlink → 必须拒绝
    escape_link = root / "innocent_looking.parquet"
    escape_link.symlink_to(outside / "secret.parquet")
    with pytest.raises(ValidationError, match="symlink 逃逸"):
        assert_no_symlink_escape(escape_link, root)


def test_symlink_toctou_directory_component_escape(tmp_path):
    """逃逸发生在**中间路径组件**（非尾部）同样必须被拒。

    这是两步式 `if not p.is_symlink()` 检查最典型的漏网场景：尾部组件是普通
    文件，但父目录是 symlink，跟随后整棵子树都在沙箱外。
    """
    from data_access.registry.paths import assert_no_symlink_escape

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.parquet").write_text("x")

    # root/subdir 是 symlink → outside；root/subdir/payload.parquet 尾部不是 symlink
    (root / "subdir").symlink_to(outside, target_is_directory=True)
    victim = root / "subdir" / "payload.parquet"
    assert victim.exists()  # 跟随后确实存在
    assert not victim.is_symlink()  # 尾部组件自身不是 symlink

    with pytest.raises(ValidationError, match="symlink 逃逸"):
        assert_no_symlink_escape(victim, root)


def test_symlink_resolution_is_realpath_based_not_two_step(tmp_path):
    """回归护栏：解析结果必须等于 os.path.realpath（整链解析，非逐段判定）。"""
    from data_access.registry.paths import assert_no_symlink_escape

    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    (root / "real" / "f.txt").write_text("v")
    # 沙箱**内**的 symlink（指向沙箱内）→ 应放行，且返回解析后的真实路径
    (root / "alias").symlink_to(root / "real", target_is_directory=True)

    out = assert_no_symlink_escape(root / "alias" / "f.txt", root)
    assert out == Path(os.path.realpath(str(root / "real" / "f.txt")))
    assert "alias" not in str(out)  # 确认真的解析了，不是原样返回


def test_symlink_loop_rejected_by_canonicalize_strict(tmp_path):
    """symlink 环路不能挂死/逃逸，必须抛 ValidationError。"""
    from data_access.registry.paths import canonicalize_strict

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.symlink_to(b)
    b.symlink_to(a)

    with pytest.raises(ValidationError, match="无法 canonical 解析"):
        canonicalize_strict(a)


# ============================================================================
# R32-P0-124: Path canonicalization — strict 存在性
# ============================================================================


def test_canonicalize_strict_requires_existence(tmp_path):
    """strict 解析：不存在的路径直接拒绝（不做纯字符串拼接）。"""
    from data_access.registry.paths import canonicalize_strict

    real = tmp_path / "exists.parquet"
    real.write_text("x")
    assert canonicalize_strict(real) == Path(os.path.realpath(str(real)))

    with pytest.raises(ValidationError, match="无法 canonical 解析"):
        canonicalize_strict(tmp_path / "does_not_exist.parquet")


def test_canonicalize_strict_resolves_symlink_to_real_target(tmp_path):
    """strict 解析必须落到 symlink 的真实目标（授权比较的前提）。"""
    from data_access.registry.paths import canonicalize_strict

    target = tmp_path / "target.parquet"
    target.write_text("x")
    link = tmp_path / "link.parquet"
    link.symlink_to(target)

    assert canonicalize_strict(link) == Path(os.path.realpath(str(target)))


def test_path_authorizer_rejects_symlink_escape_after_resolution(tmp_path):
    """PathAuthorizer 的白名单判定建立在 symlink 解析之后。"""
    from data_access.registry.paths import PathAuthorizer

    root = tmp_path / "registered_root"
    root.mkdir()
    outside = tmp_path / "unregistered"
    outside.mkdir()
    (outside / "data.parquet").write_text("x")

    authorizer = PathAuthorizer([str(root)])

    # 沙箱内正常路径 → 通过
    ok = root / "ok.parquet"
    ok.write_text("x")
    assert authorizer.resolve_and_authorize(ok) == Path(os.path.realpath(str(ok)))

    # 沙箱内的 symlink 指向未注册目录 → 解析后越界，必须拒
    link = root / "sneaky.parquet"
    link.symlink_to(outside / "data.parquet")
    with pytest.raises(ValidationError, match="路径越界|symlink"):
        authorizer.resolve_and_authorize(link)


# ============================================================================
# R32-P0-122: Production mutation lock fail-closed
# ============================================================================


def test_production_lock_fail_closed(monkeypatch, tmp_path):
    """strict/production 下解析不出锁根 → 抛异常，绝不静默无锁执行。"""
    from data_access import store as store_mod

    # 模拟 strict 语义开启
    monkeypatch.setattr(store_mod, "_strict_mutation_lock_required", lambda: True)

    class _FakeRegistry:
        def get(self, name):
            return object()  # 非 None 的 dataset，走到锁根解析

    st = store_mod.DataAccessStore.__new__(store_mod.DataAccessStore)
    st._registry = _FakeRegistry()
    # 锁根解析失败路径：非 generation dataset + raw paths 为空 → root=None
    st._is_generation_dataset = lambda ds: False
    st._resolve_raw_paths = lambda ds, time_range=None, params=None: []

    with pytest.raises(ValidationError, match="fail-closed|无法解析 mutation 锁根"):
        with st._dataset_mutation("some_dataset"):
            pass  # pragma: no cover — 不应到达


def test_production_lock_research_mode_still_permissive(monkeypatch):
    """research 模式保留旧行为（无锁放行），避免误伤本地探索。"""
    from data_access import store as store_mod

    monkeypatch.setattr(store_mod, "_strict_mutation_lock_required", lambda: False)

    class _FakeRegistry:
        def get(self, name):
            return object()

    st = store_mod.DataAccessStore.__new__(store_mod.DataAccessStore)
    st._registry = _FakeRegistry()
    st._is_generation_dataset = lambda ds: False
    st._resolve_raw_paths = lambda ds, time_range=None, params=None: []

    entered = False
    with st._dataset_mutation("some_dataset"):
        entered = True
    assert entered, "research 模式应放行"


def test_strict_mutation_lock_required_defaults_closed_on_import_error(monkeypatch):
    """导入期异常时必须**保守判 strict**（不能因 import 顺序降级成 fail-open）。"""
    import builtins

    from data_access import store as store_mod

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "data_access.read.query_budget":
            raise ImportError("simulated circular import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert store_mod._strict_mutation_lock_required() is True


# ============================================================================
# R32-P0-123: Price basis fail-closed
# ============================================================================


def test_price_basis_unknown_fail_closed():
    """未知 price basis 抛异常，**绝不** fallback 到 RAW。"""
    from data_access.r30.specs import PriceBasisSpec

    for bad in ["hfq", "qfq", "forward", "backward_adj", "ADJUSTED", "", "unknown"]:
        with pytest.raises(ValueError, match="未知 price basis"):
            PriceBasisSpec.coerce(bad)


def test_price_basis_none_fail_closed():
    """未声明（None）同样拒绝——"没说"不等于"就是 RAW"。"""
    from data_access.r30.specs import PriceBasisSpec

    with pytest.raises(ValueError, match="未声明"):
        PriceBasisSpec.coerce(None)


def test_price_basis_valid_values_accepted():
    """所有合法枚举值（含大小写/空白）正常解析。"""
    from data_access.r30.specs import PriceBasisSpec

    assert PriceBasisSpec.coerce("raw") is PriceBasisSpec.RAW
    assert PriceBasisSpec.coerce("  RAW  ") is PriceBasisSpec.RAW
    assert (
        PriceBasisSpec.coerce("backward_adjusted") is PriceBasisSpec.BACKWARD_ADJUSTED
    )
    # enum 实例透传
    assert PriceBasisSpec.coerce(PriceBasisSpec.TOTAL_RETURN) is (
        PriceBasisSpec.TOTAL_RETURN
    )
    # 每个成员都能 round-trip
    for member in PriceBasisSpec:
        assert PriceBasisSpec.coerce(member.value) is member


def test_price_basis_never_silently_returns_raw():
    """回归护栏：任何非法输入都不得**返回** RAW（必须抛）。"""
    from data_access.r30.specs import PriceBasisSpec

    for bad in [None, "", "nonsense", 0, 1, [], {}, object()]:
        with pytest.raises(ValueError):
            result = PriceBasisSpec.coerce(bad)
            assert result is not PriceBasisSpec.RAW, (
                f"{bad!r} 静默回落 RAW —— 复权价当未复权会产生除权日假信号"
            )


# ============================================================================
# R32-P0-125: 因子湖 discover 拒绝 symlink 逃逸
# ============================================================================


def test_factor_lake_discover_rejects_symlink_escape(tmp_path):
    """strict 下因子目录 symlink 指向湖外 → DataError。"""
    import json

    from data_access.read.factors import FACTOR_META_FILENAME, FactorCatalog

    lake = tmp_path / "lake"
    factors = lake / "factors"
    factors.mkdir(parents=True)

    # 合法因子
    good = factors / "good_factor"
    good.mkdir()
    (good / FACTOR_META_FILENAME).write_text(json.dumps({"factor_id": "good_factor"}))

    # 湖外目录 + 指向它的 symlink
    outside = tmp_path / "other_principal_lake"
    outside.mkdir()
    (outside / FACTOR_META_FILENAME).write_text(json.dumps({"factor_id": "stolen"}))
    (factors / "stolen").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DataError, match="symlink 逃逸"):
        FactorCatalog.discover(lake, strict=True)


def test_factor_lake_discover_skips_symlink_escape_in_research(tmp_path):
    """research 下跳过逃逸目录（不 raise），但也**绝不**把它读进 catalog。"""
    import json

    from data_access.read.factors import FACTOR_META_FILENAME, FactorCatalog

    lake = tmp_path / "lake"
    factors = lake / "factors"
    factors.mkdir(parents=True)

    good = factors / "good_factor"
    good.mkdir()
    (good / FACTOR_META_FILENAME).write_text(json.dumps({"factor_id": "good_factor"}))

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / FACTOR_META_FILENAME).write_text(json.dumps({"factor_id": "stolen"}))
    (factors / "stolen").symlink_to(outside, target_is_directory=True)

    catalog = FactorCatalog.discover(lake, strict=False)
    assert "stolen" not in catalog.records
    assert "good_factor" in catalog.records


def test_factor_lake_allows_internal_symlink(tmp_path):
    """湖**内部**的 symlink 是合法的（不能误杀）。"""
    import json

    from data_access.read.factors import FACTOR_META_FILENAME, FactorCatalog

    lake = tmp_path / "lake"
    factors = lake / "factors"
    factors.mkdir(parents=True)

    real = factors / "real_factor"
    real.mkdir()
    (real / FACTOR_META_FILENAME).write_text(json.dumps({"factor_id": "real_factor"}))
    # 湖内 alias → 指向同一个湖里的真实因子目录
    (factors / "alias_factor").symlink_to(real, target_is_directory=True)

    catalog = FactorCatalog.discover(lake, strict=True)
    assert "real_factor" in catalog.records


# ============================================================================
# R32-P0-126: _clear_dir 不跟随 symlink
# ============================================================================


def test_clear_dir_does_not_follow_symlink_out_of_sandbox(tmp_path):
    """overwrite 清目录时，symlink→外部目录只删链接本身，外部内容必须完好。"""
    from data_access.store import DataAccessStore

    target = tmp_path / "target"
    target.mkdir()
    (target / "old.parquet").write_text("old")

    outside = tmp_path / "precious"
    outside.mkdir()
    precious = outside / "must_survive.parquet"
    precious.write_text("critical data")

    # target 里放一个指向外部目录的 symlink
    (target / "evil_link").symlink_to(outside, target_is_directory=True)

    DataAccessStore._clear_dir(target)

    # 外部目录与内容必须原样存活
    assert outside.is_dir(), "symlink 被跟随，外部目录被删除"
    assert precious.exists(), "symlink 被跟随，外部文件被删除"
    assert precious.read_text() == "critical data"
    # target 自己被清空
    assert list(target.iterdir()) == []


def test_clear_dir_rejects_symlink_target_root(tmp_path):
    """target 自身是 symlink → 拒绝 overwrite（会清空沙箱外真实目录）。"""
    from data_access.store import DataAccessStore

    real = tmp_path / "real_dir"
    real.mkdir()
    (real / "data.parquet").write_text("x")

    link = tmp_path / "link_dir"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValidationError, match="symlink"):
        DataAccessStore._clear_dir(link)
    assert (real / "data.parquet").exists()


def test_clear_dir_still_removes_real_subdirectories(tmp_path):
    """真实子目录仍必须被递归删除（不能误杀正常功能）。"""
    from data_access.store import DataAccessStore

    target = tmp_path / "target"
    nested = target / "part=1" / "deep"
    nested.mkdir(parents=True)
    (nested / "a.parquet").write_text("x")
    (target / "top.parquet").write_text("y")

    DataAccessStore._clear_dir(target)

    assert target.is_dir()
    assert list(target.iterdir()) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
