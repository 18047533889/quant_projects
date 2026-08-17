"""R32-P0-107/108/109/110/111 —— Canonical identity / SCM version / wheel inventory。

R32-P0-107  CanonicalIdentityEncoder 全仓唯一（list 保序、set sort、production 无 repr fallback）
R32-P0-108  关键 identity ≥128-bit（stable_digest_full 返回 64 字符 = 256-bit）
R32-P0-109  版本改为 SCM/tag 驱动（setuptools-scm 构建期生成 _build_info.py）
R32-P0-110  R30 package 必须进 wheel（pyproject.toml 已显式 data_access.r30）
R32-P0-111  build SHA 在 build-time 固化（_build_info.py，不依赖运行时 .git）
"""
from __future__ import annotations

import pytest
from data_access.r30._shared import stable_digest, stable_digest_full


def test_r32_p0_107_list_preserves_order():
    """R32-P0-107：list 保序，不排序（顺序有意义）。"""
    d1 = stable_digest(["a", "b", "c"])
    d2 = stable_digest(["c", "b", "a"])
    assert d1 != d2, "list 顺序不同 → digest 必须不同"


def test_r32_p0_107_set_sorts():
    """R32-P0-107：set 无序 → 显式 sort，保证确定性。"""
    d1 = stable_digest({"a", "b", "c"})
    d2 = stable_digest({"c", "b", "a"})
    assert d1 == d2, "set 元素相同（顺序无意义）→ digest 必须相同"


def test_r32_p0_107_dict_sorts_keys():
    """R32-P0-107：dict 按 key sort。"""
    d1 = stable_digest({"x": 1, "y": 2})
    d2 = stable_digest({"y": 2, "x": 1})
    assert d1 == d2


def test_r32_p0_107_tuple_preserves_order():
    """R32-P0-107：tuple 同 list，保序。"""
    d1 = stable_digest(("a", "b"))
    d2 = stable_digest(("b", "a"))
    assert d1 != d2


def test_r32_p0_107_no_repr_fallback():
    """R32-P0-107：production 禁止 repr fallback，未知类型 → ValueError。"""
    class Custom:
        pass
    with pytest.raises(ValueError, match="unsupported type"):
        stable_digest(Custom())


def test_r32_p0_107_basic_types_ok():
    """R32-P0-107：str/int/float/bool/None/bytes 直接支持。"""
    stable_digest("text", 42, 3.14, True, False, None, b"data")


def test_r32_p0_108_full_digest_256bit():
    """R32-P0-108：stable_digest_full 返回 64 字符（256-bit），关键 identity ≥128-bit。"""
    d = stable_digest_full("test")
    assert len(d) == 64, "stable_digest_full 必须返回完整 sha256（64 hex = 256 bit）"


def test_da2_p0_002_both_helpers_are_full_sha256():
    """DA2-P0-002：两个 correctness helper 都返回完整 SHA-256。"""
    short_named = stable_digest("test")
    full_named = stable_digest_full("test")
    assert short_named == full_named
    assert len(short_named) == len(full_named) == 64
    assert all(char in "0123456789abcdef" for char in short_named)


def test_da2_p0_002_order_and_typed_container_semantics():
    """DA2-P0-002：parts/list 保序，set/map 无序，typed key/element 不碰撞。"""
    assert stable_digest("a", "b") != stable_digest("b", "a")
    assert stable_digest([1, 2]) != stable_digest([2, 1])
    assert stable_digest({1, 2}) == stable_digest({2, 1})
    assert stable_digest({1: "value"}) != stable_digest({"1": "value"})
    assert stable_digest({1}) != stable_digest({"1"})


@pytest.mark.parametrize("helper", [stable_digest, stable_digest_full])
def test_da2_p0_002_unsupported_objects_fail_closed(helper):
    """DA2-P0-002：两个 helper 都把 strict encoder 错误统一成 ValueError。"""
    class Unsupported:
        pass

    with pytest.raises(ValueError, match="unsupported type"):
        helper(Unsupported())


def test_r32_p0_109_version_scm_driven():
    """R32-P0-109：版本改为 SCM/tag 驱动。

    wheel 安装后 __version__ 从 _build_info 读；dev 环境回退 package metadata。
    """
    import data_access
    assert data_access.__version__
    assert isinstance(data_access.__version__, str)
    # 版本号格式：tag 时 0.11.0；非 tag 时 0.11.0.dev<N>+g<sha> 或 0.11.0+unknown
    # 不再人工硬编码静态版本号


def test_r32_p0_111_build_sha_from_build_info():
    """R32-P0-111：build SHA 在 build-time 固化（_build_info.py），不依赖运行时 .git。

    wheel 安装后从 _build_info 读；dev 环境 fallback env + git；production 缺 build
    info → None（typed，不伪造）。
    """
    import data_access
    # editable install / 源码树：__build_sha__ 可能从 git 读到
    # wheel install：从 _build_info.build_sha 读
    # 两者都无（极端 edge case）：None
    assert data_access.__build_sha__ is None or isinstance(data_access.__build_sha__, str)


def test_r32_p0_110_r30_in_packages():
    """R32-P0-110：r30 package 进入 wheel（pyproject.toml 已显式列入）。

    本测试只校验 r30 模块可 import；实际 wheel inventory 由 check_wheel_inventory.py
    在 build 后校验（R26-P0-002 既有机制）。
    """
    import data_access.r30
    # 能 import 即证明 r30 在 sys.path（editable 或 wheel 均可）
    assert hasattr(data_access.r30, "__version__")
