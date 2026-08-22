"""简单测试运行器（不依赖 pytest）。

直接运行测试，避免导入问题。
"""
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 现在可以导入
from cache.unified_cache import UnifiedCache, EvictionPolicy
from cache.tiered_cache import TieredCache, create_standard_tiered_cache
from cache.cache_decorator import cached, cached_property, memoize
from cache.cache_manager import GlobalCacheManager, get_global_cache_manager

import time
import threading


def test_unified_cache_basic():
    """测试统一缓存基本功能。"""
    print("Testing UnifiedCache basic operations...")
    cache = UnifiedCache[int](max_size=10)

    cache.set("key1", 100)
    cache.set("key2", 200)

    assert cache.get("key1") == 100
    assert cache.get("key2") == 200
    assert cache.get("key3") is None

    stats = cache.stats()
    assert stats.hits == 2
    assert stats.misses == 1
    print("✓ Basic operations passed")


def test_ttl():
    """测试 TTL 过期。"""
    print("Testing TTL expiration...")
    cache = UnifiedCache[str](default_ttl=0.1)

    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"

    time.sleep(0.15)
    assert cache.get("key1") is None
    print("✓ TTL expiration passed")


def test_lru():
    """测试 LRU 逐出。"""
    print("Testing LRU eviction...")
    cache = UnifiedCache[int](max_size=3, eviction_policy=EvictionPolicy.LRU)

    cache.set("k1", 1)
    cache.set("k2", 2)
    cache.set("k3", 3)
    cache.get("k1")  # 访问 k1
    cache.set("k4", 4)  # 应逐出 k2

    assert cache.get("k1") == 1
    assert cache.get("k2") is None
    assert cache.get("k3") == 3
    assert cache.get("k4") == 4
    print("✓ LRU eviction passed")


def test_decorator():
    """测试装饰器。"""
    print("Testing @cached decorator...")
    call_count = [0]

    @cached(maxsize=10)
    def compute(x: int) -> int:
        call_count[0] += 1
        return x * x

    assert compute(5) == 25
    assert call_count[0] == 1

    assert compute(5) == 25
    assert call_count[0] == 1  # 从缓存

    assert compute(10) == 100
    assert call_count[0] == 2

    info = compute.cache_info()
    assert info["hits"] == 1
    assert info["misses"] == 2
    print("✓ Decorator passed")


def test_tiered_cache():
    """测试分层缓存。"""
    print("Testing TieredCache...")
    tiered = create_standard_tiered_cache(
        l1_max_bytes=1024,
        l2_max_bytes=2048,
        l3_max_bytes=4096,
    )

    tiered.set("key1", "value1")
    assert tiered.get("key1") == "value1"

    stats = tiered.stats()
    assert "l1" in stats
    assert "l2" in stats
    assert "l3" in stats
    assert "total" in stats
    print("✓ TieredCache passed")


def test_global_manager():
    """测试全局管理器。"""
    print("Testing GlobalCacheManager...")
    manager = get_global_cache_manager()

    cache1 = UnifiedCache[int](name="test_cache_1")
    cache2 = UnifiedCache[str](name="test_cache_2")

    manager.register("cache1", cache1)
    manager.register("cache2", cache2)

    cache1.set("key1", 100)
    cache2.set("key2", "value2")

    stats = manager.global_stats()
    assert stats["cache_count"] == 2
    assert stats["total"]["entry_count"] == 2

    # 清理
    manager.unregister("cache1")
    manager.unregister("cache2")
    print("✓ GlobalCacheManager passed")


def test_concurrent():
    """测试并发安全。"""
    print("Testing concurrent access...")
    cache = UnifiedCache[int](max_size=100)
    errors = []

    def worker(tid: int):
        try:
            for i in range(50):
                key = f"key_{i % 25}"
                cache.set(key, tid * 1000 + i)
                value = cache.get(key)
                assert value is not None
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    print("✓ Concurrent access passed")


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("Running Cache Strategy Optimization Tests")
    print("=" * 60)

    tests = [
        test_unified_cache_basic,
        test_ttl,
        test_lru,
        test_decorator,
        test_tiered_cache,
        test_global_manager,
        test_concurrent,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
