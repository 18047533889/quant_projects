"""简单测试运行器（使用 sys.path，不用 importlib.util）。"""
import sys
import os
import time
import threading

# 添加项目根目录到 path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 通过 sys.path 进行正常 import（会触发 cache/__init__.py，但我们需要确保这是安全的）
# 直接从模块文件导入，绕过 __init__.py

import importlib.util as _ilu

def _load_direct(module_name, relative_path):
    """直接加载模块文件，注册到 sys.modules[module_name]。"""
    full_path = os.path.join(project_root, relative_path)
    # 只有在还没被加载时才加载
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = _ilu.spec_from_file_location(module_name, full_path)
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = "cache"
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# 按依赖顺序加载
_unified = _load_direct("cache.unified_cache", "cache/unified_cache.py")
_tiered = _load_direct("cache.tiered_cache", "cache/tiered_cache.py")
_decorator = _load_direct("cache.cache_decorator", "cache/cache_decorator.py")
_manager = _load_direct("cache.cache_manager", "cache/cache_manager.py")

UnifiedCache = _unified.UnifiedCache
EvictionPolicy = _unified.EvictionPolicy
TieredCache = _tiered.TieredCache
create_standard_tiered_cache = _tiered.create_standard_tiered_cache
cached = _decorator.cached
get_global_cache_manager = _manager.get_global_cache_manager


def test_unified_cache_basic():
    print("Testing UnifiedCache basic operations...")
    cache = UnifiedCache(max_size=10)
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
    print("Testing TTL expiration...")
    cache = UnifiedCache(default_ttl=0.1)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    time.sleep(0.15)
    assert cache.get("key1") is None
    print("✓ TTL expiration passed")


def test_lru():
    print("Testing LRU eviction...")
    cache = UnifiedCache(max_size=3, eviction_policy=EvictionPolicy.LRU)
    cache.set("k1", 1)
    cache.set("k2", 2)
    cache.set("k3", 3)
    cache.get("k1")  # 访问 k1，标为 MRU
    cache.set("k4", 4)  # 应逐出 k2（最久未使用）
    assert cache.get("k1") == 1
    assert cache.get("k2") is None
    assert cache.get("k3") == 3
    assert cache.get("k4") == 4
    print("✓ LRU eviction passed")


def test_pattern_clear():
    print("Testing pattern clear...")
    cache = UnifiedCache()
    cache.set("user:1:name", 1)
    cache.set("user:1:email", 2)
    cache.set("user:2:name", 3)
    cache.set("post:1:title", 4)
    count = cache.clear(pattern="user:1:*")
    assert count == 2
    assert cache.get("user:1:name") is None
    assert cache.get("user:1:email") is None
    assert cache.get("user:2:name") == 3
    assert cache.get("post:1:title") == 4
    print("✓ Pattern clear passed")


def test_factory():
    print("Testing factory function...")
    call_count = [0]
    def factory():
        call_count[0] += 1
        return 42
    cache = UnifiedCache()
    val = cache.get("k", factory=factory)
    assert val == 42
    assert call_count[0] == 1
    val2 = cache.get("k", factory=factory)
    assert val2 == 42
    assert call_count[0] == 1  # no second call
    print("✓ Factory function passed")


def test_pin_unpin():
    print("Testing pin/unpin...")
    cache = UnifiedCache(max_size=2, eviction_policy=EvictionPolicy.LRU)
    cache.set("k1", 1)
    cache.set("k2", 2)
    cache.pin("k1")
    # 插入第三个：k2 是最久未使用（且 k1 被 pin），应逐出 k2
    cache.set("k3", 3)
    assert cache.get("k1") == 1  # pinned → 保留
    assert cache.get("k2") is None  # 逐出
    assert cache.get("k3") == 3
    cache.unpin("k1")
    print("✓ Pin/unpin passed")


def test_invalidate():
    print("Testing invalidate...")
    cache = UnifiedCache()
    cache.set("key1", 100)
    assert cache.get("key1") == 100
    assert cache.invalidate("key1") is True
    assert cache.get("key1") is None
    assert cache.invalidate("key1") is False
    print("✓ Invalidate passed")


def test_stats():
    print("Testing stats...")
    cache = UnifiedCache(max_size=2)
    cache.set("k1", 1)
    cache.set("k2", 2)
    cache.get("k1")  # hit
    cache.get("k3")  # miss
    cache.set("k3", 3)  # may evict
    cache.invalidate("k1")
    s = cache.stats()
    assert s.hits == 1
    assert s.misses == 1
    assert s.sets == 3
    assert s.deletes == 1
    print("✓ Stats passed")


def test_decorator():
    print("Testing @cached decorator...")
    call_count = [0]

    @cached(maxsize=10)
    def compute(x):
        call_count[0] += 1
        return x * x

    assert compute(5) == 25
    assert call_count[0] == 1
    assert compute(5) == 25
    assert call_count[0] == 1  # from cache
    assert compute(10) == 100
    assert call_count[0] == 2

    info = compute.cache_info()
    assert info["hits"] == 1
    assert info["misses"] == 2

    # cache_clear
    compute.cache_clear()
    compute(5)
    assert call_count[0] == 3
    print("✓ Decorator passed")


def test_tiered_cache():
    print("Testing TieredCache...")
    tiered = create_standard_tiered_cache(
        l1_max_bytes=1024 * 1024,
        l2_max_bytes=2 * 1024 * 1024,
        l3_max_bytes=4 * 1024 * 1024,
    )
    tiered.set("key1", "value1")
    assert tiered.get("key1") == "value1"
    tiered.invalidate("key1")
    assert tiered.get("key1") is None

    stats = tiered.stats()
    assert "l1" in stats
    assert "l2" in stats
    assert "l3" in stats
    assert "total" in stats
    print("✓ TieredCache passed")


def test_global_manager():
    print("Testing GlobalCacheManager...")
    manager = get_global_cache_manager()

    cache1 = UnifiedCache(name="gc_test_1")
    cache2 = UnifiedCache(name="gc_test_2")

    manager.register("gc_test_1", cache1)
    manager.register("gc_test_2", cache2)

    cache1.set("a", 1)
    cache2.set("b", "hello")

    stats = manager.global_stats()
    assert stats["cache_count"] >= 2
    assert stats["total"]["entry_count"] >= 2

    health = manager.health_check()
    assert "gc_test_1" in health
    assert "gc_test_2" in health

    # summary
    summary = manager.summary()
    assert "Global Cache Manager" in summary

    manager.unregister("gc_test_1")
    manager.unregister("gc_test_2")
    print("✓ GlobalCacheManager passed")


def test_concurrent():
    print("Testing concurrent access...")
    cache = UnifiedCache(max_size=100)
    errors = []

    def worker(tid):
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

    assert len(errors) == 0, f"Concurrent errors: {errors}"
    print("✓ Concurrent access passed")


def main():
    print("=" * 60)
    print("Cache Strategy Optimization Tests")
    print("=" * 60)

    tests = [
        test_unified_cache_basic,
        test_ttl,
        test_lru,
        test_pattern_clear,
        test_factory,
        test_pin_unpin,
        test_invalidate,
        test_stats,
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
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
