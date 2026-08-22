"""测试统一缓存接口。

R43 缓存策略优化测试：
    - 基本读写操作
    - TTL 过期
    - LRU/LFU/FIFO 逐出
    - Pin/unpin 机制
    - 并发安全
    - 模式匹配清除
    - 统计指标
"""
import time
import threading
from cache.unified_cache import (
    UnifiedCache,
    EvictionPolicy,
    CacheEntry,
    MemoryCacheBackend,
)


def test_basic_get_set():
    """测试基本读写操作。"""
    cache = UnifiedCache[int](max_size=10)

    # 写入
    cache.set("key1", 100)
    cache.set("key2", 200)

    # 读取
    assert cache.get("key1") == 100
    assert cache.get("key2") == 200
    assert cache.get("key3") is None

    # 统计
    stats = cache.stats()
    assert stats.hits == 2
    assert stats.misses == 1
    assert stats.entry_count == 2


def test_ttl_expiration():
    """测试 TTL 过期。"""
    cache = UnifiedCache[str](default_ttl=0.1)

    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"

    # 等待过期
    time.sleep(0.15)
    assert cache.get("key1") is None

    stats = cache.stats()
    assert stats.expirations == 1


def test_lru_eviction():
    """测试 LRU 逐出策略。"""
    cache = UnifiedCache[int](max_size=3, eviction_policy=EvictionPolicy.LRU)

    cache.set("key1", 1)
    cache.set("key2", 2)
    cache.set("key3", 3)

    # 访问 key1（变成最近使用）
    assert cache.get("key1") == 1

    # 插入 key4（应逐出 key2，因为它是最久未使用）
    cache.set("key4", 4)

    assert cache.get("key1") == 1  # 仍在
    assert cache.get("key2") is None  # 被逐出
    assert cache.get("key3") == 3  # 仍在
    assert cache.get("key4") == 4  # 新插入

    stats = cache.stats()
    assert stats.evictions == 1


def test_lfu_eviction():
    """测试 LFU 逐出策略。"""
    cache = UnifiedCache[int](max_size=3, eviction_policy=EvictionPolicy.LFU)

    cache.set("key1", 1)
    cache.set("key2", 2)
    cache.set("key3", 3)

    # 访问 key1 多次（增加访问计数）
    for _ in range(5):
        cache.get("key1")

    # 访问 key2 一次
    cache.get("key2")

    # key3 从未访问

    # 插入 key4（应逐出 key3，因为它访问次数最少）
    cache.set("key4", 4)

    assert cache.get("key1") == 1  # 仍在（访问多）
    assert cache.get("key2") == 2  # 仍在
    assert cache.get("key3") is None  # 被逐出（从未访问）
    assert cache.get("key4") == 4  # 新插入


def test_pin_unpin():
    """测试 pin/unpin 机制。"""
    cache = UnifiedCache[int](max_size=2, eviction_policy=EvictionPolicy.LRU)

    cache.set("key1", 1)
    cache.set("key2", 2)

    # Pin key1
    assert cache.pin("key1") is True

    # 插入 key3（应逐出 key2，因为 key1 被 pin）
    cache.set("key3", 3)

    assert cache.get("key1") == 1  # 仍在（被 pin）
    assert cache.get("key2") is None  # 被逐出
    assert cache.get("key3") == 3  # 新插入

    # Unpin key1
    cache.unpin("key1")

    # 插入 key4（现在可以逐出 key1）
    cache.set("key4", 4)

    # key1 或 key3 被逐出（取决于 LRU 顺序）
    # 由于 key1 被访问过（get），key3 是更早的
    result = cache.get("key1")
    # LRU: key3 被逐出（最早插入且未访问）
    assert cache.get("key3") is None


def test_factory_function():
    """测试 factory 函数。"""
    call_count = [0]

    def factory():
        call_count[0] += 1
        return call_count[0] * 100

    cache = UnifiedCache[int]()

    # 第一次：调用 factory
    value1 = cache.get("key1", factory=factory)
    assert value1 == 100
    assert call_count[0] == 1

    # 第二次：从缓存读取，不调用 factory
    value2 = cache.get("key1", factory=factory)
    assert value2 == 100
    assert call_count[0] == 1  # 没有增加


def test_pattern_clear():
    """测试模式匹配清除。"""
    cache = UnifiedCache[int]()

    cache.set("user:1:name", 1)
    cache.set("user:1:email", 2)
    cache.set("user:2:name", 3)
    cache.set("post:1:title", 4)

    # 清除所有 user:1:* 键
    count = cache.clear(pattern="user:1:*")
    assert count == 2

    assert cache.get("user:1:name") is None
    assert cache.get("user:1:email") is None
    assert cache.get("user:2:name") == 3
    assert cache.get("post:1:title") == 4


def test_concurrent_access():
    """测试并发访问安全。"""
    cache = UnifiedCache[int](max_size=100)
    errors = []

    def worker(thread_id: int):
        try:
            for i in range(100):
                key = f"key_{i % 50}"
                cache.set(key, thread_id * 1000 + i)
                value = cache.get(key)
                assert value is not None
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent access errors: {errors}"


def test_invalidate():
    """测试失效操作。"""
    cache = UnifiedCache[int]()

    cache.set("key1", 100)
    assert cache.get("key1") == 100

    # 失效
    assert cache.invalidate("key1") is True
    assert cache.get("key1") is None

    # 重复失效
    assert cache.invalidate("key1") is False


def test_stats():
    """测试统计指标。"""
    cache = UnifiedCache[int](max_size=2)

    cache.set("key1", 1)
    cache.set("key2", 2)
    cache.get("key1")  # hit (makes key1 most recent in LRU)
    cache.get("key3")  # miss
    cache.set("key3", 3)  # eviction of key2 (LRU victim), now cache has key1 and key3
    cache.invalidate("key1")  # delete key1, now cache has only key3

    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.sets == 3
    assert stats.deletes == 1
    assert stats.evictions == 1
    assert stats.entry_count == 1  # Only key3 remains


def test_memory_backend():
    """测试内存后端。"""
    backend = MemoryCacheBackend[str]()

    entry1 = CacheEntry(
        key="key1",
        value="value1",
        size_bytes=100,
        created_at=time.monotonic(),
        last_accessed=time.monotonic(),
        access_count=0,
    )

    backend.set(entry1)
    assert backend.get("key1") == entry1
    assert len(backend) == 1

    assert backend.delete("key1") is True
    assert backend.get("key1") is None
    assert len(backend) == 0


def test_max_bytes_budget():
    """测试字节预算限制。"""
    cache = UnifiedCache[str](
        max_bytes=1000,
        eviction_policy=EvictionPolicy.LRU,
        size_estimator=lambda x: len(x),
    )

    # 插入大字符串（每个约 100 字节）
    for i in range(20):
        cache.set(f"key_{i}", "x" * 100)

    # 由于字节限制，应该只保留约 10 个条目
    stats = cache.stats()
    assert stats.size_bytes <= 1000
    assert stats.evictions > 0


if __name__ == "__main__":
    test_basic_get_set()
    test_ttl_expiration()
    test_lru_eviction()
    test_lfu_eviction()
    test_pin_unpin()
    test_factory_function()
    test_pattern_clear()
    test_concurrent_access()
    test_invalidate()
    test_stats()
    test_memory_backend()
    test_max_bytes_budget()
    print("All tests passed!")
