"""测试缓存装饰器。

R43 缓存装饰器测试：
    - @cached 基本功能
    - TTL 过期
    - 自定义 key 函数
    - 条件缓存（predicate）
    - @cached_property
    - cache_info/cache_clear
"""
import time
from cache.cache_decorator import cached, cached_property, memoize


def test_cached_decorator_basic():
    """测试 @cached 基本功能。"""
    call_count = [0]

    @cached(maxsize=10)
    def expensive_func(x: int) -> int:
        call_count[0] += 1
        return x * x

    # 第一次调用：计算
    result1 = expensive_func(5)
    assert result1 == 25
    assert call_count[0] == 1

    # 第二次调用：从缓存
    result2 = expensive_func(5)
    assert result2 == 25
    assert call_count[0] == 1  # 没有增加

    # 不同参数：计算
    result3 = expensive_func(10)
    assert result3 == 100
    assert call_count[0] == 2

    # 检查统计
    info = expensive_func.cache_info()
    assert info["hits"] == 1
    assert info["misses"] == 2


def test_cached_ttl():
    """测试 TTL 过期。"""
    call_count = [0]

    @cached(ttl=0.1)
    def func_with_ttl(x: int) -> int:
        call_count[0] += 1
        return x + 100

    result1 = func_with_ttl(1)
    assert result1 == 101
    assert call_count[0] == 1

    # 未过期：从缓存
    result2 = func_with_ttl(1)
    assert result2 == 101
    assert call_count[0] == 1

    # 等待过期
    time.sleep(0.15)

    # 过期：重新计算
    result3 = func_with_ttl(1)
    assert result3 == 101
    assert call_count[0] == 2


def test_cached_custom_key():
    """测试自定义 key 函数。"""
    call_count = [0]

    @cached(key_func=lambda obj, field: f"{obj['id']}:{field}")
    def get_field(obj: dict, field: str) -> str:
        call_count[0] += 1
        return obj[field]

    obj1 = {"id": 1, "name": "Alice", "age": 30}
    obj2 = {"id": 2, "name": "Bob", "age": 25}

    # 第一次：计算
    name1 = get_field(obj1, "name")
    assert name1 == "Alice"
    assert call_count[0] == 1

    # 相同对象和字段：从缓存
    name1_cached = get_field(obj1, "name")
    assert name1_cached == "Alice"
    assert call_count[0] == 1

    # 不同对象：计算
    name2 = get_field(obj2, "name")
    assert name2 == "Bob"
    assert call_count[0] == 2


def test_cached_predicate():
    """测试条件缓存。"""
    call_count = [0]

    @cached(predicate=lambda result: result is not None)
    def fetch_data(key: str) -> str | None:
        call_count[0] += 1
        if key == "valid":
            return "data"
        return None

    # 第一次：计算（返回 None，不缓存）
    result1 = fetch_data("invalid")
    assert result1 is None
    assert call_count[0] == 1

    # 第二次：仍需计算（未缓存）
    result2 = fetch_data("invalid")
    assert result2 is None
    assert call_count[0] == 2

    # 有效结果：缓存
    result3 = fetch_data("valid")
    assert result3 == "data"
    assert call_count[0] == 3

    # 从缓存读取
    result4 = fetch_data("valid")
    assert result4 == "data"
    assert call_count[0] == 3


def test_cache_clear():
    """测试 cache_clear。"""
    call_count = [0]

    @cached()
    def func(x: int) -> int:
        call_count[0] += 1
        return x * 2

    func(5)
    func(5)
    assert call_count[0] == 1

    # 清空缓存
    func.cache_clear()

    # 重新计算
    func(5)
    assert call_count[0] == 2


def test_cache_invalidate():
    """测试 cache_invalidate。"""
    call_count = [0]

    @cached()
    def func(x: int, y: int) -> int:
        call_count[0] += 1
        return x + y

    func(1, 2)
    func(1, 2)
    assert call_count[0] == 1

    # 失效特定参数
    func.cache_invalidate(1, 2)

    # 重新计算
    func(1, 2)
    assert call_count[0] == 2


def test_cached_property_decorator():
    """测试 @cached_property。"""
    call_count = [0]

    class MyClass:
        def __init__(self, value: int):
            self._value = value

        @cached_property(ttl=0.2)
        def expensive_property(self) -> int:
            call_count[0] += 1
            return self._value * 100

    obj = MyClass(5)

    # 第一次访问：计算
    result1 = obj.expensive_property
    assert result1 == 500
    assert call_count[0] == 1

    # 第二次访问：从缓存
    result2 = obj.expensive_property
    assert result2 == 500
    assert call_count[0] == 1

    # 等待 TTL 过期
    time.sleep(0.25)

    # 过期：重新计算
    result3 = obj.expensive_property
    assert result3 == 500
    assert call_count[0] == 2


def test_memoize():
    """测试 @memoize 装饰器。"""
    call_count = [0]

    @memoize
    def fibonacci(n: int) -> int:
        call_count[0] += 1
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    result = fibonacci(10)
    assert result == 55
    # 由于记忆化，调用次数远少于递归次数
    assert call_count[0] == 11  # 只计算 0-10 各一次


def test_typed_cache():
    """测试 typed=True（区分参数类型）。"""
    call_count = [0]

    @cached(typed=True)
    def func(x):
        call_count[0] += 1
        return x + 100

    # int(1)
    func(1)
    assert call_count[0] == 1

    # float(1.0) 应该被视为不同键
    func(1.0)
    assert call_count[0] == 2

    # 再次 int(1)：从缓存
    func(1)
    assert call_count[0] == 2


def test_namespace():
    """测试命名空间隔离。"""
    call_count = [0]

    @cached(namespace="ns1")
    def func1(x: int) -> int:
        call_count[0] += 1
        return x * 2

    @cached(namespace="ns2")
    def func2(x: int) -> int:
        call_count[0] += 1
        return x * 3

    # 两个函数使用不同命名空间，缓存独立
    func1(5)
    func2(5)
    assert call_count[0] == 2

    # 各自从缓存读取
    func1(5)
    func2(5)
    assert call_count[0] == 2


if __name__ == "__main__":
    test_cached_decorator_basic()
    test_cached_ttl()
    test_cached_custom_key()
    test_cached_predicate()
    test_cache_clear()
    test_cache_invalidate()
    test_cached_property_decorator()
    test_memoize()
    test_typed_cache()
    test_namespace()
    print("All decorator tests passed!")
