"""Deep immutability helper (package-local; not a shared common/utils)."""
from collections.abc import Mapping


class _FrozenDict(Mapping):
    """Pickleable, hash-compatible deep-frozen mapping.

    Unlike ``MappingProxyType``, this class supports ``copy.deepcopy`` and
    pickling, so it is safe inside containers that are themselves deep-copied
    (e.g. ``PolicyRegistry.register`` uses ``deepcopy`` to hand out isolated
    snapshots). It is deeply immutable: every key/value is recursively frozen.
    """

    __slots__ = ("_items", "_values")

    def __init__(self, value):
        self._items = tuple((_deep_freeze(k), _deep_freeze(v)) for k, v in value.items())
        self._values = dict(self._items)

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __eq__(self, other):
        if isinstance(other, Mapping):
            return dict(self._values) == dict(other)
        return NotImplemented

    def __hash__(self):
        return hash(self._items)

    def __repr__(self):
        return f"FrozenDict({self._values!r})"

    def __reduce__(self):
        # Support pickling / deepcopy: rebuild from a plain dict.
        return (_FrozenDict, (self._values,))


def _deep_freeze(value):
    """Recursively freeze a nested container into an immutable form.

    dict -> ``_FrozenDict``, list/tuple -> tuple, set/frozenset -> frozenset.
    Scalars are returned unchanged. This guarantees that a ``frozen=True``
    dataclass is also *deeply* immutable (its nested containers cannot be
    mutated in place), which is a hard requirement before such objects may be
    hashed or used as identity keys (DLIB-FP-026).
    """
    if isinstance(value, _FrozenDict):
        return value
    if isinstance(value, Mapping):
        return _FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(v) for v in value)
    return value


def as_plain(value):
    """Convert a deeply-frozen value back to plain dict/list/set (debugging)."""
    if isinstance(value, _FrozenDict):
        return {as_plain(k): as_plain(v) for k, v in value.items()}
    if isinstance(value, Mapping):
        return {as_plain(k): as_plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(as_plain(v) for v in value)
    if isinstance(value, frozenset):
        return frozenset(as_plain(v) for v in value)
    return value


def deep_freeze(value):
    """Public entry point that deep-freezes a container."""
    return _deep_freeze(value)
