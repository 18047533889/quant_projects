"""R21-STABLE-VALUE-COLLISION regression tests.

``_stable_value`` previously serialized Mapping keys via ``str(key)``, so the
int ``1`` and the str ``"1"`` (and ``1.0``) collapsed onto the same key. That
silently merged semantically distinct candidates in the correctness-identity
path. Keys are now type-tagged (``int:1`` vs ``str:1`` vs ``float:1.0``).
"""
from __future__ import annotations

import pytest

from mining.campaign import _stable_key, _stable_value, candidate_semantic_hash


def test_stable_key_type_tags_int_vs_str():
    """int 1 and str "1" must serialize to distinct tagged keys."""
    assert _stable_key(1) == "int:1"
    assert _stable_key("1") == "str:1"
    assert _stable_key(1) != _stable_key("1")


def test_stable_key_distinguishes_int_float_bool_none():
    """Numeric/boolean/None keys must not collide with each other."""
    keys = {_stable_key(1), _stable_key(1.0), _stable_key(True), _stable_key(None)}
    assert len(keys) == 4
    assert _stable_key(1) != _stable_key(1.0)
    assert _stable_key(1) != _stable_key(True)
    assert _stable_key(True) != _stable_key("True")


def test_stable_value_mapping_preserves_int_vs_str_keys():
    """A Mapping with both 1 and "1" must keep both entries distinct."""
    value = {1: "a", "1": "b"}
    stable = _stable_value(value)
    assert stable == {"int:1": "a", "str:1": "b"}
    assert len(stable) == 2


def test_candidate_semantic_hash_distinguishes_int_vs_str_kwargs():
    """Candidates differing only in int-vs-str kwarg keys must hash differently."""
    from expr.cleaned_call import CleanedCall
    from expr.column import ColumnRef

    close = ColumnRef("close")
    volume = ColumnRef("volume")

    int_key = CleanedCall("add", (close, volume), (("params", {1: "a"}),))
    str_key = CleanedCall("add", (close, volume), (("params", {"1": "a"}),))

    assert candidate_semantic_hash(int_key) != candidate_semantic_hash(str_key)
