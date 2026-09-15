import polars as pl
import pytest

from factor_engine.backend.long_alignment import AlignmentError, assert_exact_key_set


def frame(times, assets):
    return pl.DataFrame({'ts': times, 'inst': assets, '_v': [1.] * len(times)},
        schema={'ts':pl.Int64,'inst':pl.String,'_v':pl.Float64}).lazy()


def test_exact_alignment_collects_each_subtree_once():
    calls = []
    def observed(value):
        calls.append(value.height)
        return value
    left = frame([1,2],['A','B']).map_batches(observed, projection_pushdown=False)
    right = frame([2,1],['B','A']).map_batches(observed, projection_pushdown=False)
    assert_exact_key_set(left,right)
    assert calls == [2,2]


def test_directional_missing_keys_remain_rejected():
    with pytest.raises(AlignmentError,match='missing_right=1.*missing_left=1'):
        assert_exact_key_set(frame([1,2],['A','B']),frame([1,3],['A','B']))


@pytest.mark.parametrize('reverse',[False,True])
def test_duplicate_keys_on_either_operand_remain_rejected(reverse):
    pair=[frame([1,1],['A','A']),frame([1],['A'])]
    if reverse: pair.reverse()
    with pytest.raises(AlignmentError,match='重复 key'):
        assert_exact_key_set(*pair)


def test_empty_and_shuffled_exact_sets():
    assert_exact_key_set(frame([],[]),frame([],[]))
    assert_exact_key_set(frame([1,2],['A','B']),frame([2,1],['B','A']))
