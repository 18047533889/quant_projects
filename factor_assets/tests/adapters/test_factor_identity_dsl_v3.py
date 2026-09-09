import pytest
import json

from factor_engine.api.dsl_parser import parse_expr
from factor_engine.expr.base import ensure_expr
from factor_assets.adapters.factor_engine import FEIdentityProvider


@pytest.mark.parametrize('text', ['close', 'rank(close)', 'ts_mean(close, 20)', 'close + open'])
def test_real_dsl_and_expression_tree_have_same_factor_identity(text):
    provider = FEIdentityProvider()
    assert provider.get_canonical_hash(text) == provider.get_canonical_hash(parse_expr(text))
    assert provider.validate_expression(text)


def test_bare_field_is_not_a_string_literal_identity():
    provider = FEIdentityProvider()
    # FE resolves the DSL alias to its canonical catalog FieldRef, which is
    # intentionally distinct from a raw unqualified col('close') reference.
    payload = json.loads(provider.get_canonical_repr('close'))
    assert payload['kind'] == 'field'
    assert payload['field_id']
    assert provider.get_canonical_hash('close') != provider.get_canonical_hash(ensure_expr('close'))


@pytest.mark.parametrize('text', ['', ' ', 'ts_mean(', '__import__("os").system("true")'])
def test_invalid_dsl_fails_at_identity_boundary(text):
    provider = FEIdentityProvider()
    assert not provider.validate_expression(text)
    with pytest.raises(ValueError):
        provider.get_canonical_hash(text)
