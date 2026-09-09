"""
Tests for FactorEngine adapter.

Tests identity provider integration with mock FE expression system.
"""

import pytest
from unittest.mock import MagicMock, patch
import hashlib
import json

from factor_assets.adapters import OptionalDependencyMissing


# Mock FE expression types
class MockExpr:
    """Mock base Expr type."""
    pass


class MockCleanedCall(MockExpr):
    """Mock CleanedCall."""
    def __init__(self, op, args, kwargs=None):
        self.op = op
        self.args = args
        self.kwargs = kwargs or {}


class MockColumnRef(MockExpr):
    """Mock ColumnRef."""
    def __init__(self, name):
        self.name = name


class MockLiteral(MockExpr):
    """Mock Literal."""
    def __init__(self, value):
        self.value = value


def mock_expression_payload(node):
    """Mock expression_payload function."""
    if isinstance(node, MockColumnRef):
        return {"kind": "column", "name": node.name}
    if isinstance(node, MockLiteral):
        return {"kind": "literal", "value": node.value}
    if isinstance(node, MockCleanedCall):
        return {
            "kind": "call",
            "op": node.op,
            "args": [mock_expression_payload(child) for child in node.args],
            "kwargs": {str(k): v for k, v in sorted(node.kwargs.items())},
        }
    raise TypeError(f"Unsupported node: {type(node)}")


def mock_canonical_expression(node):
    """Mock canonical_expression function."""
    payload = mock_expression_payload(node)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# Module names that must be mocked so the REAL factor_engine.expr package is
# never imported while the adapter is (re)loaded. If the real package chain is
# pulled in while factor_engine.expr.base is a MagicMock, its @dataclass
# definitions blow up ("Mock object has no attribute '__mro__'") and the
# failure permanently poisons sys.modules for later tests.
_MOCKED_FE_MODULES = ("expr", "factor_engine.expr", "factor_engine.expr.base")
_MISSING = object()


def _install_fe_mocks(mock_expr):
    """Install hermetic FE mocks into sys.modules; return the prior state."""
    import sys
    saved = {}
    for name in _MOCKED_FE_MODULES:
        saved[name] = sys.modules.get(name, _MISSING)
    sys.modules["expr"] = mock_expr
    sys.modules["factor_engine.expr"] = mock_expr
    sys.modules["factor_engine.expr.base"] = mock_expr.base
    return saved


def _restore_fe_mocks(saved):
    """Restore sys.modules to its pre-test state so later tests stay clean."""
    import sys
    for name in _MOCKED_FE_MODULES:
        prior = saved.get(name)
        if prior is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior


class TestFEIdentityProvider:
    """Test FE identity provider."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock FE availability
        from unittest.mock import MagicMock

        # Create mock expr module (hermetic: the real factor_engine.expr
        # package must never be imported while the adapter is reloaded)
        mock_expr = MagicMock()
        mock_expr.Expr = MockExpr
        mock_expr.CleanedCall = MockCleanedCall
        mock_expr.ColumnRef = MockColumnRef
        mock_expr.Literal = MockLiteral
        mock_expr.canonical_expression = mock_canonical_expression
        mock_expr.expression_payload = mock_expression_payload
        mock_expr.base.ensure_expr = lambda x: x
        self._saved_fe_mocks = _install_fe_mocks(mock_expr)

        # Force reload
        import importlib
        import factor_assets.adapters.factor_engine
        importlib.reload(factor_assets.adapters.factor_engine)

    def teardown_method(self):
        """Restore sys.modules so no mock pollution leaks into other tests."""
        _restore_fe_mocks(getattr(self, "_saved_fe_mocks", {}))

    def test_get_canonical_hash_simple_column(self):
        """Test canonical hash for simple column reference."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr(compiler_generation="fe-test-1.0")

        # Create simple expression: column("close")
        expr = MockColumnRef("close")

        canonical_hash = provider.get_canonical_hash(expr)

        # Verify hash is deterministic
        expected_repr = '{"kind":"column","name":"close"}'
        expected_hash = hashlib.sha256(expected_repr.encode("utf-8")).hexdigest()
        assert canonical_hash == expected_hash
        assert len(canonical_hash) == 64  # SHA256 hex

    def test_get_canonical_hash_call_expression(self):
        """Test canonical hash for call expression."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr()

        # Create expression: ts_mean(close, 20)
        expr = MockCleanedCall(
            op="ts_mean",
            args=[MockColumnRef("close"), MockLiteral(20)],
        )

        canonical_hash = provider.get_canonical_hash(expr)

        # Verify hash is deterministic
        assert len(canonical_hash) == 64

        # Same expression should produce same hash
        expr2 = MockCleanedCall(
            op="ts_mean",
            args=[MockColumnRef("close"), MockLiteral(20)],
        )
        canonical_hash2 = provider.get_canonical_hash(expr2)
        assert canonical_hash == canonical_hash2

    def test_get_canonical_repr(self):
        """Test canonical representation."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr()

        expr = MockColumnRef("volume")
        canonical_repr = provider.get_canonical_repr(expr)

        assert canonical_repr == '{"kind":"column","name":"volume"}'

    def test_get_canonical_repr_nested(self):
        """Test canonical repr for nested expression."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr()

        # Create: ts_rank(ts_mean(close, 20), 60)
        inner = MockCleanedCall(
            op="ts_mean",
            args=[MockColumnRef("close"), MockLiteral(20)],
        )
        outer = MockCleanedCall(
            op="ts_rank",
            args=[inner, MockLiteral(60)],
        )

        canonical_repr = provider.get_canonical_repr(outer)

        # Verify JSON structure
        payload = json.loads(canonical_repr)
        assert payload["kind"] == "call"
        assert payload["op"] == "ts_rank"
        assert len(payload["args"]) == 2
        assert payload["args"][0]["op"] == "ts_mean"

    def test_get_identity_ref(self):
        """Test FE identity reference."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr()

        expr = MockColumnRef("close")
        identity_ref = provider.get_identity_ref(expr)

        # For now, identity ref is same as canonical hash
        canonical_hash = provider.get_canonical_hash(expr)
        assert identity_ref == canonical_hash

    def test_get_full_identity(self):
        """Test getting complete factor identity."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr(compiler_generation="fe-0.9.7")

        expr = MockCleanedCall(
            op="ts_mean",
            args=[MockColumnRef("close"), MockLiteral(20)],
        )

        identity = provider.get_full_identity(
            expression=expr,
            complexity_score=2.5,
        )

        # Verify identity fields
        assert identity.canonical_repr is not None
        assert identity.canonical_hash is not None
        assert identity.fe_identity_ref is not None
        assert identity.fe_compiler_generation == "fe-0.9.7"
        assert identity.complexity_score == 2.5

        # Verify hash length
        assert len(identity.canonical_hash) == 64

    def test_get_full_identity_no_complexity(self):
        """Test identity without complexity score."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr()

        expr = MockColumnRef("volume")
        identity = provider.get_full_identity(expression=expr)

        assert identity.complexity_score is None

    def test_validate_expression_valid(self):
        """Test expression validation for valid expression."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr()

        expr = MockColumnRef("close")
        assert provider.validate_expression(expr) is True

    def test_validate_expression_invalid(self):
        """Test expression validation for invalid expression."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr()

        # Pass non-Expr object
        assert provider.validate_expression("not-an-expr") is False

    def test_invalid_expression_type(self):
        """Test error for invalid expression type."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr()

        with pytest.raises(ValueError, match="requires Expr node"):
            provider.get_canonical_hash("string-expression")

    def test_compiler_generation_default(self):
        """Test default compiler generation."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr()
        assert provider.compiler_generation == "fe-0.9.x"

    def test_compiler_generation_custom(self):
        """Test custom compiler generation."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr(compiler_generation="fe-1.0.0")
        assert provider.compiler_generation == "fe-1.0.0"

    def test_deterministic_hashing(self):
        """Test that hashing is deterministic across multiple calls."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr()

        expr1 = MockCleanedCall(
            op="ts_rank",
            args=[MockColumnRef("close"), MockLiteral(20)],
            kwargs={"method": "average"},
        )

        hash1 = provider.get_canonical_hash(expr1)
        hash2 = provider.get_canonical_hash(expr1)
        hash3 = provider.get_canonical_hash(expr1)

        assert hash1 == hash2 == hash3

    def test_different_expressions_different_hashes(self):
        """Test that different expressions produce different hashes."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr()

        expr1 = MockCleanedCall(
            op="ts_mean",
            args=[MockColumnRef("close"), MockLiteral(20)],
        )

        expr2 = MockCleanedCall(
            op="ts_mean",
            args=[MockColumnRef("close"), MockLiteral(30)],  # Different window
        )

        expr3 = MockCleanedCall(
            op="ts_std",  # Different operator
            args=[MockColumnRef("close"), MockLiteral(20)],
        )

        hash1 = provider.get_canonical_hash(expr1)
        hash2 = provider.get_canonical_hash(expr2)
        hash3 = provider.get_canonical_hash(expr3)

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_kwargs_ordering_normalized(self):
        """Test that kwargs order doesn't affect hash."""
        from factor_assets.adapters.factor_engine import FEIdentityProviderFromExpr

        provider = FEIdentityProviderFromExpr()

        # Create two expressions with same kwargs but different order
        # Note: In Python 3.7+, dicts maintain insertion order, but our
        # canonical repr sorts keys, so order shouldn't matter

        expr1 = MockCleanedCall(
            op="ts_rank",
            args=[MockColumnRef("close")],
            kwargs={"method": "average", "ascending": True},
        )

        expr2 = MockCleanedCall(
            op="ts_rank",
            args=[MockColumnRef("close")],
            kwargs={"ascending": True, "method": "average"},
        )

        hash1 = provider.get_canonical_hash(expr1)
        hash2 = provider.get_canonical_hash(expr2)

        assert hash1 == hash2


class TestFEIdentityProviderStringParsing:
    """Verify delegation without importing FE math into an optional unit test."""

    def test_string_parsing_delegates_to_fe_parser(self, monkeypatch):
        from unittest.mock import MagicMock
        from types import SimpleNamespace
        import sys

        # Mock FE (hermetic — never import the real factor_engine.expr)
        mock_expr = MagicMock()
        mock_expr.Expr = MockExpr
        mock_expr.canonical_expression = mock_canonical_expression
        mock_expr.expression_payload = mock_expression_payload
        mock_expr.base.ensure_expr = lambda x: x
        saved = _install_fe_mocks(mock_expr)
        calls = []
        parsed = MockCleanedCall('ts_mean', [MockColumnRef('close'), MockLiteral(20)])
        def parse(text, **kwargs):
            calls.append((text, kwargs))
            return parsed
        monkeypatch.setitem(sys.modules, 'factor_engine.api.dsl_parser', SimpleNamespace(parse_expr=parse))
        try:
            import importlib
            import factor_assets.adapters.factor_engine
            importlib.reload(factor_assets.adapters.factor_engine)

            from factor_assets.adapters.factor_engine import FEIdentityProvider

            provider = FEIdentityProvider()

            assert provider.get_canonical_hash("ts_mean(close, 20)") == provider.get_canonical_hash(parsed)
            assert calls == [('ts_mean(close, 20)', {'surface': 'daily', 'dialect': 'native', 'dialect_version': None})]
        finally:
            _restore_fe_mocks(saved)


class TestOptionalDependency:
    """Test optional dependency handling."""

    def test_missing_fe_raises_error(self):
        """Test that missing FE raises OptionalDependencyMissing."""
        import sys
        import factor_assets.adapters.factor_engine as fe_mod

        # This test does not mock the FE modules, so force a reload to undo
        # any mocked-state left by earlier tests, then restore sys.modules.
        saved = {}
        for name in _MOCKED_FE_MODULES:
            saved[name] = sys.modules.get(name, _MISSING)
            sys.modules.pop(name, None)
        try:
            import importlib
            importlib.reload(fe_mod)

            # Directly patch FE_AVAILABLE to simulate missing dependency
            with patch.object(fe_mod, 'FE_AVAILABLE', False):
                from factor_assets.adapters.factor_engine import FEIdentityProvider

                with pytest.raises(OptionalDependencyMissing) as exc_info:
                    FEIdentityProvider()

                assert exc_info.value.package_name == "factor_engine"
                assert exc_info.value.adapter_name == "FEIdentityProvider"
        finally:
            import factor_assets.adapters.factor_engine as fe_final
            importlib.reload(fe_final)
            for name, prior in saved.items():
                if prior is _MISSING:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prior
