# Contributing to Quant Projects

Thank you for your interest in contributing to the Quant Projects platform. This document provides guidelines for contributing code, documentation, and reporting issues.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Standards](#testing-standards)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

- Be respectful and inclusive to all contributors
- Focus on constructive feedback
- Help maintain a welcoming environment
- Report unacceptable behavior to maintainers

## Getting Started

### Development Setup

1. **Fork and Clone**

```bash
git clone https://github.com/yourusername/quant_projects.git
cd quant_projects
```

2. **Create Virtual Environment**

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install in Development Mode**

```bash
# Install all modules
pip install -e ./dataaccess
pip install -e ./factor_engine
pip install -e ./factor_preprocess
pip install -e ./factor_optimizer
pip install -e ./factor_assets
pip install -e ./quant_evaluator
pip install -e ./research_control
```

4. **Install Development Dependencies**

```bash
pip install pytest pytest-cov black isort mypy flake8 sphinx
```

5. **Verify Installation**

```bash
pytest --version
black --version
make -C docs html
```

## Development Workflow

### Branching Strategy

- `main` - Production-ready, stable code
- `develop` - Integration branch for features
- `feature/xxx` - New features
- `fix/xxx` - Bug fixes
- `docs/xxx` - Documentation updates
- `refactor/xxx` - Code refactoring

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Write Code**
   - Follow code standards (see below)
   - Add type hints
   - Write docstrings

2. **Write Tests**
   - Add unit tests for new functionality
   - Ensure edge cases are covered
   - Maintain 80%+ coverage

3. **Run Tests**

```bash
pytest
pytest --cov=your_module --cov-report=html
```

4. **Format Code**

```bash
black .
isort .
```

5. **Lint Code**

```bash
flake8 .
mypy your_module
```

### Committing Changes

Use conventional commit messages:

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `style:` - Code formatting (no logic change)
- `refactor:` - Code restructuring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

**Example:**

```bash
git commit -m "feat: add exponential decay to momentum operator

- Implement decay_rate parameter
- Add validation for decay_rate range
- Include tests for edge cases
- Update documentation

Closes #123"
```

## Code Standards

### Python Style

Follow **PEP 8** with these project-specific rules:

- **Line length:** 120 characters
- **Imports:** Use `isort` with black profile
- **Type hints:** Required for all public functions
- **Docstrings:** Required for all public APIs

### Code Example

```python
from typing import Literal
import pandas as pd
import numpy as np


def compute_momentum(
    prices: pd.DataFrame,
    period: int = 20,
    *,
    method: Literal["simple", "log"] = "simple",
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute momentum factor.

    Args:
        prices: DataFrame with price data (must contain 'close' column)
        period: Lookback period in days (must be >= 1)
        method: Calculation method - 'simple' or 'log' returns
        min_periods: Minimum observations required (defaults to period)

    Returns:
        DataFrame with momentum values

    Raises:
        ValueError: If period < 1
        KeyError: If prices missing 'close' column

    Example:
        >>> prices = pd.DataFrame({'close': [100, 105, 110, 115, 120]})
        >>> momentum = compute_momentum(prices, period=2)
        >>> print(momentum)
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    if "close" not in prices.columns:
        raise KeyError("prices must contain 'close' column")

    if min_periods is None:
        min_periods = period

    if method == "simple":
        return prices["close"].pct_change(period)
    elif method == "log":
        return np.log(prices["close"] / prices["close"].shift(period))
    else:
        raise ValueError(f"Invalid method: {method}")
```

### Type Hints

Use comprehensive type hints:

```python
from typing import Protocol, TypeVar, Generic, Literal
from collections.abc import Callable, Iterator

T = TypeVar("T")
Backend = Literal["pandas", "polars", "duckdb"]


class Operator(Protocol):
    """Operator protocol."""

    def compute(self, data: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """Compute operator result."""
        ...


class Cache(Generic[T]):
    """Generic cache implementation."""

    def get(self, key: str) -> T | None:
        """Get cached value."""
        ...

    def set(self, key: str, value: T) -> None:
        """Set cached value."""
        ...
```

## Testing Standards

### Test Structure

```python
import pytest
import pandas as pd
from factor_engine import compute_momentum


class TestMomentum:
    """Test suite for momentum computation."""

    @pytest.fixture
    def sample_prices(self) -> pd.DataFrame:
        """Sample price data for testing."""
        return pd.DataFrame({
            "close": [100, 105, 110, 115, 120]
        })

    def test_basic_momentum(self, sample_prices):
        """Test basic momentum calculation."""
        result = compute_momentum(sample_prices, period=1)
        expected = pd.Series([np.nan, 0.05, 0.0476, 0.0455, 0.0435])
        pd.testing.assert_series_equal(result["close"], expected, atol=1e-4)

    def test_invalid_period_raises(self):
        """Test that invalid period raises ValueError."""
        prices = pd.DataFrame({"close": [100, 105, 110]})
        with pytest.raises(ValueError, match="period must be >= 1"):
            compute_momentum(prices, period=0)

    @pytest.mark.parametrize("period", [1, 5, 20, 60])
    def test_various_periods(self, sample_prices, period):
        """Test momentum with various periods."""
        result = compute_momentum(sample_prices, period=period)
        assert result.shape == sample_prices.shape

    def test_missing_column_raises(self):
        """Test that missing column raises KeyError."""
        prices = pd.DataFrame({"price": [100, 105, 110]})
        with pytest.raises(KeyError, match="close"):
            compute_momentum(prices, period=1)
```

### Test Coverage

- Maintain **80%+ coverage** for new code
- Test edge cases and error conditions
- Include integration tests for complex features
- Add performance tests for critical paths

### Running Tests

```bash
# Run all tests
pytest

# Run specific module
pytest factor_engine/tests

# Run with coverage
pytest --cov=factor_engine --cov-report=html --cov-report=term

# Run specific test
pytest factor_engine/tests/test_operators.py::TestMomentum::test_basic_momentum

# Run with verbose output
pytest -v

# Run and stop on first failure
pytest -x
```

## Documentation Standards

### Docstring Format

Use **Google-style** docstrings:

```python
def example_function(arg1: str, arg2: int, arg3: float = 1.0) -> bool:
    """One-line summary of the function.

    More detailed description of what the function does.
    Can span multiple lines and include implementation details.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2 with more details
            Can span multiple lines if needed
        arg3: Optional argument with default value

    Returns:
        Description of the return value

    Raises:
        ValueError: When arg2 is negative
        TypeError: When arg1 is not a string

    Example:
        >>> result = example_function("test", 42, arg3=2.0)
        >>> print(result)
        True

    Note:
        Additional notes, warnings, or implementation details

    See Also:
        related_function: Related functionality
    """
    pass
```

### Building Documentation

```bash
cd docs
make clean
make html
# View: open build/html/index.html
```

### Documentation Updates

When adding new features:

1. Update API reference documentation
2. Add usage examples
3. Update quickstart guide if applicable
4. Update changelog

## Pull Request Process

### Before Submitting

Checklist:

- [ ] All tests pass locally
- [ ] Code follows style guide (black, isort, flake8)
- [ ] Type hints added and checked (mypy)
- [ ] Docstrings added/updated
- [ ] Tests added/updated (80%+ coverage)
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] No unrelated changes included
- [ ] Commit messages follow convention

### Submitting PR

1. **Push to your fork**

```bash
git push origin feature/your-feature-name
```

2. **Create Pull Request**
   - Use a clear, descriptive title
   - Reference related issues (`Closes #123`, `Fixes #456`)
   - Describe what changed and why
   - Include testing instructions
   - Add screenshots if UI changes

3. **PR Description Template**

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2
- Change 3

## Testing
How was this tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] All CI checks pass

## Related Issues
Closes #123
```

### Review Process

- Maintainers will review within 3-5 business days
- Address feedback promptly
- Keep discussions professional and constructive
- Update PR based on feedback
- Once approved, maintainer will merge

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
Clear description of what the bug is

**To Reproduce**
Steps to reproduce:
1. Load data with '...'
2. Compute factor using '...'
3. See error

**Expected behavior**
What you expected to happen

**Actual behavior**
What actually happened

**Code to reproduce**
```python
# Minimal reproducible example
from factor_engine import compute_momentum
...
```

**Environment:**
- OS: Ubuntu 20.04
- Python: 3.10.12
- Package versions: pandas 2.0.3, numpy 1.24.3
- Module version: factor_engine 1.0.0

**Additional context**
Any other relevant information
```

### Feature Requests

Use the feature request template:

```markdown
**Problem Statement**
What problem does this solve?

**Proposed Solution**
Describe your proposed solution

**Alternative Solutions**
Other solutions you've considered

**Use Case**
Example of how this would be used

**Willingness to Contribute**
Are you willing to implement this?
```

## Getting Help

- **Documentation:** https://your-docs-site.com
- **GitHub Issues:** For bugs and features
- **GitHub Discussions:** For questions and ideas

## License

By contributing, you agree that your contributions will be licensed under the project's license.

---

Thank you for contributing to Quant Projects!
