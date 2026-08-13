.. _contributing:

Contributing Guide
==================

We welcome contributions to the Quant Projects platform. This guide explains how to contribute effectively.

Code of Conduct
---------------

* Be respectful and inclusive
* Focus on constructive feedback
* Help maintain a welcoming environment
* Report unacceptable behavior to maintainers

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. Fork the repository
2. Clone your fork:

.. code-block:: bash

   git clone https://github.com/yourusername/quant_projects.git
   cd quant_projects

3. Create a virtual environment:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

4. Install in development mode:

.. code-block:: bash

   pip install -e ./dataaccess
   pip install -e ./factor_engine
   pip install -e ./factor_preprocess
   pip install -e ./factor_optimizer
   pip install -e ./factor_assets
   pip install -e ./quant_evaluator
   pip install -e ./research_control

5. Install development dependencies:

.. code-block:: bash

   pip install pytest pytest-cov black isort mypy flake8

Development Workflow
--------------------

Branching Strategy
~~~~~~~~~~~~~~~~~~

* ``main`` - production-ready code
* ``develop`` - integration branch
* ``feature/xxx`` - new features
* ``fix/xxx`` - bug fixes
* ``docs/xxx`` - documentation updates

Creating a Branch
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git checkout -b feature/your-feature-name develop

Making Changes
~~~~~~~~~~~~~~

1. Make your changes
2. Write or update tests
3. Ensure tests pass:

.. code-block:: bash

   pytest

4. Format code:

.. code-block:: bash

   black .
   isort .

5. Run linters:

.. code-block:: bash

   flake8 .
   mypy .

Committing Changes
~~~~~~~~~~~~~~~~~~

Write clear commit messages:

.. code-block:: bash

   git commit -m "feat: add momentum operator with decay parameter

   - Implement exponential decay weighting
   - Add parameter validation
   - Include unit tests for edge cases
   "

Use conventional commit prefixes:

* ``feat:`` - new feature
* ``fix:`` - bug fix
* ``docs:`` - documentation
* ``style:`` - formatting
* ``refactor:`` - code restructuring
* ``test:`` - adding tests
* ``chore:`` - maintenance

Submitting Changes
~~~~~~~~~~~~~~~~~~

1. Push to your fork:

.. code-block:: bash

   git push origin feature/your-feature-name

2. Create a pull request
3. Describe your changes clearly
4. Reference related issues
5. Wait for review

Code Standards
--------------

Python Style
~~~~~~~~~~~~

Follow PEP 8 with these specifics:

* Line length: 120 characters
* Use type hints for all functions
* Use docstrings for all public APIs
* Prefer explicit over implicit

Example:

.. code-block:: python

   def compute_momentum(
       prices: pd.DataFrame,
       period: int = 20,
       *,
       min_periods: int | None = None
   ) -> pd.DataFrame:
       """Compute momentum factor.

       Args:
           prices: DataFrame with price data
           period: Lookback period in days
           min_periods: Minimum observations required

       Returns:
           DataFrame with momentum values

       Raises:
           ValueError: If period < 1
       """
       if period < 1:
           raise ValueError("period must be >= 1")

       if min_periods is None:
           min_periods = period

       return prices.pct_change(period)

Documentation Style
~~~~~~~~~~~~~~~~~~~

Use Google-style docstrings:

.. code-block:: python

   def example_function(arg1: str, arg2: int) -> bool:
       """One-line summary.

       Detailed description of what the function does.
       Can span multiple lines.

       Args:
           arg1: Description of arg1
           arg2: Description of arg2

       Returns:
           Description of return value

       Raises:
           ValueError: When arg2 < 0
           TypeError: When arg1 is not string

       Example:
           >>> example_function("test", 42)
           True

       Note:
           Additional notes or warnings
       """
       pass

Testing Standards
~~~~~~~~~~~~~~~~~

Write comprehensive tests:

.. code-block:: python

   import pytest
   import pandas as pd
   from factor_engine import compute_momentum


   class TestMomentum:
       """Test suite for momentum computation."""

       def test_basic_momentum(self):
           """Test basic momentum calculation."""
           prices = pd.DataFrame({
               'close': [100, 105, 110, 115, 120]
           })
           result = compute_momentum(prices, period=1)
           assert result.iloc[-1] == 0.05  # 5% gain

       def test_momentum_with_missing_data(self):
           """Test momentum handles missing data."""
           prices = pd.DataFrame({
               'close': [100, None, 110, 115, 120]
           })
           result = compute_momentum(prices, period=2)
           assert result.notna().sum() >= 3

       def test_invalid_period_raises(self):
           """Test invalid period raises ValueError."""
           prices = pd.DataFrame({'close': [100, 105, 110]})
           with pytest.raises(ValueError, match="period must be >= 1"):
               compute_momentum(prices, period=0)

       @pytest.mark.parametrize("period", [1, 5, 20, 60])
       def test_various_periods(self, period):
           """Test momentum works for various periods."""
           prices = pd.DataFrame({
               'close': range(100, 200)
           })
           result = compute_momentum(prices, period=period)
           assert result.shape == prices.shape

Test Coverage
~~~~~~~~~~~~~

Maintain high test coverage:

* Minimum 80% coverage for new code
* Test edge cases and error conditions
* Include integration tests for complex features
* Add performance tests for critical paths

.. code-block:: bash

   # Check coverage
   pytest --cov=factor_engine --cov-report=html
   # View in browser: htmlcov/index.html

Type Hints
~~~~~~~~~~

Use comprehensive type hints:

.. code-block:: python

   from typing import Literal, Protocol
   import pandas as pd

   Backend = Literal['pandas', 'polars', 'duckdb']

   class Operator(Protocol):
       """Operator protocol."""

       def compute(self, data: pd.DataFrame) -> pd.DataFrame:
           """Compute operator result."""
           ...

   def create_engine(backend: Backend) -> FactorEngine:
       """Create factor engine with specified backend."""
       ...

Documentation Contributions
---------------------------

Improving Documentation
~~~~~~~~~~~~~~~~~~~~~~~

Documentation is as important as code:

* Fix typos and clarify confusing sections
* Add examples for complex features
* Update outdated information
* Improve API documentation

Building Docs Locally
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs
   make clean
   make html
   # Open build/html/index.html

Documentation Structure
~~~~~~~~~~~~~~~~~~~~~~~

* ``introduction.rst`` - Platform overview
* ``installation.rst`` - Setup instructions
* ``quickstart.rst`` - Quick start guide
* ``usage_examples.rst`` - Detailed examples
* ``api/`` - API reference documentation

Adding Examples
~~~~~~~~~~~~~~~

Include runnable examples:

.. code-block:: python

   """
   Example: Computing Multi-Factor Alpha
   ======================================

   This example shows how to combine multiple factors
   into a composite alpha signal.
   """

   from factor_engine import FactorEngine
   from factor_preprocess import neutralize, combine_factors

   # Initialize engine
   engine = FactorEngine()

   # Compute individual factors
   momentum = engine.compute('ts_returns', prices, {'period': 20})
   value = engine.compute('earnings_yield', fundamentals)
   quality = engine.compute('roe', fundamentals)

   # Combine factors
   alpha = combine_factors(
       [momentum, value, quality],
       weights=[0.4, 0.3, 0.3]
   )

Reporting Issues
----------------

Bug Reports
~~~~~~~~~~~

Include:

* Clear description of the bug
* Steps to reproduce
* Expected behavior
* Actual behavior
* Version information
* Minimal reproducible example

Example:

.. code-block:: text

   **Bug**: compute_momentum returns incorrect values for period=1

   **Steps to reproduce**:
   ```python
   import pandas as pd
   from factor_engine import compute_momentum

   prices = pd.DataFrame({'close': [100, 105, 110]})
   result = compute_momentum(prices, period=1)
   print(result)
   ```

   **Expected**: [NaN, 0.05, 0.0476...]
   **Actual**: [NaN, 0.0, 0.0]

   **Environment**:
   - Python 3.10.12
   - pandas 2.0.3
   - factor_engine 1.0.0

Feature Requests
~~~~~~~~~~~~~~~~

Describe:

* Use case for the feature
* Proposed API or interface
* Alternative solutions considered
* Willingness to implement

Review Process
--------------

Pull Request Reviews
~~~~~~~~~~~~~~~~~~~~

Reviewers will check:

* Code quality and style
* Test coverage
* Documentation updates
* Performance implications
* Breaking changes

Address feedback promptly and professionally.

Review Checklist
~~~~~~~~~~~~~~~~

Before requesting review:

- [ ] Tests pass locally
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] No unrelated changes
- [ ] Commit messages are clear

Merging
~~~~~~~

* Requires approval from maintainer
* All CI checks must pass
* Conflicts must be resolved
* Commits may be squashed

Release Process
---------------

Version Numbering
~~~~~~~~~~~~~~~~~

Follow semantic versioning (MAJOR.MINOR.PATCH):

* MAJOR: Breaking changes
* MINOR: New features (backward compatible)
* PATCH: Bug fixes

Release Checklist
~~~~~~~~~~~~~~~~~

1. Update version numbers
2. Update changelog
3. Run full test suite
4. Build documentation
5. Create release tag
6. Publish to package index

Getting Help
------------

* GitHub Issues - Bug reports and features
* Discussions - Questions and ideas
* Documentation - Usage guides
* Examples - Code samples

Thank you for contributing to Quant Projects!
