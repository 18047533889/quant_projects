# Documentation Guide for Quant Projects

This guide explains the documentation setup for the Quant Projects platform.

## Overview

The documentation is built using **Sphinx**, a powerful Python documentation generator that creates professional API documentation from docstrings and reStructuredText files.

## What Has Been Set Up

### 1. Sphinx Configuration (`source/conf.py`)

Complete Sphinx configuration with:
- Auto-generation from docstrings (autodoc)
- Google and NumPy style docstring support (napoleon)
- Type hints in documentation (sphinx-autodoc-typehints)
- Markdown support (myst-parser)
- Read the Docs theme
- Cross-references to Python, NumPy, Pandas, Polars docs
- Custom CSS styling

### 2. API Reference Documentation (`source/api/`)

Auto-generated API documentation for all modules:
- `dataaccess.rst` - Data access layer
- `factor_engine.rst` - Factor computation engine
- `factor_preprocess.rst` - Preprocessing utilities
- `factor_optimizer.rst` - Portfolio optimization
- `factor_assets.rst` - Asset universe management
- `quant_evaluator.rst` - Factor evaluation and backtesting
- `research_control.rst` - Experiment tracking

### 3. User Documentation (`source/`)

Complete user guides:
- `introduction.rst` - Platform overview and architecture
- `installation.rst` - Installation and setup
- `quickstart.rst` - Getting started guide
- `usage_examples.rst` - Detailed usage examples
- `contributing.rst` - Contribution guidelines
- `changelog.rst` - Version history

### 4. Build System

- `Makefile` - Build commands for all formats (HTML, PDF, ePub)
- `build_docs.sh` - Automated build script
- `requirements.txt` - Documentation dependencies

### 5. ReadTheDocs Integration

- `.readthedocs.yaml` - Configuration for automatic builds
- Configured to build on every commit to main branch
- Supports HTML, PDF, and ePub formats

### 6. Contributing Guide

- `CONTRIBUTING.md` - Comprehensive contribution guidelines
- Code standards and style guide
- Testing requirements
- Documentation writing guidelines
- Pull request process

### 7. Custom Styling

- `source/_static/custom.css` - Custom CSS for better appearance
- Enhanced code block styling
- Better table formatting
- Improved admonition boxes

## Quick Start

### Building Documentation

```bash
# Navigate to docs directory
cd docs

# Build HTML documentation
make html

# View in browser
open build/html/index.html
```

Or use the build script:

```bash
cd docs
./build_docs.sh
```

### Serving Documentation Locally

```bash
cd docs/build/html
python3 -m http.server 8000
# Open http://localhost:8000 in browser
```

### Cleaning Build

```bash
cd docs
make clean
```

## Documentation Structure

```
docs/
├── source/                      # Source files
│   ├── conf.py                 # Sphinx configuration
│   ├── index.rst               # Main index
│   ├── introduction.rst        # Platform intro
│   ├── installation.rst        # Setup guide
│   ├── quickstart.rst          # Quick start
│   ├── usage_examples.rst      # Detailed examples
│   ├── contributing.rst        # Contributing guide
│   ├── changelog.rst           # Version history
│   ├── api/                    # API reference
│   │   ├── modules.rst         # Module overview
│   │   ├── dataaccess.rst      # DataAccess API
│   │   ├── factor_engine.rst   # FactorEngine API
│   │   ├── factor_preprocess.rst
│   │   ├── factor_optimizer.rst
│   │   ├── factor_assets.rst
│   │   ├── quant_evaluator.rst
│   │   └── research_control.rst
│   ├── _static/                # Static files
│   │   └── custom.css          # Custom CSS
│   └── _templates/             # Custom templates
├── build/                       # Generated docs (git-ignored)
│   └── html/                   # HTML output
├── Makefile                     # Build commands
├── build_docs.sh                # Build script
├── requirements.txt             # Doc dependencies
└── README.md                    # This file
```

## Writing Documentation

### Adding New Pages

1. Create a new `.rst` file in `source/`:

```rst
My New Page
===========

Content goes here.

Section
-------

More content.
```

2. Add it to `index.rst`:

```rst
.. toctree::
   :maxdepth: 2

   introduction
   my_new_page
```

3. Rebuild:

```bash
make html
```

### Documenting Code

Use Google-style docstrings:

```python
def compute_factor(
    data: pd.DataFrame,
    period: int = 20,
    *,
    method: Literal["simple", "log"] = "simple"
) -> pd.DataFrame:
    """Compute a factor from price data.

    This function computes a momentum factor using either simple
    or logarithmic returns over a specified lookback period.

    Args:
        data: DataFrame containing price data with 'close' column
        period: Lookback period in days (must be >= 1)
        method: Calculation method - 'simple' or 'log' returns

    Returns:
        DataFrame with factor values, same shape as input

    Raises:
        ValueError: If period < 1
        KeyError: If data missing 'close' column

    Example:
        >>> prices = pd.DataFrame({'close': [100, 105, 110, 115, 120]})
        >>> momentum = compute_factor(prices, period=2)
        >>> print(momentum.tail(1))
        close    0.047619
        dtype: float64

    Note:
        Missing values at the beginning are filled with NaN
        due to insufficient history.

    See Also:
        compute_returns: For simple return calculations
        compute_log_returns: For logarithmic returns
    """
    pass
```

### Code Examples in RST

```rst
Basic Usage
-----------

Here's how to compute a momentum factor:

.. code-block:: python

   from factor_engine import FactorEngine
   import pandas as pd

   # Create engine
   engine = FactorEngine()

   # Load data
   prices = pd.read_parquet("prices.parquet")

   # Compute factor
   momentum = engine.compute('ts_returns', prices, {'period': 20})

The result is a DataFrame with the same shape as the input.
```

### Cross-References

```rst
See :class:`~factor_engine.FactorEngine` for the main engine.

Use :func:`~factor_preprocess.winsorize` to remove outliers.

Refer to :ref:`installation` for setup instructions.
```

## Maintenance

### Updating After Code Changes

1. Update docstrings in code
2. Rebuild documentation:

```bash
cd docs
make clean
make html
```

3. Review changes in browser

### Adding New Modules

1. Create API documentation file in `source/api/`:

```rst
New Module
==========

.. automodule:: new_module
   :members:
   :undoc-members:
   :show-inheritance:
```

2. Add to `source/api/modules.rst`

3. Rebuild documentation

### Updating Changelog

Edit `source/changelog.rst`:

```rst
[1.1.0] - 2026-08-15
--------------------

Added
~~~~~

* New momentum operator with decay
* Support for intraday data

Fixed
~~~~~

* Fixed memory leak in cache manager
```

## ReadTheDocs Setup

### Initial Setup

1. Go to https://readthedocs.org
2. Sign in with GitHub
3. Import your repository
4. Configuration is auto-detected from `.readthedocs.yaml`

### Automatic Builds

Documentation builds automatically on:
- Every push to main branch
- Every pull request (preview)

### Viewing Builds

- Production: `https://your-project.readthedocs.io/`
- Latest: `https://your-project.readthedocs.io/en/latest/`
- Specific version: `https://your-project.readthedocs.io/en/v1.0.0/`

## Troubleshooting

### Import Errors During Build

If autodoc fails to import modules:

```bash
# Ensure modules are installed
pip install -e ./dataaccess
pip install -e ./factor_engine
# ... etc

# Test imports
python3 -c "import dataaccess; import factor_engine"
```

### Missing Dependencies

```bash
# Install all documentation dependencies
pip install -r docs/requirements.txt
```

### Sphinx Warnings

Build shows 135 warnings - these are mostly from:
- Missing docstrings (expected for internal code)
- Cross-reference targets not found (harmless)
- Duplicate object descriptions (from multiple imports)

To see specific warnings:

```bash
cd docs
make html 2>&1 | grep -i warning | head -20
```

### PDF Build Fails

PDF generation requires LaTeX. Install it:

```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-base texlive-latex-extra latexmk

# macOS
brew install --cask mactex
```

Then rebuild:

```bash
cd docs
make latexpdf
```

## Best Practices

### Documentation Standards

1. **Always document public APIs** - Every public function, class, method
2. **Include examples** - Especially for complex features
3. **Document exceptions** - What errors can be raised and why
4. **Keep it current** - Update docs when code changes
5. **Test examples** - Ensure code examples actually work

### Docstring Guidelines

- Use Google-style docstrings consistently
- Include type hints in function signatures (not docstrings)
- Provide realistic examples that users can run
- Document edge cases and gotchas
- Link to related functions

### RST Writing Tips

- Use clear section hierarchies (=, -, ~, ^)
- Keep sentences concise
- Use code blocks for all code
- Include cross-references liberally
- Use admonitions (note, warning, tip) appropriately

## Generated Files

The build generates:

- **HTML**: `build/html/` - Web documentation (25MB)
- **LaTeX**: `build/latex/` - LaTeX source for PDF
- **ePub**: `build/epub/` - E-book format
- **Search index**: Automatically generated for HTML

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Google Style Python Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [ReadTheDocs Guide](https://docs.readthedocs.io/)

## Support

For documentation issues:
- Check build logs for errors
- Review Sphinx documentation
- Open an issue with `documentation` label
- Include error messages and steps to reproduce

---

**Documentation is up to date as of:** 2026-08-14  
**Built with:** Sphinx 8.0+, sphinx-rtd-theme 2.0+
