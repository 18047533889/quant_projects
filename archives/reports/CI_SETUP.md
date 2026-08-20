# CI/CD Setup Documentation

This document describes the continuous integration and continuous deployment (CI/CD) infrastructure for the quantitative platform monorepo.

## Overview

The CI/CD system consists of three main GitHub Actions workflows and comprehensive local development tooling:

- **ci.yml** - Continuous integration testing on every push/PR
- **benchmarks.yml** - Weekly performance benchmarks and regression tracking
- **quality.yml** - Code quality checks (linting, formatting, type checking, security)

## GitHub Actions Workflows

### 1. CI Workflow (`ci.yml`)

**Triggers:**
- Push to `main`, `master`, or `develop` branches
- Pull requests to these branches

**Jobs:**

#### test-matrix
Runs the full test suite across Python 3.10 and 3.11:
- DataAccess tests
- FactorEngine tests (with Numba disabled for CI)
- FactorPreprocess tests
- FactorOptimizer tests
- QuantEvaluator tests
- ResearchControl tests
- Integration tests

Tests marked as `slow` or `integration` are skipped in the matrix build for speed.

**Timeout:** 60 minutes

#### test-coverage
Generates test coverage reports:
- Runs on Python 3.11
- Uses pytest-cov to track coverage across all modules
- Uploads coverage to Codecov (optional)
- Generates both XML and terminal reports

**Timeout:** 60 minutes

#### pre-commit-checks
Validates code against pre-commit hooks:
- Runs all configured pre-commit hooks
- Ensures DataAccess allowlist compliance
- Checks formatting, imports, and common issues

**Timeout:** 10 minutes

### 2. Benchmarks Workflow (`benchmarks.yml`)

**Triggers:**
- Scheduled: Every Monday at 02:00 UTC
- Manual dispatch with quick mode option

**Jobs:**

#### benchmark
Runs comprehensive benchmark suite:
- Platform-wide benchmarks (QE, FP, FA, FO)
- DataAccess-specific benchmarks (COS, join, incremental)
- Generates JSON and text reports
- Stores historical baseline for main/master branch
- Posts summary comment on PRs
- Uploads artifacts with 90-day retention

**Timeout:** 120 minutes

#### dataaccess-perf
DataAccess performance regression gates:
- Runs performance-marked tests
- Validates no regression on critical paths
- Uploads results for tracking

**Timeout:** 40 minutes

#### factor-engine-perf
FactorEngine performance regression gates:
- Runs performance-marked tests
- Ensures operator performance stays within bounds
- Uploads results for tracking

**Timeout:** 40 minutes

### 3. Quality Workflow (`quality.yml`)

**Triggers:**
- Push to `main`, `master`, or `develop` branches
- Pull requests to these branches

**Jobs:**

#### black
Validates code formatting with Black:
- Line length: 120
- Target: Python 3.10+
- Checks all modules for consistent formatting

**Timeout:** 10 minutes

#### flake8
Linting with Flake8:
- Includes flake8-docstrings and flake8-bugbear plugins
- Enforces PEP 8 compliance
- Checks for common bugs and anti-patterns

**Timeout:** 10 minutes

#### mypy
Type checking with MyPy:
- Validates type annotations across all modules
- Runs per-module for better error isolation
- Continues on error to show all issues

**Timeout:** 15 minutes

#### pylint
Static analysis with Pylint:
- Deep code analysis for quality issues
- Checks for code smells and potential bugs
- Uses project-specific .pylintrc configuration

**Timeout:** 15 minutes

#### isort
Import sorting validation:
- Ensures consistent import ordering
- Black-compatible profile
- Line length: 120

**Timeout:** 10 minutes

#### security
Security scanning:
- Bandit for security vulnerabilities
- Safety for dependency vulnerabilities
- Uploads security reports as artifacts

**Timeout:** 10 minutes

#### complexity
Code complexity analysis:
- Radon cyclomatic complexity metrics
- Maintainability index calculation
- Generates reports for all modules

**Timeout:** 10 minutes

## Pre-commit Hooks

The `.pre-commit-config.yaml` file configures local git hooks that run before each commit.

**Enabled Hooks:**

1. **Black** - Auto-format Python code
2. **isort** - Sort imports
3. **Flake8** - Lint for errors and style issues
4. **Trailing whitespace** - Remove trailing whitespace
5. **End of file fixer** - Ensure files end with newline
6. **YAML/JSON/TOML checks** - Validate configuration files
7. **Large files check** - Prevent commits of files >5MB
8. **Merge conflict check** - Detect unresolved conflicts
9. **Debug statements** - Catch debug imports
10. **Bandit** - Security vulnerability scanning
11. **DataAccess allowlist** - Enforce DataAccess layer usage
12. **Quick pytest** - Run fast tests on pre-push (optional)

**Installation:**

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files  # First run on all files
```

## Makefile Commands

The root `Makefile` provides unified commands for local development.

### Installation

```bash
make install          # Install all dependencies
make install-dev      # Install dev dependencies + tools
```

### Testing

```bash
make test             # Run all tests
make test-quick       # Run fast tests only (skip slow/integration)
make test-coverage    # Generate coverage report (htmlcov/)
make test-integration # Run integration tests only
```

### Linting

```bash
make lint             # Run all linters (flake8, pylint, mypy)
make lint-flake8      # Run flake8 only
make lint-pylint      # Run pylint only
make lint-mypy        # Run mypy only
```

### Formatting

```bash
make format           # Auto-format with black and isort
make format-check     # Check formatting without changes
```

### Benchmarks

```bash
make benchmark        # Run comprehensive benchmark suite
make benchmark-quick  # Run quick benchmarks
```

### Audits

```bash
make audit                # Run security and complexity audits
make audit-security       # Run bandit + safety
make audit-complexity     # Run radon complexity analysis
make audit-factor-engine  # Run FactorEngine-specific audit
```

### Maintenance

```bash
make clean            # Remove build artifacts and caches
make clean-all        # Clean everything including venv
make pre-commit       # Install and run pre-commit hooks
make help             # Show all available commands
```

## Configuration Files

### `.flake8`
Flake8 configuration at repository root:
- Line length: 120
- Excludes: `.venv/`, `build/`, `dist/`, etc.
- Custom ignore rules for specific error codes

### `.pylintrc`
Pylint configuration at repository root:
- Module-specific settings
- Custom message suppression
- Code quality thresholds

### `pyproject.toml`
Black and isort configuration:
- Line length: 120
- Target version: Python 3.10+
- Black-compatible isort profile

### `pytest.ini`
Pytest markers definition:
- `integration` - Heavy tests with external dependencies
- `perf` - Performance regression tests
- `slow` - Long-running tests for nightly builds

### `.data_access_allowlist.yaml`
DataAccess layer enforcement:
- Lists allowed direct DuckDB/Parquet usage
- Enforced by pre-commit hook and CI
- Prevents bypassing the DataAccess abstraction layer

## Test Organization

Tests follow pytest conventions with markers for filtering:

```bash
# Run all tests
pytest

# Skip slow tests
pytest -m "not slow"

# Skip integration tests
pytest -m "not integration"

# Run only performance tests
pytest -m perf

# Run specific module
pytest dataaccess/tests/
```

## Benchmark Organization

Benchmarks are located in `benchmarks/` directory:

- `run_all_benchmarks.py` - Main benchmark runner
- `bench_qe_metrics.py` - QuantEvaluator metrics
- `bench_fp_transforms.py` - FactorPreprocess transforms
- `bench_fa_operations.py` - Fundamental analysis operations
- `bench_fo_search.py` - FactorOptimizer search algorithms

Module-specific benchmarks:
- `dataaccess/benchmarks/` - DataAccess performance tests
- `factor_engine/benchmarks/` - FactorEngine operator benchmarks
- `quant_evaluator/benchmark_ic_speedup.py` - IC calculation optimizations

## CI Environment Variables

### FactorEngine
- `FACTOR_ENGINE_DISABLE_NUMBA=1` - Disable Numba JIT in CI (faster startup)

### DataAccess
- Credentials are handled via CI secrets (not in repository)
- Mock fixtures used for most tests

## Artifact Retention

- **Benchmark results**: 90 days
- **Performance results**: 30 days
- **Security reports**: 30 days
- **Complexity reports**: 30 days
- **Coverage reports**: Uploaded to Codecov

## Branch Protection

Recommended branch protection rules for `main`/`master`:

- Require status checks:
  - `test-matrix (3.10)`
  - `test-matrix (3.11)`
  - `test-coverage`
  - `pre-commit-checks`
  - `black`
  - `flake8`
  - `isort`
- Require pull request reviews (1+)
- Dismiss stale reviews on push
- Require linear history
- No force pushes
- No deletions

## Performance Monitoring

Benchmarks run weekly and store historical baselines in `.github/benchmark-history/`:

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-08-14T02:00:00Z",
  "platform": "quantitative_analysis",
  "benchmarks": {
    "qe": {...},
    "fp": {...},
    "fa": {...},
    "fo": {...}
  }
}
```

Baseline files are kept for 30 days and can be used for regression analysis.

## Security Scanning

### Bandit
Scans for common security issues:
- SQL injection vulnerabilities
- Hard-coded credentials
- Use of insecure functions
- Weak cryptography

### Safety
Checks Python dependencies for known vulnerabilities:
- CVE tracking
- Security advisories
- Automatic updates via Dependabot (optional)

## Troubleshooting

### Test failures in CI but not locally

1. Check Python version matches (3.10 or 3.11)
2. Verify all dependencies installed: `make install`
3. Check for environment-specific issues (timezone, locale)
4. Review CI logs for specific error messages

### Pre-commit hook failures

1. Run `pre-commit run --all-files` to see all issues
2. Auto-fix formatting: `make format`
3. Check specific linter: `make lint-flake8`
4. Update hooks: `pre-commit autoupdate`

### Benchmark timeouts

1. Use `--quick` mode for faster execution
2. Check for resource contention
3. Review benchmark scale parameters
4. Increase timeout in workflow file if necessary

### MyPy type errors

1. MyPy jobs continue on error (non-blocking)
2. Fix type annotations incrementally
3. Use `# type: ignore` for third-party issues
4. Add stubs: `pip install types-*`

## Local Development Workflow

Recommended workflow for contributors:

```bash
# Initial setup
make install-dev
make pre-commit

# Before committing
make format          # Auto-format code
make lint            # Check for issues
make test-quick      # Run fast tests

# Before pushing
make test            # Full test suite
make audit           # Security and quality checks

# Weekly/monthly
make benchmark       # Performance regression check
```

## CI Performance Optimization

Current optimizations:

1. **Pip caching** - `actions/setup-python` with `cache: 'pip'`
2. **Concurrency** - Cancel in-progress runs for same ref
3. **Test filtering** - Skip `slow` and `integration` markers in quick builds
4. **Matrix parallelization** - Python 3.10 and 3.11 run concurrently
5. **Job isolation** - Independent jobs can run in parallel

## Future Enhancements

Potential improvements:

1. **Dependabot** - Automated dependency updates
2. **CodeQL** - Advanced security analysis
3. **Deploy workflows** - Automated deployment on release
4. **Nightly builds** - Full integration + slow tests
5. **Docker builds** - Containerized testing environments
6. **Performance trends** - Historical benchmark visualization
7. **Test parallelization** - pytest-xdist for faster execution
8. **Caching** - Dependency and build caching between runs

## Support

For CI/CD issues:

1. Check workflow run logs in GitHub Actions tab
2. Review recent changes to workflow files
3. Test locally with `make` commands
4. Consult team documentation in `/docs`
5. Open issue with `ci` label for persistent problems

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pre-commit Framework](https://pre-commit.com/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
