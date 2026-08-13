# Smoke Test Report

**Generated:** 2026-08-14 01:16:52
**Duration:** 8.86 seconds
**Status:** ✅ PASSED

## Summary

This report covers comprehensive smoke tests for the quant_projects platform:

1. **Package Imports** - All 4 core packages (dataaccess, factor_engine, factor_optimizer, quant_evaluator)
2. **Backend Availability** - numpy, numba, polars, cupy, pandas, duckdb
3. **Basic Workflows** - End-to-end workflow tests for each package
4. **CLI Tools** - Script execution and syntax validation
5. **Documentation** - Documentation completeness and readability
6. **Benchmarks** - Benchmark suite availability and execution

## Test Results

```
[1m============================= test session starts ==============================[0m
platform linux -- Python 3.10.12, pytest-9.1.1, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/shw/quant_projects
configfile: pytest.ini
plugins: anyio-4.14.2
[1mcollecting ... [0mcollected 30 items

tests/smoke/test_backends.py::TestBackends::test_numpy_backend [32mPASSED[0m[32m    [  3%][0m
tests/smoke/test_backends.py::TestBackends::test_numba_backend [32mPASSED[0m[32m    [  6%][0m
tests/smoke/test_backends.py::TestBackends::test_polars_backend [32mPASSED[0m[32m   [ 10%][0m
tests/smoke/test_backends.py::TestBackends::test_cupy_backend [33mSKIPPED[0m[32m    [ 13%][0m
tests/smoke/test_backends.py::TestBackends::test_pandas_backend [32mPASSED[0m[32m   [ 16%][0m
tests/smoke/test_backends.py::TestBackends::test_duckdb_backend [32mPASSED[0m[32m   [ 20%][0m
tests/smoke/test_benchmarks.py::TestBenchmarks::test_benchmarks_directory_exists [32mPASSED[0m[32m [ 23%][0m
tests/smoke/test_benchmarks.py::TestBenchmarks::test_benchmark_scripts_exist [32mPASSED[0m[32m [ 26%][0m
tests/smoke/test_benchmarks.py::TestBenchmarks::test_benchmark_help_flag [32mPASSED[0m[32m [ 30%][0m
tests/smoke/test_benchmarks.py::TestBenchmarks::test_benchmark_readme_exists [32mPASSED[0m[32m [ 33%][0m
tests/smoke/test_benchmarks.py::TestBenchmarks::test_baseline_json_exists [32mPASSED[0m[32m [ 36%][0m
tests/smoke/test_benchmarks.py::TestBenchmarks::test_analyze_results_script [32mPASSED[0m[32m [ 40%][0m
tests/smoke/test_cli_tools.py::TestCLITools::test_help_flag_scripts [32mPASSED[0m[32m [ 43%][0m
tests/smoke/test_cli_tools.py::TestCLITools::test_shell_scripts_syntax [32mPASSED[0m[32m [ 46%][0m
tests/smoke/test_documentation.py::TestDocumentation::test_main_readme_exists [32mPASSED[0m[32m [ 50%][0m
tests/smoke/test_documentation.py::TestDocumentation::test_docs_directory_exists [32mPASSED[0m[32m [ 53%][0m
tests/smoke/test_documentation.py::TestDocumentation::test_docs_readme_exists [32mPASSED[0m[32m [ 56%][0m
tests/smoke/test_documentation.py::TestDocumentation::test_package_readmes_exist [32mPASSED[0m[32m [ 60%][0m
tests/smoke/test_documentation.py::TestDocumentation::test_markdown_files_readable [32mPASSED[0m[32m [ 63%][0m
tests/smoke/test_documentation.py::TestDocumentation::test_chinese_docs_readable [32mPASSED[0m[32m [ 66%][0m
tests/smoke/test_imports.py::TestPackageImports::test_dataaccess_imports [32mPASSED[0m[32m [ 70%][0m
tests/smoke/test_imports.py::TestPackageImports::test_factor_engine_imports [32mPASSED[0m[32m [ 73%][0m
tests/smoke/test_imports.py::TestPackageImports::test_factor_optimizer_imports [32mPASSED[0m[32m [ 76%][0m
tests/smoke/test_imports.py::TestPackageImports::test_quant_evaluator_imports [32mPASSED[0m[32m [ 80%][0m
tests/smoke/test_imports.py::TestPackageImports::test_all_packages_have_version [33mSKIPPED[0m[32m [ 83%][0m
tests/smoke/test_workflows.py::TestDataAccessWorkflow::test_dataaccess_basic_workflow [32mPASSED[0m[32m [ 86%][0m
tests/smoke/test_workflows.py::TestFactorEngineWorkflow::test_factor_engine_backend_import [33mSKIPPED[0m[32m [ 90%][0m
tests/smoke/test_workflows.py::TestFactorEngineWorkflow::test_factor_engine_simple_operation [32mPASSED[0m[32m [ 93%][0m
tests/smoke/test_workflows.py::TestFactorOptimizerWorkflow::test_factor_optimizer_basic [32mPASSED[0m[32m [ 96%][0m
tests/smoke/test_workflows.py::TestQuantEvaluatorWorkflow::test_quant_evaluator_metrics [32mPASSED[0m[32m [100%][0m

-- generated xml file: /home/shw/quant_projects/tests/smoke/smoke_results.xml --
[36m[1m=========================== short test summary info ============================[0m
[33mSKIPPED[0m [1] tests/smoke/test_backends.py:60: cupy not available or no GPU: No module named 'cupy'
[33mSKIPPED[0m [1] tests/smoke/test_imports.py:52: factor_engine has no version attribute
[33mSKIPPED[0m [1] tests/smoke/test_workflows.py:39: FactorEngine workflow requires setup: No module named 'planner'
[32m======================== [32m[1m27 passed[0m, [33m3 skipped[0m[32m in 8.15s[0m[32m =========================[0m

```

## Test Categories

### 1. Package Imports
Tests that all core packages can be imported without errors:
- dataaccess
- factor_engine
- factor_optimizer
- quant_evaluator

### 2. Backend Availability
Tests computational backend availability:
- **numpy**: Core numerical computing (required)
- **numba**: JIT compilation for performance
- **polars**: High-performance dataframes
- **cupy**: GPU acceleration (optional)
- **pandas**: Data manipulation
- **duckdb**: SQL analytics engine

### 3. Basic Workflows
Tests end-to-end functionality:
- DataAccess: Registry and data source operations
- FactorEngine: Operator registry and execution
- FactorOptimizer: Basic optimization workflows
- QuantEvaluator: Metrics calculation

### 4. CLI Tools
Tests command-line tools:
- Python scripts: --help flag and basic execution
- Shell scripts: Syntax validation

### 5. Documentation
Tests documentation completeness:
- Main README
- Package READMEs
- Documentation directory structure
- Markdown file readability (including Chinese)

### 6. Benchmarks
Tests benchmark suite:
- Benchmark scripts existence
- Execution capability
- Documentation availability
- Results analysis tools

## Performance Target

Target: < 2 minutes
Actual: 8.86 seconds
Status: ✅ Within target

## Next Steps

✅ All smoke tests passed. System is operational.

## Test Execution

To run smoke tests manually:

```bash
cd /home/shw/quant_projects
python3 -m pytest tests/smoke/ -v
```

To run a specific test category:

```bash
python3 -m pytest tests/smoke/test_imports.py -v
python3 -m pytest tests/smoke/test_backends.py -v
python3 -m pytest tests/smoke/test_workflows.py -v
python3 -m pytest tests/smoke/test_cli_tools.py -v
python3 -m pytest tests/smoke/test_documentation.py -v
python3 -m pytest tests/smoke/test_benchmarks.py -v
```

---
*Generated by smoke test runner*
