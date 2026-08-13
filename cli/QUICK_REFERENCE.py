#!/usr/bin/env python3
"""
CLI Quick Reference and Examples
Run this to see example commands for all CLI tools
"""

QUICK_REFERENCE = """
================================================================================
                    CLI TOOLS QUICK REFERENCE
================================================================================

All tools are located in: /home/shw/quant_projects/cli/

--------------------------------------------------------------------------------
1. QE_CLI - Quant Evaluator
--------------------------------------------------------------------------------
Evaluate factor quality metrics

Basic usage:
  python3 qe_cli.py evaluate factor.csv label.csv -m rank_ic
  python3 qe_cli.py info factor.csv

Full example:
  python3 qe_cli.py evaluate factor.csv label.csv \\
    -m rank_ic -m ic -m ir -m turnover \\
    -o results.json -f json

--------------------------------------------------------------------------------
2. FP_CLI - Factor Preprocessing
--------------------------------------------------------------------------------
Transform and prepare factor data

Basic usage:
  python3 fp_cli.py transform input.csv output.csv -t zscore
  python3 fp_cli.py stats input.csv

Full pipeline:
  python3 fp_cli.py transform raw.csv clean.csv \\
    -t fillna -t winsorize -t zscore \\
    --fillna-method forward --winsorize-std 3.0 --axis cs

Check data quality:
  python3 fp_cli.py outliers input.csv --top-n 10
  python3 fp_cli.py stats input.csv

--------------------------------------------------------------------------------
3. FA_CLI - Factor Analysis (Registry)
--------------------------------------------------------------------------------
Query factor operator registry

Basic usage:
  python3 fa_cli.py stats
  python3 fa_cli.py families
  python3 fa_cli.py list-operators

Search operators:
  python3 fa_cli.py search "moving average"
  python3 fa_cli.py list-operators --family technical --timing DAILY
  python3 fa_cli.py info rolling_mean

--------------------------------------------------------------------------------
4. FO_CLI - Factor Optimization
--------------------------------------------------------------------------------
Run factor search campaigns

Basic usage:
  python3 fo_cli.py search --n-trials 100 -o results.json
  python3 fo_cli.py analyze results.json

Full campaign:
  python3 fo_cli.py search \\
    --n-trials 500 --budget 2000 --seed 42 \\
    -o search_results.json

  python3 fo_cli.py analyze search_results.json \\
    --top-n 20 --metric ic

Other commands:
  python3 fo_cli.py grammar
  python3 fo_cli.py benchmark --size 10000

--------------------------------------------------------------------------------
5. BENCHMARK_CLI - Performance Benchmarks
--------------------------------------------------------------------------------
Run and analyze benchmarks

Basic usage:
  python3 benchmark_cli.py run --suite all -o baseline.json
  python3 benchmark_cli.py report baseline.json

Run specific suites:
  python3 benchmark_cli.py run --suite qe --suite fp --quick

Compare for regressions:
  python3 benchmark_cli.py compare baseline.json current.json --threshold 0.15

List available:
  python3 benchmark_cli.py list-suites

================================================================================
                        COMMON WORKFLOWS
================================================================================

Workflow 1: Factor Evaluation Pipeline
---------------------------------------
# Step 1: Preprocess
python3 fp_cli.py transform raw_factor.csv clean_factor.csv \\
  -t fillna -t winsorize -t zscore \\
  --fillna-method forward --winsorize-std 3.0 --axis cs

# Step 2: Check quality
python3 fp_cli.py stats clean_factor.csv
python3 fp_cli.py outliers clean_factor.csv --top-n 5

# Step 3: Evaluate
python3 qe_cli.py evaluate clean_factor.csv labels.csv \\
  -m rank_ic -m ic -m ir -o eval_results.json -f json

Workflow 2: Search and Optimize
--------------------------------
# Run search
python3 fo_cli.py search --n-trials 500 --budget 2000 -o search.json

# Analyze results
python3 fo_cli.py analyze search.json --top-n 20 --metric ic

# View mutation grammar
python3 fo_cli.py grammar

Workflow 3: Registry Exploration
---------------------------------
# Overview
python3 fa_cli.py stats

# Find operators
python3 fa_cli.py search "volatility"
python3 fa_cli.py list-operators --family statistical

# Get details
python3 fa_cli.py info ewm_std

Workflow 4: Performance Monitoring
-----------------------------------
# Establish baseline
python3 benchmark_cli.py run --suite all -o baseline.json

# After code changes
python3 benchmark_cli.py run --suite all -o current.json

# Check for regressions
python3 benchmark_cli.py compare baseline.json current.json --threshold 0.10

# Generate report
python3 benchmark_cli.py report current.json > report.txt

================================================================================
                        FILE FORMATS
================================================================================

Factor/Label Data (CSV):
  - Index: dates (datetime)
  - Columns: asset identifiers
  - Values: factor values or returns

  Example:
            A001      A002      A003
  2023-01-01  0.123    -0.456     0.789
  2023-01-02  0.234     0.567    -0.890

Parquet files also supported (auto-detected by extension)

Output Formats:
  - text: Human-readable
  - json: Machine-readable, structured
  - csv: Tabular data

================================================================================
                        TESTING
================================================================================

Run all tests:
  cd /home/shw/quant_projects/cli
  pytest tests/ -v

Run specific test file:
  pytest tests/test_qe_cli.py -v
  pytest tests/test_fp_cli.py -v

Current status: 31 passed, 6 skipped (8.21s)

================================================================================
                        HELP & DOCUMENTATION
================================================================================

Every command has --help:
  python3 qe_cli.py --help
  python3 qe_cli.py evaluate --help

Full documentation:
  cat /home/shw/quant_projects/cli/README.md

Implementation details:
  cat /home/shw/quant_projects/cli/IMPLEMENTATION_SUMMARY.md

================================================================================
"""

if __name__ == "__main__":
    print(QUICK_REFERENCE)
