# CLI Tools

Command-line interfaces for the quantitative analysis platform.

## Overview

This directory contains five CLI tools for interacting with the platform:

1. **qe_cli.py** - Quant Evaluator: Evaluate factor quality metrics
2. **fp_cli.py** - Factor Preprocessing: Transform and prepare factor data
3. **fa_cli.py** - Factor Analysis: Query the factor operator registry
4. **fo_cli.py** - Factor Optimization: Run factor search campaigns
5. **benchmark_cli.py** - Benchmarks: Run and analyze performance benchmarks

## Installation

Install required dependencies:

```bash
pip install click pandas numpy scipy
```

For full functionality with factor_engine integration:

```bash
cd /home/shw/quant_projects/factor_engine
pip install -e .
```

## Usage

### 1. Quant Evaluator CLI (qe_cli.py)

Evaluate factors against labels with various metrics.

```bash
# Evaluate a factor
python qe_cli.py evaluate factor.csv label.csv -m rank_ic -m ic -m ir

# Get factor info
python qe_cli.py info factor.csv

# Save results to file
python qe_cli.py evaluate factor.csv label.csv -o results.json -f json
```

**Available metrics:**
- `rank_ic` - Rank Information Coefficient (Spearman)
- `ic` - Information Coefficient (Pearson)
- `ir` - Information Ratio (IC mean/std)
- `turnover` - Average factor turnover

**Output formats:** `text`, `json`, `csv`

### 2. Factor Preprocessing CLI (fp_cli.py)

Transform and prepare factor data.

```bash
# Apply z-score normalization
python fp_cli.py transform input.csv output.csv -t zscore --axis cs

# Multiple transformations
python fp_cli.py transform input.csv output.csv \
  -t fillna -t winsorize -t zscore \
  --fillna-method mean --winsorize-std 3.0

# Detect outliers
python fp_cli.py outliers input.csv --top-n 10

# Show statistics
python fp_cli.py stats input.csv
```

**Available transforms:**
- `zscore` - Z-score normalization
- `rank` - Rank transformation (percentile)
- `winsorize` - Outlier winsorization
- `fillna` - Fill missing values
- `neutralize` - Market neutralization

**Axes:** `cs` (cross-sectional), `ts` (time-series), `both`

### 3. Factor Analysis CLI (fa_cli.py)

Query and inspect the factor operator registry.

```bash
# List all operators
python fa_cli.py list-operators

# Filter by family, timing, or lane
python fa_cli.py list-operators --family technical --timing DAILY

# Get operator count
python fa_cli.py list-operators --count

# Show operator details
python fa_cli.py info rolling_mean

# List all families
python fa_cli.py families

# Search for operators
python fa_cli.py search "moving average"

# Registry statistics
python fa_cli.py stats
```

### 4. Factor Optimization CLI (fo_cli.py)

Run factor search campaigns with mutation tracking.

```bash
# Run a search campaign
python fo_cli.py search --n-trials 100 --budget 1000 -o results.json

# Analyze results
python fo_cli.py analyze results.json --top-n 10 --metric ic

# Benchmark deduplication cache
python fo_cli.py benchmark --size 10000

# Show mutation grammar
python fo_cli.py grammar
```

**Mutation types:**
- `param_tweak` - Adjust numeric parameters
- `operator_swap` - Replace operators
- `window_adjust` - Modify window sizes
- `combine` - Combine factors
- `transform` - Apply transformations

### 5. Benchmark CLI (benchmark_cli.py)

Run and analyze performance benchmarks.

```bash
# Run all benchmarks
python benchmark_cli.py run --suite all -o baseline.json

# Run specific suites
python benchmark_cli.py run --suite qe --suite fp --quick

# Generate report
python benchmark_cli.py report baseline.json

# Compare benchmarks (detect regressions)
python benchmark_cli.py compare baseline.json current.json --threshold 0.1

# List available suites
python benchmark_cli.py list-suites
```

**Available suites:**
- `qe` - Quant Evaluator metrics
- `fp` - Factor Preprocessing transforms
- `fa` - Factor Analysis operations
- `fo` - Factor Optimization search

## Examples

### Complete Factor Evaluation Pipeline

```bash
# 1. Preprocess factor data
python fp_cli.py transform raw_factor.csv clean_factor.csv \
  -t fillna -t winsorize -t zscore \
  --fillna-method forward --winsorize-std 3.0 --axis cs

# 2. Check for issues
python fp_cli.py stats clean_factor.csv
python fp_cli.py outliers clean_factor.csv --top-n 5

# 3. Evaluate against labels
python qe_cli.py evaluate clean_factor.csv labels.csv \
  -m rank_ic -m ic -m ir -o eval_results.json -f json

# 4. View results
cat eval_results.json
```

### Search and Optimize Factors

```bash
# 1. Run search campaign
python fo_cli.py search --n-trials 500 --budget 2000 \
  --seed 42 -o search_results.json

# 2. Analyze top performers
python fo_cli.py analyze search_results.json \
  --top-n 20 --metric ic

# 3. Check mutation performance
python fo_cli.py analyze search_results.json
```

### Registry Exploration

```bash
# 1. Get overview
python fa_cli.py stats

# 2. List families
python fa_cli.py families

# 3. Search for specific operators
python fa_cli.py search "volatility"

# 4. Get details
python fa_cli.py info ewm_std
```

### Performance Monitoring

```bash
# 1. Establish baseline
python benchmark_cli.py run --suite all -o baseline.json

# 2. After changes, run again
python benchmark_cli.py run --suite all -o current.json

# 3. Compare for regressions
python benchmark_cli.py compare baseline.json current.json --threshold 0.15

# 4. Generate report
python benchmark_cli.py report current.json > performance_report.txt
```

## Testing

Run the test suite:

```bash
cd /home/shw/quant_projects/cli
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_qe_cli.py -v
pytest tests/test_fp_cli.py -v
pytest tests/test_fa_cli.py -v
pytest tests/test_fo_cli.py -v
pytest tests/test_benchmark_cli.py -v
```

## Common Options

All CLIs support:
- `--help` - Show help message
- `--version` - Show version

## File Formats

The CLIs support:
- **CSV** - Standard comma-separated values
- **Parquet** - Apache Parquet (auto-detected by extension)
- **JSON** - For structured results

Factor data should be in wide format:
- Index: dates (datetime)
- Columns: asset identifiers
- Values: factor values or returns

## Notes

- **fa_cli** requires factor_engine to be importable
- **benchmark_cli** may need specific dependencies per suite
- All tools use `/home/shw/quant_projects` as root
- Tests use fixtures and may skip if dependencies unavailable

## Architecture

Each CLI is self-contained with:
- Click for argument parsing
- Minimal external dependencies
- Clear error messages
- Structured output formats
- Comprehensive help text

The CLIs are designed to be:
1. **Composable** - Output from one can feed another
2. **Scriptable** - Easy to use in automation
3. **Documented** - Built-in help for every command
4. **Tested** - Full test coverage with pytest
5. **Performant** - Efficient implementations
