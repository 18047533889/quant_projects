# Data Quality Reporting System

Comprehensive data quality reporting and monitoring system for quantitative research and financial data analysis.

## Features

- **Data Profiling**: Analyze coverage, distributions, correlations, and statistical properties
- **Anomaly Detection**: Detect outliers, missing values, range violations, and unusual patterns
- **Report Generation**: Create HTML and JSON reports with visual styling
- **Alert System**: Configurable quality alert rules and notifications

## Installation

The data quality module is part of the quant_projects utilities:

```python
from utils.data_quality import (
    DataProfiler,
    AnomalyDetector,
    ReportGenerator,
    AlertSystem,
)
```

## Quick Start

```python
import pandas as pd
from utils.data_quality import DataProfiler, AnomalyDetector, ReportGenerator, ReportFormat

# Load your data
df = pd.read_csv("financial_data.csv")

# Profile the data
profiler = DataProfiler()
profile_result = profiler.profile(df, date_column="date", entity_column="symbol")

# Detect anomalies
detector = AnomalyDetector()
anomaly_result = detector.detect(df, date_column="date")

# Generate HTML report
generator = ReportGenerator(title="Data Quality Report")
report = generator.generate(
    profile_result,
    anomaly_result,
    format=ReportFormat.HTML,
    output_path="report.html"
)
```

## Components

### DataProfiler

Analyzes data quality, coverage, distributions, and correlations.

```python
profiler = DataProfiler(
    compute_correlations=True,
    correlation_threshold=0.95,
    max_categories=20,
    detect_time_series=True,
)

profile_result = profiler.profile(
    df,
    date_column="date",
    entity_column="symbol",
)

# Access results
print(f"Row count: {profile_result.row_count}")
print(f"Missing data: {profile_result.column_profiles['price'].null_percentage}%")

# Get high correlations
high_corr = profiler.get_high_correlations(profile_result)

# Get low coverage columns
low_coverage = profiler.get_low_coverage_columns(profile_result, threshold=70.0)
```

### AnomalyDetector

Detects outliers, missing values, range violations, and other anomalies.

```python
detector = AnomalyDetector(
    outlier_method="iqr",  # or "zscore", "mad"
    outlier_threshold=3.0,
    missing_threshold=10.0,
    sudden_change_threshold=5.0,
)

anomaly_result = detector.detect(
    df,
    date_column="date",
    expected_ranges={
        "price": (0, 1000),
        "volume": (0, 100000000),
    }
)

# Access results
print(f"Total anomalies: {anomaly_result.total_anomalies}")
print(f"High severity: {anomaly_result.severity_counts['high']}")

# Filter by type
from utils.data_quality import AnomalyType
outliers = anomaly_result.get_by_type(AnomalyType.OUTLIER)
```

### ReportGenerator

Creates HTML and JSON reports combining profiling and anomaly detection.

```python
generator = ReportGenerator(
    title="Monthly Data Quality Report",
    include_charts=True,
)

# Generate HTML report
html_report = generator.generate(
    profile_result,
    anomaly_result,
    format=ReportFormat.HTML,
    output_path="report.html",
)

# Generate JSON report
json_report = generator.generate(
    profile_result,
    anomaly_result,
    format=ReportFormat.JSON,
    output_path="report.json",
)
```

### AlertSystem

Evaluates quality metrics against configurable rules and generates alerts.

```python
alert_system = AlertSystem()

# Evaluate with default rules
alerts = alert_system.evaluate(profile_result, anomaly_result)

# Check alert levels
if alert_system.has_critical_alerts():
    print("Critical issues detected!")

# Add custom rule
from utils.data_quality import AlertRule, AlertLevel

custom_rule = AlertRule(
    name="min_rows_check",
    condition=lambda prof, anom: prof.row_count < 1000,
    alert_level=AlertLevel.WARNING,
    message_template="Dataset has fewer than 1000 rows",
)
alert_system.add_rule(custom_rule)

# Export alerts
text_export = alert_system.export_alerts(format="text")
json_export = alert_system.export_alerts(format="json")
```

## Anomaly Types

- `OUTLIER`: Statistical outliers detected by IQR, z-score, or MAD methods
- `MISSING_VALUE`: Excessive missing values in columns
- `INFINITE_VALUE`: Infinite values in numeric columns
- `DUPLICATE`: Duplicate rows in dataset
- `RANGE_VIOLATION`: Values outside expected ranges
- `SUDDEN_CHANGE`: Sudden changes in time series data
- `STALE_DATA`: Data that hasn't been updated recently
- `ZERO_VARIANCE`: Columns with constant values

## Alert Levels

- `INFO`: Informational alerts
- `WARNING`: Warnings that should be reviewed
- `ERROR`: Errors that need attention
- `CRITICAL`: Critical issues requiring immediate action

## Default Alert Rules

The system includes default rules for:
- High missing value rates (>50%)
- Critical anomalies (high severity issues)
- Stale data detection
- Infinite values
- Low data coverage (<70%)
- Duplicate rows
- Range violations
- Zero variance columns

## Examples

See `examples.py` for complete usage examples:

```bash
python -m utils.data_quality.examples
```

## Testing

Run the test suite:

```bash
pytest utils/data_quality/tests/ -v
```

## Dependencies

- pandas >= 2.2.3
- numpy == 2.1.3
- Python >= 3.8

## HTML Report Features

HTML reports include:
- Responsive design with fluid typography
- Dark-themed interface optimized for data analysis
- Summary statistics with visual metrics
- Column profiles with distributions
- Coverage visualization with progress bars
- Correlation matrix with high correlation highlights
- Anomaly details with severity badges
- Time series information
- Export-ready format

## Best Practices

1. **Regular Monitoring**: Run quality checks on data pipelines regularly
2. **Custom Rules**: Add domain-specific alert rules for your use case
3. **Expected Ranges**: Define expected ranges for critical metrics
4. **Time Series**: Always specify date columns for time series data
5. **Thresholds**: Adjust detection thresholds based on your data characteristics
6. **Archive Reports**: Save reports for historical quality tracking

## License

Part of the quant_projects utilities.
