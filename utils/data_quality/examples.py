"""
Example usage script demonstrating the data quality system.

Run with: python -m utils.data_quality.examples.basic_usage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.data_quality import (
    DataProfiler,
    AnomalyDetector,
    ReportGenerator,
    AlertSystem,
    ReportFormat,
)


def create_sample_data():
    """Create sample financial data for demonstration."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    data = {
        "date": dates,
        "symbol": ["AAPL"] * 50 + ["MSFT"] * 50,
        "price": np.random.uniform(100, 200, 100),
        "volume": np.random.randint(1000000, 10000000, 100),
        "returns": np.random.normal(0, 0.02, 100),
        "bid": np.random.uniform(99, 199, 100),
        "ask": np.random.uniform(101, 201, 100),
    }

    # Introduce some quality issues
    data["price"][10] = np.inf  # Infinite value
    data["price"][20] = np.nan  # Missing value
    data["volume"][30] = -1000000  # Negative volume
    data["returns"][40] = 10.0  # Outlier
    data["optional_field"] = [np.nan] * 90 + list(range(10))

    return pd.DataFrame(data)


def main():
    """Run complete data quality analysis."""
    print("=" * 80)
    print("Data Quality Analysis Example")
    print("=" * 80)
    print()

    # Create sample data
    print("Creating sample data...")
    df = create_sample_data()
    print(f"Dataset shape: {df.shape}")
    print()

    # Step 1: Profile the data
    print("Step 1: Profiling data...")
    print("-" * 80)

    profiler = DataProfiler(
        compute_correlations=True,
        correlation_threshold=0.95,
        detect_time_series=True,
    )

    profile_result = profiler.profile(
        df,
        date_column="date",
        entity_column="symbol",
    )

    print(f"Row count: {profile_result.row_count:,}")
    print(f"Column count: {profile_result.column_count}")
    print(f"Memory usage: {profile_result.memory_usage_mb:.2f} MB")
    print(f"Date range: {profile_result.date_range}")
    print(f"Entity count: {profile_result.entity_count}")
    print()

    # Show some column profiles
    print("Column profiles:")
    for col_name, col_profile in list(profile_result.column_profiles.items())[:3]:
        print(f"  {col_name}:")
        print(f"    Type: {col_profile.dtype}")
        print(f"    Missing: {col_profile.null_percentage:.1f}%")
        if col_profile.mean is not None:
            print(f"    Mean: {col_profile.mean:.4f}")
            print(f"    Std: {col_profile.std:.4f}")
    print()

    # Check for high correlations
    high_corr = profiler.get_high_correlations(profile_result)
    if high_corr:
        print(f"High correlations found: {len(high_corr)}")
        for col1, col2, corr in high_corr[:3]:
            print(f"  {col1} <-> {col2}: {corr:.4f}")
    else:
        print("No high correlations found")
    print()

    # Check for low coverage
    low_coverage = profiler.get_low_coverage_columns(profile_result, threshold=70.0)
    if low_coverage:
        print(f"Low coverage columns: {len(low_coverage)}")
        for col, coverage in low_coverage:
            print(f"  {col}: {coverage:.1f}%")
    else:
        print("All columns have good coverage")
    print()

    # Step 2: Detect anomalies
    print("Step 2: Detecting anomalies...")
    print("-" * 80)

    detector = AnomalyDetector(
        outlier_method="iqr",
        outlier_threshold=3.0,
        missing_threshold=10.0,
        sudden_change_threshold=5.0,
    )

    expected_ranges = {
        "volume": (0, 100000000),
        "returns": (-0.5, 0.5),
        "price": (0, 1000),
    }

    anomaly_result = detector.detect(
        df,
        date_column="date",
        expected_ranges=expected_ranges,
    )

    print(f"Total anomalies: {anomaly_result.total_anomalies}")
    print(f"Severity breakdown:")
    for severity, count in anomaly_result.severity_counts.items():
        print(f"  {severity}: {count}")
    print()

    print(f"Anomaly types:")
    for anom_type, count in anomaly_result.type_counts.items():
        print(f"  {anom_type}: {count}")
    print()

    # Show some anomalies
    if anomaly_result.anomalies:
        print("Sample anomalies:")
        for anomaly in anomaly_result.anomalies[:5]:
            print(f"  [{anomaly.severity.upper()}] {anomaly.column}: {anomaly.description}")
    print()

    # Step 3: Generate alerts
    print("Step 3: Generating alerts...")
    print("-" * 80)

    alert_system = AlertSystem()
    alerts = alert_system.evaluate(profile_result, anomaly_result)

    print(f"Total alerts: {len(alerts)}")

    summary = alert_system.get_summary()
    print(f"Alert breakdown:")
    for level, count in summary.items():
        if level != "total":
            print(f"  {level}: {count}")
    print()

    if alerts:
        print("Sample alerts:")
        for alert in alerts[:5]:
            print(f"  [{alert.level.value.upper()}] {alert.title}: {alert.message}")
    print()

    # Check for critical issues
    if alert_system.has_critical_alerts():
        print("⚠️  CRITICAL ALERTS DETECTED!")
    elif alert_system.has_errors():
        print("⚠️  ERROR ALERTS DETECTED!")
    else:
        print("✓ No critical issues")
    print()

    # Step 4: Generate reports
    print("Step 4: Generating reports...")
    print("-" * 80)

    generator = ReportGenerator(
        title="Sample Data Quality Report",
        include_charts=True,
    )

    # Generate HTML report
    html_report = generator.generate(
        profile_result,
        anomaly_result,
        format=ReportFormat.HTML,
        output_path="/tmp/data_quality_report.html",
    )
    print(f"HTML report saved to: /tmp/data_quality_report.html")
    print(f"HTML report size: {len(html_report):,} characters")

    # Generate JSON report
    json_report = generator.generate(
        profile_result,
        anomaly_result,
        format=ReportFormat.JSON,
        output_path="/tmp/data_quality_report.json",
    )
    print(f"JSON report saved to: /tmp/data_quality_report.json")
    print(f"JSON report size: {len(json_report):,} characters")
    print()

    # Export alerts
    print("Step 5: Exporting alerts...")
    print("-" * 80)

    text_alerts = alert_system.export_alerts(format="text")
    print("Alerts (text format):")
    print(text_alerts)
    print()

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
