"""
Data profiling module for analyzing data quality, coverage, distributions, and correlations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """Profile statistics for a single column."""

    name: str
    dtype: str
    count: int
    null_count: int
    null_percentage: float
    unique_count: int

    # Numeric statistics
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

    # Categorical statistics
    most_common: Optional[List[Tuple[Any, int]]] = None

    # Quality metrics
    inf_count: int = 0
    negative_inf_count: int = 0
    zero_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "count": self.count,
            "null_count": self.null_count,
            "null_percentage": self.null_percentage,
            "unique_count": self.unique_count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "most_common": self.most_common,
            "inf_count": self.inf_count,
            "negative_inf_count": self.negative_inf_count,
            "zero_count": self.zero_count,
        }


@dataclass
class ProfileResult:
    """Complete data profiling result."""

    timestamp: datetime
    row_count: int
    column_count: int
    memory_usage_mb: float
    column_profiles: Dict[str, ColumnProfile]
    correlations: Optional[pd.DataFrame] = None
    coverage_matrix: Optional[pd.DataFrame] = None

    # Time series specific
    date_column: Optional[str] = None
    date_range: Optional[Tuple[Any, Any]] = None
    date_gaps: Optional[List[Any]] = None

    # Panel data specific
    entity_column: Optional[str] = None
    entity_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_usage_mb": self.memory_usage_mb,
            "column_profiles": {k: v.to_dict() for k, v in self.column_profiles.items()},
            "date_column": self.date_column,
            "date_range": [str(self.date_range[0]), str(self.date_range[1])] if self.date_range else None,
            "date_gaps_count": len(self.date_gaps) if self.date_gaps else 0,
            "entity_column": self.entity_column,
            "entity_count": self.entity_count,
        }


class DataProfiler:
    """
    Comprehensive data profiler for financial time series and panel data.

    Analyzes data quality, coverage, distributions, correlations, and time series properties.
    """

    def __init__(
        self,
        compute_correlations: bool = True,
        correlation_threshold: float = 0.95,
        max_categories: int = 20,
        detect_time_series: bool = True,
    ):
        """
        Initialize profiler.

        Args:
            compute_correlations: Whether to compute correlation matrix
            correlation_threshold: Threshold for flagging high correlations
            max_categories: Maximum unique values to treat as categorical
            detect_time_series: Automatically detect time series structure
        """
        self.compute_correlations = compute_correlations
        self.correlation_threshold = correlation_threshold
        self.max_categories = max_categories
        self.detect_time_series = detect_time_series

    def profile(
        self,
        df: pd.DataFrame,
        date_column: Optional[str] = None,
        entity_column: Optional[str] = None,
    ) -> ProfileResult:
        """
        Profile a DataFrame.

        Args:
            df: DataFrame to profile
            date_column: Optional date column name for time series analysis
            entity_column: Optional entity column name for panel data analysis

        Returns:
            ProfileResult with comprehensive statistics
        """
        if df.empty:
            raise ValueError("Cannot profile empty DataFrame")

        # Auto-detect date column if requested
        if date_column is None and self.detect_time_series:
            date_column = self._detect_date_column(df)

        # Compute basic stats
        row_count = len(df)
        column_count = len(df.columns)
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Profile each column
        column_profiles = {}
        for col in df.columns:
            column_profiles[col] = self._profile_column(df[col])

        # Compute correlations for numeric columns
        correlations = None
        if self.compute_correlations:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlations = df[numeric_cols].corr()

        # Compute coverage matrix
        coverage_matrix = self._compute_coverage_matrix(df)

        # Time series analysis
        date_range = None
        date_gaps = None
        if date_column and date_column in df.columns:
            date_range = (df[date_column].min(), df[date_column].max())
            date_gaps = self._detect_date_gaps(df[date_column])

        # Panel data analysis
        entity_count = None
        if entity_column and entity_column in df.columns:
            entity_count = df[entity_column].nunique()

        return ProfileResult(
            timestamp=datetime.now(),
            row_count=row_count,
            column_count=column_count,
            memory_usage_mb=memory_usage_mb,
            column_profiles=column_profiles,
            correlations=correlations,
            coverage_matrix=coverage_matrix,
            date_column=date_column,
            date_range=date_range,
            date_gaps=date_gaps,
            entity_column=entity_column,
            entity_count=entity_count,
        )

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        """Profile a single column."""
        count = len(series)
        null_count = int(series.isna().sum())
        null_percentage = (null_count / count * 100) if count > 0 else 0.0

        # Valid values for unique count
        valid_series = series.dropna()
        unique_count = int(valid_series.nunique())

        profile = ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            count=count,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
        )

        # Numeric statistics (skip boolean dtype which is technically numeric but not quantifiable)
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            if len(valid_series) > 0:
                profile.mean = float(valid_series.mean())
                profile.std = float(valid_series.std())
                profile.min = float(valid_series.min())
                profile.max = float(valid_series.max())
                profile.median = float(valid_series.median())
                profile.q25 = float(valid_series.quantile(0.25))
                profile.q75 = float(valid_series.quantile(0.75))

                # Higher order moments
                profile.skewness = float(valid_series.skew())
                profile.kurtosis = float(valid_series.kurtosis())

                # Special value counts
                profile.inf_count = int(np.isinf(series).sum())
                profile.negative_inf_count = int((series == -np.inf).sum())
                profile.zero_count = int((series == 0).sum())

        # Categorical statistics
        elif unique_count <= self.max_categories and unique_count > 0:
            value_counts = valid_series.value_counts().head(10)
            profile.most_common = [(val, int(cnt)) for val, cnt in value_counts.items()]

        return profile

    def _compute_coverage_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute data coverage (non-null percentage) per column."""
        coverage = (1 - df.isna().mean()) * 100
        return pd.DataFrame({
            "column": coverage.index,
            "coverage_pct": coverage.values,
        })

    def _detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Attempt to detect date column automatically."""
        candidates = ["date", "datetime", "timestamp", "time", "trading_date", "trade_date"]

        for col in df.columns:
            if col.lower() in candidates:
                return col
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        return None

    def _detect_date_gaps(self, date_series: pd.Series) -> List[Any]:
        """Detect gaps in time series data."""
        if not pd.api.types.is_datetime64_any_dtype(date_series):
            try:
                date_series = pd.to_datetime(date_series)
            except Exception:
                return []

        sorted_dates = date_series.dropna().sort_values().unique()
        if len(sorted_dates) < 2:
            return []

        # Infer frequency
        diffs = pd.Series(sorted_dates[1:]) - pd.Series(sorted_dates[:-1])
        mode_diff = diffs.mode()

        if len(mode_diff) == 0:
            return []

        expected_diff = mode_diff.iloc[0]

        # Find gaps
        gaps = []
        for i in range(len(sorted_dates) - 1):
            actual_diff = sorted_dates[i + 1] - sorted_dates[i]
            if actual_diff > expected_diff * 1.5:  # Allow 50% tolerance
                gaps.append(sorted_dates[i])

        return gaps

    def get_high_correlations(
        self,
        profile_result: ProfileResult,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Extract pairs of highly correlated columns.

        Args:
            profile_result: Profile result containing correlation matrix
            threshold: Correlation threshold (uses instance default if None)

        Returns:
            List of (col1, col2, correlation) tuples
        """
        if profile_result.correlations is None:
            return []

        threshold = threshold if threshold is not None else self.correlation_threshold
        corr_matrix = profile_result.correlations

        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]

                if abs(corr_val) >= threshold:
                    high_corr.append((col1, col2, float(corr_val)))

        return sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)

    def get_low_coverage_columns(
        self,
        profile_result: ProfileResult,
        threshold: float = 50.0,
    ) -> List[Tuple[str, float]]:
        """
        Get columns with coverage below threshold.

        Args:
            profile_result: Profile result
            threshold: Minimum coverage percentage

        Returns:
            List of (column, coverage_pct) tuples
        """
        low_coverage = []
        for col_name, col_profile in profile_result.column_profiles.items():
            coverage_pct = 100.0 - col_profile.null_percentage
            if coverage_pct < threshold:
                low_coverage.append((col_name, coverage_pct))

        return sorted(low_coverage, key=lambda x: x[1])
