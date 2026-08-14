"""
FM2-P0-005/006: gap_days enforcement tests for standalone modeling package.

Tests that SplitSpec.__post_init__ actually enforces gap_days (not just documents it).
"""
import pytest
from datetime import datetime

from modeling.contracts import SplitSpec
from modeling.errors import SplitError


class TestGapDaysEnforcement:
    """Test that gap_days is ACTUALLY enforced at construction time."""

    def test_gap_days_zero_allows_adjacent_dates(self):
        """gap_days=0 (default) allows train_end and val_start to be adjacent."""
        split = SplitSpec(
            split_id="adjacent",
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 12, 31),
            val_start=datetime(2021, 1, 1),  # Next day
            val_end=datetime(2021, 6, 30),
            gap_days=0,
        )
        assert split.val_start >= split.train_end

    def test_gap_days_five_enforced(self):
        """gap_days=5 requires at least 6 calendar days between train_end and val_start."""
        # Valid: 6+ days gap
        split = SplitSpec(
            split_id="valid_gap",
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 12, 31),
            val_start=datetime(2021, 1, 6),  # 6 days later (5 full days between)
            val_end=datetime(2021, 6, 30),
            gap_days=5,
        )
        assert (split.val_start - split.train_end).days == 6

    def test_gap_days_insufficient_gap_raises_error(self):
        """SplitSpec with gap_days=5 but only 3 days actual gap raises SplitError."""
        with pytest.raises(SplitError, match="gap_days=5 requires at least 6 days"):
            SplitSpec(
                split_id="insufficient_gap",
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 12, 31),
                val_start=datetime(2021, 1, 3),  # Only 3 days gap
                val_end=datetime(2021, 6, 30),
                gap_days=5,  # Requires 6 days
            )

    def test_gap_days_exact_boundary_passes(self):
        """gap_days=5 with exactly 6 days gap passes."""
        split = SplitSpec(
            split_id="exact_boundary",
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 12, 31),
            val_start=datetime(2021, 1, 6),  # Exactly 6 days (5 full days between)
            val_end=datetime(2021, 6, 30),
            gap_days=5,
        )
        assert (split.val_start - split.train_end).days == 6

    def test_gap_days_one_off_boundary_fails(self):
        """gap_days=5 with 5 days gap (one day short) fails."""
        with pytest.raises(SplitError, match="gap_days=5 requires at least 6 days"):
            SplitSpec(
                split_id="one_off",
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 12, 31),
                val_start=datetime(2021, 1, 5),  # Only 5 days gap
                val_end=datetime(2021, 6, 30),
                gap_days=5,
            )

    def test_gap_days_ten_enforced(self):
        """gap_days=10 requires at least 11 calendar days."""
        # Valid
        split = SplitSpec(
            split_id="gap_10",
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 12, 31),
            val_start=datetime(2021, 1, 11),  # 11 days later
            val_end=datetime(2021, 6, 30),
            gap_days=10,
        )
        assert (split.val_start - split.train_end).days == 11

        # Invalid
        with pytest.raises(SplitError, match="gap_days=10 requires at least 11 days"):
            SplitSpec(
                split_id="gap_10_insufficient",
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 12, 31),
                val_start=datetime(2021, 1, 10),  # Only 10 days gap
                val_end=datetime(2021, 6, 30),
                gap_days=10,
            )

    def test_gap_days_not_checked_when_no_validation(self):
        """A train-only split has no boundary after train to validate."""
        split = SplitSpec(
            split_id="train_only",
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 12, 31),
            gap_days=100,
        )
        assert split.val_start is None

    def test_gap_days_train_to_test_without_validation_enforced(self):
        with pytest.raises(SplitError, match="train_end and test_start"):
            SplitSpec(
                split_id="no_validation_gap",
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 1, 10),
                test_start=datetime(2020, 1, 12),
                test_end=datetime(2020, 1, 20),
                gap_days=5,
            )

    def test_gap_days_validation_to_test_enforced(self):
        with pytest.raises(SplitError, match="val_end and test_start"):
            SplitSpec(
                split_id="validation_test_gap",
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 1, 10),
                val_start=datetime(2020, 1, 16),
                val_end=datetime(2020, 1, 20),
                test_start=datetime(2020, 1, 22),
                test_end=datetime(2020, 1, 30),
                gap_days=5,
            )

    def test_gap_days_all_boundaries_exact_boundary_passes(self):
        split = SplitSpec(
            split_id="all_boundaries_exact",
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 1, 10),
            val_start=datetime(2020, 1, 16),
            val_end=datetime(2020, 1, 20),
            test_start=datetime(2020, 1, 26),
            test_end=datetime(2020, 1, 30),
            gap_days=5,
        )
        assert (split.val_start - split.train_end).days == 6
        assert (split.test_start - split.val_end).days == 6

    def test_gap_days_negative_not_allowed(self):
        with pytest.raises(SplitError, match="gap_days must be non-negative"):
            SplitSpec(
                split_id="negative_gap",
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 12, 31),
                gap_days=-5,
            )
