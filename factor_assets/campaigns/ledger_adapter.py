"""
Ledger adapter for campaign tracking.

Bridges campaigns to append-only split contamination ledger.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime


class LedgerAdapter:
    """
    Adapter for campaign-ledger integration.

    Provides high-level operations for split tracking and contamination detection.
    """

    def __init__(self, ledger):
        """
        Initialize adapter.

        Args:
            ledger: SplitContaminationLedger instance
        """
        self.ledger = ledger

    def register_campaign_splits(
        self,
        campaign_id: str,
        train_splits: List[str],
        validation_splits: List[str],
        test_splits: List[str],
    ) -> None:
        """
        Register all splits for a campaign.

        Args:
            campaign_id: Campaign identifier
            train_splits: Training split identifiers
            validation_splits: Validation split identifiers
            test_splits: Test split identifiers
        """
        for split_id in train_splits:
            self.ledger.register_split(
                split_id=split_id,
                split_type="train",
                campaign_id=campaign_id,
            )

        for split_id in validation_splits:
            self.ledger.register_split(
                split_id=split_id,
                split_type="validation",
                campaign_id=campaign_id,
            )

        for split_id in test_splits:
            self.ledger.register_split(
                split_id=split_id,
                split_type="test",
                campaign_id=campaign_id,
            )

    def record_evaluation(
        self,
        candidate_id: str,
        split_id: str,
        usage_type: str,
        campaign_id: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record candidate evaluation on a split.

        Args:
            candidate_id: Candidate identifier
            split_id: Split identifier
            usage_type: Type of usage (training, validation, testing)
            campaign_id: Campaign identifier
            metrics: Optional evaluation metrics
        """
        metadata = {
            "campaign_id": campaign_id,
            "metrics": metrics or {},
        }

        self.ledger.record_usage(
            split_id=split_id,
            candidate_id=candidate_id,
            usage_type=usage_type,
            metadata=metadata,
        )

    def check_contamination(
        self,
        candidate_id: str,
        test_split_id: str,
    ) -> bool:
        """
        Check if candidate is contaminated for test split.

        Args:
            candidate_id: Candidate identifier
            test_split_id: Test split identifier

        Returns:
            True if contamination detected
        """
        contamination = self.ledger.detect_contamination(
            candidate_id=candidate_id,
            test_split_id=test_split_id,
        )
        return contamination is not None

    def get_candidate_history(
        self,
        candidate_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get full usage history for candidate.

        Args:
            candidate_id: Candidate identifier

        Returns:
            List of usage records
        """
        return self.ledger.get_usage_history(candidate_id=candidate_id)

    def get_split_usage(
        self,
        split_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all usage records for a split.

        Args:
            split_id: Split identifier

        Returns:
            List of usage records
        """
        return self.ledger.get_split_usage(split_id=split_id)

    def get_campaign_contamination_summary(
        self,
        campaign_id: str,
    ) -> Dict[str, Any]:
        """
        Get contamination summary for campaign.

        Args:
            campaign_id: Campaign identifier

        Returns:
            Summary with contamination counts and candidates
        """
        # Get all contamination records
        all_contamination = self.ledger.list_contamination()

        # Filter by campaign
        campaign_contamination = [
            c for c in all_contamination
            if c.get("metadata", {}).get("campaign_id") == campaign_id
        ]

        # Count unique contaminated candidates
        contaminated_candidates = set(
            c["candidate_id"] for c in campaign_contamination
        )

        return {
            "campaign_id": campaign_id,
            "contamination_count": len(campaign_contamination),
            "contaminated_candidates": list(contaminated_candidates),
            "contamination_records": campaign_contamination,
        }

    def seal_test_splits(
        self,
        campaign_id: str,
        test_split_ids: List[str],
    ) -> None:
        """
        Mark test splits as sealed (no further modifications allowed).

        DLIB-FA-003 / §43: this is a LOGICAL seal — a projection/audit record
        only. It is NOT a real test-protection boundary. Test authority is FO
        TestAuthorityBroker; FA's ledger adapter is projection/audit only. No
        production code may treat this seal as real test protection.

        Args:
            campaign_id: Campaign identifier
            test_split_ids: Test split identifiers to seal
        """
        for split_id in test_split_ids:
            # Record sealing event as metadata update
            metadata = {
                "sealed": True,
                "sealed_at": datetime.utcnow().isoformat(),
                "campaign_id": campaign_id,
            }
            # This is a logical seal - actual enforcement in split ledger
            self.ledger.record_usage(
                split_id=split_id,
                candidate_id=f"__seal_{campaign_id}",
                usage_type="seal",
                metadata=metadata,
            )
