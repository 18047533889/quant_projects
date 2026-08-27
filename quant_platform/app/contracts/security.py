"""Security vocabulary — teams, roles, security classification.

DRAFT. PURE stdlib enums. This module has NO dependencies so it can be imported
by both ``permission.py`` and ``principal.py`` without import cycles. It is the
single canonical home for ``Role``, ``Team``, and ``SecurityClassification``.
"""

from __future__ import annotations

import enum

__all__ = [
    "Role",
    "Team",
    "SecurityClassification",
]


class SecurityClassification(enum.Enum):
    """Sensitivity tiers for factor assets (§5.8)."""

    PUBLIC_METADATA = "PUBLIC_METADATA"
    INTERNAL_RESEARCH = "INTERNAL_RESEARCH"
    CONFIDENTIAL_ALPHA = "CONFIDENTIAL_ALPHA"
    RESTRICTED_RAW_VALUES = "RESTRICTED_RAW_VALUES"
    PRODUCTION_ONLY = "PRODUCTION_ONLY"

    # Ordered by ascending sensitivity — higher rank = more restricted.
    @property
    def rank(self) -> int:
        return {
            SecurityClassification.PUBLIC_METADATA: 0,
            SecurityClassification.INTERNAL_RESEARCH: 1,
            SecurityClassification.CONFIDENTIAL_ALPHA: 2,
            SecurityClassification.RESTRICTED_RAW_VALUES: 3,
            SecurityClassification.PRODUCTION_ONLY: 4,
        }[self]


class Role(enum.Enum):
    """Roles (spec §24.1, §5.8)."""

    MEMBER = "MEMBER"
    LEAD = "LEAD"
    CORE = "CORE"
    ADMIN = "ADMIN"
    SERVICE = "SERVICE"


class Team(enum.Enum):
    """Teams (§5.8)."""

    FACTOR_TEAM = "FACTOR_TEAM"
    MODEL_TEAM = "MODEL_TEAM"
    PRODUCTION_TEAM = "PRODUCTION_TEAM"
    EXECUTIVE = "EXECUTIVE"
    PLATFORM_ADMIN = "PLATFORM_ADMIN"
