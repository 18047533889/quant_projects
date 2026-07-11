"""Registry for CogAlpha post-factor toolkit transforms."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TransformSpec:
    """Metadata for one deterministic cross-sectional transform."""

    name: str
    default_parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    when_to_use: str = ""
    leakage_notes: str = ""
    exposed_to_agent: bool = True


_SAME_DAY_NOTE = (
    "Uses only same-date cross-sectional values after raw single-stock factors "
    "have been assembled into a date x asset panel; it does not look across time."
)


CROSS_SECTIONAL_TRANSFORMS: dict[str, TransformSpec] = {
    "none": TransformSpec(
        name="none",
        description="Leave the raw factor panel unchanged.",
        when_to_use="Use when raw factor scale and sign already carry the intended signal.",
        leakage_notes="No transform is applied.",
    ),
    "cs_rank": TransformSpec(
        name="cs_rank",
        description="Convert each date's factor values to cross-sectional percentile ranks.",
        when_to_use="Use when only relative ordering matters and raw magnitude is noisy.",
        leakage_notes=_SAME_DAY_NOTE,
    ),
    "cs_zscore": TransformSpec(
        name="cs_zscore",
        description="Demean and divide by same-date cross-sectional standard deviation.",
        when_to_use="Use when magnitude is meaningful but should be standardized each day.",
        leakage_notes=_SAME_DAY_NOTE,
    ),
    "cs_robust_zscore": TransformSpec(
        name="cs_robust_zscore",
        description="Median/MAD based same-date robust z-score.",
        when_to_use="Use when outliers are common but signed magnitude still matters.",
        leakage_notes=_SAME_DAY_NOTE,
    ),
    "cs_winsorize_rank": TransformSpec(
        name="cs_winsorize_rank",
        default_parameters={"lower": 0.01, "upper": 0.99},
        description="Winsorize each date at 1%/99%, then rank cross-sectionally.",
        when_to_use="Use when extreme raw values are unstable but ordering is useful.",
        leakage_notes=_SAME_DAY_NOTE,
    ),
    "cs_winsorize_zscore": TransformSpec(
        name="cs_winsorize_zscore",
        default_parameters={"lower": 0.01, "upper": 0.99},
        description="Winsorize each date at 1%/99%, then cross-sectional z-score.",
        when_to_use="Use when signed magnitude matters after clipping extreme tails.",
        leakage_notes=_SAME_DAY_NOTE,
    ),
    "cs_rank_gauss": TransformSpec(
        name="cs_rank_gauss",
        default_parameters={"clip": 1e-4},
        description="Map same-date percentile ranks to an approximately Gaussian score.",
        when_to_use="Use when downstream metrics benefit from smoother normalized ranks.",
        leakage_notes=_SAME_DAY_NOTE,
    ),
    "cs_quantile_bucket": TransformSpec(
        name="cs_quantile_bucket",
        default_parameters={"quantiles": 10},
        description="Assign each date's values to integer buckets 1 through 10.",
        when_to_use="Use for deliberately coarse cross-sectional state signals.",
        leakage_notes=_SAME_DAY_NOTE,
    ),
    "cs_winsorize": TransformSpec(
        name="cs_winsorize",
        default_parameters={"lower": 0.01, "upper": 0.99},
        description="Clip each date's values at same-date quantiles.",
        when_to_use="Internal primitive for fixed exposed transforms.",
        leakage_notes=_SAME_DAY_NOTE,
        exposed_to_agent=False,
    ),
    "cs_demean": TransformSpec(
        name="cs_demean",
        description="Subtract each date's cross-sectional mean.",
        when_to_use="Internal primitive for demeaning without scaling.",
        leakage_notes=_SAME_DAY_NOTE,
        exposed_to_agent=False,
    ),
    "cs_scale": TransformSpec(
        name="cs_scale",
        description="Scale each date by the sum of absolute same-date values.",
        when_to_use="Internal primitive for L1-like cross-sectional normalization.",
        leakage_notes=_SAME_DAY_NOTE,
        exposed_to_agent=False,
    ),
    "cs_minmax": TransformSpec(
        name="cs_minmax",
        description="Scale each date into the 0 to 1 range.",
        when_to_use="Internal primitive when bounded magnitude is desired.",
        leakage_notes=_SAME_DAY_NOTE,
        exposed_to_agent=False,
    ),
    "cs_signed_power": TransformSpec(
        name="cs_signed_power",
        default_parameters={"exponent": 0.5},
        description="Compress magnitude by signed square root while preserving sign.",
        when_to_use="Internal primitive for reducing tail dominance.",
        leakage_notes=_SAME_DAY_NOTE,
        exposed_to_agent=False,
    ),
}

CROSS_SECTIONAL_TRANSFORM_NAMES = tuple(CROSS_SECTIONAL_TRANSFORMS)


def _cross_sectional_transforms_disabled() -> bool:
    """Return whether agent-exposed cross-sectional transforms are disabled."""

    value = os.environ.get("COGALPHA_DISABLE_CROSS_SECTIONAL_TRANSFORMS", "0")
    return value.lower() in {"1", "true", "yes", "on"}


EXPOSED_CROSS_SECTIONAL_TRANSFORM_NAMES = (
    ("none",)
    if _cross_sectional_transforms_disabled()
    else tuple(
        name
        for name, spec in CROSS_SECTIONAL_TRANSFORMS.items()
        if spec.exposed_to_agent
    )
)


def get_transform_spec(name: str) -> TransformSpec:
    """Return metadata for a registered transform."""

    try:
        return CROSS_SECTIONAL_TRANSFORMS[name]
    except KeyError as exc:
        raise ValueError(f"unknown cross-sectional transform: {name!r}") from exc


def is_allowed_transform(name: str, *, exposed_only: bool = True) -> bool:
    """Return whether a transform name is registered and optionally agent-exposed."""

    spec = CROSS_SECTIONAL_TRANSFORMS.get(name)
    if spec is None:
        return False
    if not exposed_only:
        return True
    if _cross_sectional_transforms_disabled():
        return name == "none"
    return bool(spec.exposed_to_agent)
