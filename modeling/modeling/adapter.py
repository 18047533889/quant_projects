"""
Adapters to reuse factor_preprocess implementations.

This module provides adapters that translate modeling contracts to factor_preprocess
contracts, allowing reuse of validated implementations without copying code.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from modeling.contracts import (
    PreprocessContract,
    FitWindow,
    ModelReadyData,
    TransformMode,
)
from modeling.errors import AdapterError, FutureLeakageError

try:
    from factor_preprocess import (
        PreprocessingPolicy,
        TransformSpec,
        TransformKind,
        TransformMode as FPTransformMode,
        FittedState,
    )
    FACTOR_PREPROCESS_AVAILABLE = True
except ImportError:
    FACTOR_PREPROCESS_AVAILABLE = False


class FactorPreprocessAdapter:
    """
    Adapter to use factor_preprocess implementations via modeling contracts.

    This allows modeling to define its own contracts while reusing the
    battle-tested implementations from factor_preprocess.
    """

    def __init__(self):
        if not FACTOR_PREPROCESS_AVAILABLE:
            raise AdapterError(
                "factor_preprocess not available. Install with: pip install -e ../factor_preprocess"
            )

    def translate_contract(
        self,
        contract: PreprocessContract,
        application_period_start: Optional[datetime] = None,
    ) -> "PreprocessingPolicy":
        """
        Translate modeling PreprocessContract to factor_preprocess PreprocessingPolicy.

        Args:
            contract: Modeling preprocessing contract
            application_period_start: Start time of period where transforms will be applied

        Returns:
            PreprocessingPolicy compatible with factor_preprocess

        Raises:
            FutureLeakageError: If fit window would leak future information
            AdapterError: If contract cannot be translated
        """
        # Validate no future leakage
        if contract.mode == TransformMode.FITTED and contract.fit_window is not None:
            if application_period_start is not None:
                if not contract.fit_window.is_valid_for_application(
                    application_period_start, application_period_start
                ):
                    raise FutureLeakageError(
                        f"Fit window {contract.fit_window.fit_end} must end before "
                        f"application period {application_period_start}"
                    )

        # Translate transforms
        fp_transforms = []
        for i, transform_dict in enumerate(contract.transforms):
            fp_transform = self._translate_transform(transform_dict, i)
            fp_transforms.append(fp_transform)

        # Build PreprocessingPolicy - exact field mapping, no guessing
        fp_policy = PreprocessingPolicy(
            policy_id=f"modeling_{contract.contract_id}",
            transforms=fp_transforms,
            requires_universe=contract.requires_universe,
            requires_industry=contract.requires_industry,
            requires_size=contract.requires_size,
        )

        return fp_policy

    def _translate_transform(self, transform_dict: Dict[str, Any], index: int) -> "TransformSpec":
        """
        Translate a single transform specification.

        Raises:
            AdapterError: If kind or mode is unknown (fail-closed)
        """
        name = transform_dict.get("name")
        if not name:
            raise AdapterError(f"Transform {index} missing 'name' field")

        kind_str = transform_dict.get("kind")
        if not kind_str:
            raise AdapterError(f"Transform '{name}' missing 'kind' field")

        mode_str = transform_dict.get("mode")
        if not mode_str:
            raise AdapterError(f"Transform '{name}' missing 'mode' field")

        version = transform_dict.get("version", "0.1.0")
        parameters = transform_dict.get("parameters", {})

        # Map kind - fail-closed on unknown
        kind_mapping = {
            "cross_sectional": TransformKind.CROSS_SECTIONAL,
            "rolling": TransformKind.ROLLING,
            "neutralization": TransformKind.NEUTRALIZATION,
            "representation": TransformKind.REPRESENTATION,
        }
        if kind_str not in kind_mapping:
            raise AdapterError(
                f"Unknown transform kind '{kind_str}' for transform '{name}'. "
                f"Valid kinds: {list(kind_mapping.keys())}"
            )
        fp_kind = kind_mapping[kind_str]

        # Map mode - fail-closed on unknown
        mode_mapping = {
            "stateless": FPTransformMode.STATELESS,
            "fitted": FPTransformMode.FITTED,
        }
        if mode_str not in mode_mapping:
            raise AdapterError(
                f"Unknown transform mode '{mode_str}' for transform '{name}'. "
                f"Valid modes: {list(mode_mapping.keys())}"
            )
        fp_mode = mode_mapping[mode_str]

        return TransformSpec(
            name=name,
            kind=fp_kind,
            mode=fp_mode,
            version=version,
            parameters=parameters,
        )

    def translate_fit_window(self, fit_window: FitWindow) -> Dict[str, Any]:
        """
        Translate modeling FitWindow to factor_preprocess compatible metadata.

        Returns:
            Dictionary with fit window metadata
        """
        return {
            "fit_start_time": fit_window.fit_start,
            "fit_end_time": fit_window.fit_end,
            "fit_universe_ref": fit_window.universe_ref,
        }

    def wrap_output(
        self,
        fp_output: Any,
        contract: PreprocessContract,
        data_start: datetime,
        data_end: datetime,
    ) -> ModelReadyData:
        """
        Wrap factor_preprocess output into modeling ModelReadyData.

        Args:
            fp_output: Output from factor_preprocess (must be FeatureBundle)
            contract: Original modeling contract
            data_start: Start of data period
            data_end: End of data period

        Returns:
            ModelReadyData with proper metadata

        Raises:
            AdapterError: If fp_output is not a valid FeatureBundle
        """
        if not FACTOR_PREPROCESS_AVAILABLE:
            raise AdapterError("factor_preprocess not available")

        # Import FeatureBundle for validation
        from factor_preprocess.contracts.feature_bundle import FeatureBundle

        # Validate input type
        if not isinstance(fp_output, FeatureBundle):
            raise AdapterError(
                f"Expected FeatureBundle from factor_preprocess, got {type(fp_output).__name__}"
            )

        # Extract validated fields from FeatureBundle
        features = fp_output.features

        # Extract feature names from channels or axis
        feature_names = []
        if hasattr(fp_output, 'channels') and fp_output.channels:
            for channel in fp_output.channels:
                if hasattr(channel, 'feature_ids'):
                    feature_names.extend(channel.feature_ids)

        # Extract missing indicators if present
        missing_indicators = None
        if hasattr(fp_output, 'missing_mask'):
            missing_indicators = fp_output.missing_mask

        # Extract exposure residuals if present (from neutralization)
        exposure_residuals = None
        if hasattr(fp_output, 'exposure_residuals'):
            exposure_residuals = fp_output.exposure_residuals

        return ModelReadyData(
            features=features,
            feature_names=feature_names,
            data_start=data_start,
            data_end=data_end,
            preprocess_contract_id=contract.contract_id,
            fit_window=contract.fit_window,
            missing_indicators=missing_indicators,
            exposure_residuals=exposure_residuals,
            producer="modeling.adapter.factor_preprocess",
        )


def is_factor_preprocess_available() -> bool:
    """Check if factor_preprocess is available."""
    return FACTOR_PREPROCESS_AVAILABLE
