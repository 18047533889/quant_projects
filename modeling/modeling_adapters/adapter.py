"""
Adapters to reuse factor_preprocess implementations.

This module provides adapters that translate modeling contracts to factor_preprocess
contracts, allowing reuse of validated implementations without copying code.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from modeling_adapters.contracts import (
    PreprocessContract,
    FitWindow,
    ModelReadyData,
    TransformMode,
)
from modeling_adapters.errors import AdapterError, FutureLeakageError

try:
    from factor_preprocess import (
        PreprocessingPolicy,
        TransformSpec,
        TransformKind,
        TransformMode as FPTransformMode,
        FittedState,
    )
    FACTOR_PREPROCESS_AVAILABLE = True
    _FACTOR_PREPROCESS_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as _exc:
    FACTOR_PREPROCESS_AVAILABLE = False
    _FACTOR_PREPROCESS_IMPORT_ERROR = _exc


class FactorPreprocessAdapter:
    """
    Adapter to use factor_preprocess implementations via modeling contracts.

    This allows modeling to define its own contracts while reusing the
    battle-tested implementations from factor_preprocess.
    """

    def __init__(self):
        if not FACTOR_PREPROCESS_AVAILABLE:
            # A nested ImportError inside factor_preprocess (e.g. a broken
            # scipy install) must not be reported as "not installed".  The
            # message of a "No module named X" names the failing module; a
            # nested failure names one of factor_preprocess's dependencies
            # while the top-level import of factor_preprocess itself is the
            # origin of the chain.
            _nested = (
                _FACTOR_PREPROCESS_IMPORT_ERROR is not None
                and str(_FACTOR_PREPROCESS_IMPORT_ERROR).strip()
                not in (
                    f"No module named 'factor_preprocess'",
                    f"No module named 'factor_preprocess'",
                )
            )
            cause = (
                f" Import failed: {_FACTOR_PREPROCESS_IMPORT_ERROR}"
                if _nested
                else ""
            )
            raise AdapterError(
                "factor_preprocess not available. "
                "Install with: pip install -e ../factor_preprocess"
                f"{cause}"
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
        # Fitted transforms must always declare the application period. Without
        # it, the temporal contract cannot distinguish OOS use from leakage.
        if contract.mode == TransformMode.FITTED:
            if contract.fit_window is None:
                raise AdapterError("FITTED contract requires fit_window")
            if application_period_start is None:
                raise AdapterError(
                    "application_period_start is required for fitted transforms"
                )
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
            "fit_snapshot_ref": fit_window.data_snapshot_ref,
        }

    @staticmethod
    def validate_fitted_state_features(state: Any, feature_ids: List[str]) -> None:
        """Require exact ordered feature compatibility for a fitted state."""
        # `is not None` semantics: getattr with a tuple default already
        # avoids the truthiness crash a numpy array would trigger on `or`.
        raw_order = getattr(state, "feature_order", None)
        if raw_order is None:
            expected: List[str] = []
        else:
            expected = list(raw_order)
        if expected and list(feature_ids) != expected:
            raise AdapterError(
                "FittedState feature order mismatch: "
                f"expected {expected}, got {list(feature_ids)}"
            )

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

        # FeatureBundle exposes its payload as `values`.
        features = fp_output.values

        # Extract feature names from the FeatureBundle channel mapping.
        feature_names = []
        for channel in fp_output.channels.values():
            if channel.channel_type == "feature":
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
            created_at=data_end,
            producer="modeling_adapters.adapter.factor_preprocess",
        )


def is_factor_preprocess_available() -> bool:
    """Check if factor_preprocess is available."""
    return FACTOR_PREPROCESS_AVAILABLE
