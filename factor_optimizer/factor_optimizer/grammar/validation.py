"""MutationValidator: validate mutation proposals against grammar and FE legality."""

from typing import Dict, Any, Optional, Tuple, List
from ..contracts.candidate_mutation import CandidateMutation
from .registry import MutationRegistry, get_mutation_registry


class ValidationResult:
    """Result of mutation validation."""

    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.metadata = metadata or {}

    def __bool__(self) -> bool:
        return self.is_valid

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class MutationValidator:
    """
    Validates mutation proposals against grammar and optionally FE legality.

    This validator checks:
    1. Grammar: mutation type exists and parameters are valid
    2. Parent count: correct number of parent factors
    3. FE legality (optional): through adapter protocol
    """

    def __init__(self, registry: Optional[MutationRegistry] = None, fe_adapter=None):
        """
        Initialize validator.

        Args:
            registry: Mutation registry (uses global if None)
            fe_adapter: Optional FE adapter for legality checks
        """
        self.registry = registry or get_mutation_registry()
        self.fe_adapter = fe_adapter

    def validate(self, mutation: CandidateMutation) -> ValidationResult:
        """
        Validate a mutation proposal.

        Args:
            mutation: CandidateMutation to validate

        Returns:
            ValidationResult with is_valid flag and error/warning messages
        """
        errors = []
        warnings = []
        metadata = {}

        # 1. Check mutation type exists
        spec = self.registry.get(mutation.mutation_type)
        if spec is None:
            errors.append(f"Unknown mutation type: {mutation.mutation_type}")
            return ValidationResult(False, errors, warnings, metadata)

        metadata["mutation_spec_version"] = spec.version

        # 2. Check version compatibility
        if mutation.mutation_spec_version != spec.version:
            warnings.append(
                f"Mutation spec version mismatch: "
                f"mutation={mutation.mutation_spec_version}, registry={spec.version}"
            )

        # 3. Validate parent count
        parent_count = len(mutation.parent_factor_ids)
        if spec.requires_single_parent and parent_count != 1:
            errors.append(
                f"Mutation type {mutation.mutation_type} requires exactly 1 parent, got {parent_count}"
            )
        if spec.requires_multiple_parents and parent_count < 2:
            errors.append(
                f"Mutation type {mutation.mutation_type} requires 2+ parents, got {parent_count}"
            )

        # 4. Validate parameters
        is_valid, param_errors = spec.validate_parameters(mutation.parameters)
        if not is_valid:
            errors.extend(param_errors)

        # 5. Optional: FE legality check via adapter
        if self.fe_adapter is not None and len(errors) == 0:
            try:
                legality_result = self._check_fe_legality(mutation, spec)
                metadata["fe_legality"] = legality_result
                if not legality_result.get("is_legal", False):
                    errors.append(f"FE legality check failed: {legality_result.get('reason', 'unknown')}")
            except Exception as e:
                warnings.append(f"FE legality check error: {str(e)}")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, metadata)

    def _check_fe_legality(self, mutation: CandidateMutation, spec) -> Dict[str, Any]:
        """
        Check FE legality through adapter protocol.

        This is a placeholder for the adapter interface.
        Real implementation requires FE adapter with:
        - get_parent_definitions(factor_ids) -> definitions
        - validate_mutation(mutation, spec, parent_defs) -> result
        """
        if not hasattr(self.fe_adapter, "validate_mutation"):
            return {"is_legal": True, "reason": "FE adapter not available, skipping check"}

        return self.fe_adapter.validate_mutation(mutation, spec)

    def validate_batch(self, mutations: List[CandidateMutation]) -> Dict[str, ValidationResult]:
        """
        Validate multiple mutations.

        Args:
            mutations: List of CandidateMutation objects

        Returns:
            Dictionary mapping mutation_id to ValidationResult
        """
        results = {}
        for mutation in mutations:
            results[mutation.mutation_id] = self.validate(mutation)
        return results

    def quick_check(self, mutation_type: str, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Quick parameter validation without full mutation object.

        Args:
            mutation_type: Mutation type identifier
            parameters: Parameter dictionary

        Returns:
            (is_valid, error_messages)
        """
        spec = self.registry.get(mutation_type)
        if spec is None:
            return False, [f"Unknown mutation type: {mutation_type}"]

        is_valid, errors = spec.validate_parameters(parameters)
        return is_valid, errors
