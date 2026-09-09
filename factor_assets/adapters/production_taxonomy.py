"""Production FE analysis → DA catalog → FA classification composition.

Economic field roles in DA (e.g. price_level) are not formula usage roles.
The latter must be supplied explicitly by the approved recipe/assembly caller.
"""
from dataclasses import replace
from types import SimpleNamespace
from collections.abc import Mapping


def classify_definition_taxonomy(dsl_or_expr, *, taxonomy_provider,
                                 field_roles: Mapping[str,str], surface="daily", policy=None):
    """Classify real FE field uses with canonical DA descriptors; never guess names.

    ``field_roles`` binds FE canonical field IDs to alpha/control/eligibility/
    weight usage in this definition. Every field must be bound, including
    controls. DA aliases are resolved before classification; conflicting roles
    that collapse to the same DA canonical ID fail closed.
    """
    from factor_engine.api.static_analysis import analyze_factor_definition
    from factor_assets.profiling.taxonomy import classify_factor_taxonomy
    from factor_assets.profiling.policies import get_taxonomy_policy
    policy = policy or get_taxonomy_policy()
    if policy.mechanism_sources != "alpha_only":
        raise ValueError("production composition requires alpha_only taxonomy policy")
    analysis = analyze_factor_definition(dsl_or_expr,surface=surface)
    usages = tuple(analysis.field_usages)
    field_ids = tuple(u.canonical_field_id for u in usages)
    if set(field_roles) != set(field_ids):
        raise ValueError("field_roles must exactly bind all FE canonical field uses")
    if any(role not in {"alpha","control","eligibility","weight"} for role in field_roles.values()):
        raise ValueError("invalid formula usage role (DA economic roles cannot substitute)")
    descriptors = taxonomy_provider.describe_fields(field_ids)
    canonical_usages, taxonomy, roles = [], {}, {}
    for usage in usages:
        descriptor = descriptors.get(usage.canonical_field_id)
        if descriptor is None or not descriptor.canonical_field_id:
            raise ValueError(f"DA did not resolve field {usage.canonical_field_id}")
        canonical = descriptor.canonical_field_id
        role = field_roles[usage.canonical_field_id]
        if canonical in roles and roles[canonical] != role:
            raise ValueError("DA alias resolution collapses incompatible formula usage roles")
        domains = tuple(descriptor.data_domains)
        # Registered but semantically unclassified fields remain explicitly
        # UNKNOWN. Unknown/unregistered names fail in the catalog itself.
        domains = domains or ("UNKNOWN",)
        if canonical in taxonomy and taxonomy[canonical] != domains:
            raise ValueError("conflicting DA semantic descriptions for one canonical field")
        taxonomy[canonical], roles[canonical] = domains, role
        canonical_usages.append(replace(usage,canonical_field_id=canonical))
    # Projection only: do not mutate FE's signed/hash-bearing analysis object.
    projection = SimpleNamespace(factor_definition_id=analysis.factor_definition_id,
        canonical_dsl_hash=analysis.canonical_dsl_hash,
        operator_usages=analysis.operator_usages,field_usages=tuple(canonical_usages),
        existing_treatment_semantic_ids=analysis.existing_treatment_semantic_ids)
    return classify_factor_taxonomy(projection,policy=policy,field_taxonomy=taxonomy,
                                    field_roles=roles,production=True)
