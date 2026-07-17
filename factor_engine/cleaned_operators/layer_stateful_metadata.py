from dataclasses import asdict, replace
from cleaned_operators.registry import OperatorRegistry
from stateful_contract import StatefulCheckpointRegistry


def attach_final_stateful_metadata():
    for name, spec in list(StatefulCheckpointRegistry._specs.items()):
        if name not in OperatorRegistry._operators:
            raise RuntimeError(f"missing stateful operator: {name}")
        catalog = OperatorRegistry._catalog.setdefault(name, {})
        meta = dict((catalog.get("backend_meta") or {}).get("pandas_numpy") or {})
        version = str(meta.get("semantic_version") or spec.semantic_version)
        if version != spec.semantic_version:
            spec = replace(spec, semantic_version=version)
            StatefulCheckpointRegistry._specs[name] = spec
        catalog["stateful"] = True
        catalog["segmented_execution_requires_checkpoint"] = bool(spec.checkpoint_required_for_segmented)
        catalog["checkpoint_contract"] = asdict(spec)


__all__ = ["attach_final_stateful_metadata"]
