"""Bind declared COS factor values to their exact research landing record."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class BoundResearchFactor:
    factor_id: str
    factor: object
    manifest_uri: str
    manifest_sha256: str
    expression: str
    source_status: str
    contains_cs_rank: bool
    output_is_cs_rank: bool
    lineage: object

    @property
    def treatment_signature(self):
        """Carry positive rank evidence even when full DSL lineage is unknown.

        A rank inside a branch vetoes duplicate ranking; it does not prove that
        the whole output is ranked or that all treatments are understood.
        """
        from factor_preprocess.contracts.treatment_lineage import (
            ExistingTreatmentSignature, build_signature_from_lineage,
        )
        if self.lineage is not None:
            return build_signature_from_lineage(self.lineage)
        return ExistingTreatmentSignature(cs_rank=self.contains_cs_rank, status='incomplete')


def _declared_lineage(expression):
    """Recognize a deliberately narrow complete source-declared value grammar.

    Unknown calls/branches are unresolved, never assumed to be untreated.
    This consumes a bound producer declaration, not an execution/PIT proof.
    """
    from factor_preprocess.contracts.treatment_lineage import TransformLineage, TransformStep
    root = ast.parse(expression, mode="eval").body
    steps = []
    while isinstance(root, ast.Call) and isinstance(root.func, ast.Name) and root.func.id in {"rank", "cs_rank"}:
        if len(root.args) != 1 or root.keywords:
            return None
        steps.append(TransformStep("CS_RANK:pct", "representation", root.func.id))
        root = root.args[0]

    def feature(node):
        if isinstance(node, ast.Constant):
            v = node.value
            return ((type(v) is int and v.bit_length() <= 64)
                    or (type(v) is float and math.isfinite(v)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return feature(node.operand)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
            return False
        name, args = node.func.id, node.args
        if name == "col":
            return (len(args) == 1 and isinstance(args[0], ast.Constant)
                    and isinstance(args[0].value, str) and bool(args[0].value))
        if name == "ts_delta":
            return (len(args) == 2 and feature(args[0]) and isinstance(args[1], ast.Constant)
                    and type(args[1].value) is int and 1 <= args[1].value <= 10000)
        arity = {"add": 2, "subtract": 2, "multiply": 2, "divide": 2,
                 "safe_div_null": 2, "neg": 1, "abs": 1}.get(name)
        return arity is not None and len(args) == arity and all(feature(a) for a in args)
    try:
        return TransformLineage(tuple(reversed(steps))) if feature(root) else None
    except RecursionError:
        return None


def _rank_scope(expression):
    # Parse only; never eval/exec source_python, fe_dsl or metadata instructions.
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 65536:
        raise ValueError("bounded, nonempty fe_dsl is required")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("invalid fe_dsl expression") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 8192:
        raise ValueError("fe_dsl exceeds syntax budget")
    allowed = (ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant,
               ast.keyword, ast.List, ast.Tuple, ast.UnaryOp, ast.USub, ast.UAdd)
    if any(not isinstance(n, allowed) for n in nodes):
        raise ValueError("unsupported fe_dsl syntax; no code is executed")
    ranks = {"rank", "cs_rank"}
    is_rank = lambda n: isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ranks
    return any(is_rank(n) for n in nodes), is_rank(tree.body)


def read_bound_factor(store, manifest_dataset, factor_dataset, factor_id, *,
                      manifest_params=None, factor_params=None, allow_research=False):
    """Read declared datasets through DataAccess; enforce URI, bytes and SHA256.

    Only known nonblocked research landing statuses are admitted. This is not
    PIT certification, statistical acceptance, complete treatment lineage, or
    production admission. Rank flags describe syntax scope, not economic merit.
    """
    from data_access.cos.research import read_declared_cos_object
    from data_access.cos.remote import resolve_remote_paths

    if allow_research is not True:
        raise ValueError("explicit allow_research=True required")
    if not isinstance(factor_id, str) or not factor_id:
        raise ValueError("factor_id is required")
    manifest = read_declared_cos_object(store, manifest_dataset,
        params=manifest_params, allow_research=True)
    rows = manifest.table.to_pylist()
    if len(rows) != 1 or not isinstance(rows[0].get("factors"), dict):
        raise ValueError("one landing manifest with a factors mapping is required")
    record = rows[0]["factors"].get(factor_id)
    if not isinstance(record, dict) or record.get("verified") is not True:
        raise ValueError("factor record is missing or not verified")
    status = record.get("status")
    if not isinstance(status, str) or status not in {"materialized_not_evaluated", "evaluated_optimization_pending"}:
        raise ValueError("factor landing status is blocked or unsupported")
    sha = record.get("sha256")
    if not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{64}", sha):
        raise ValueError("manifest factor SHA256 is missing or invalid")
    size = record.get("bytes")
    if type(size) is not int or size <= 0:
        raise ValueError("manifest factor byte count must be a positive integer")
    contains_rank, output_rank = _rank_scope(record.get("fe_dsl"))
    # Do not let untrusted metadata choose an arbitrary download destination.
    # The caller's registered dataset, parameters and authorization remain in control.
    store.authorize_dataset(factor_dataset)
    store._authorize_factor_params(factor_dataset, factor_params)
    paths = resolve_remote_paths(store.get_dataset(factor_dataset), params=factor_params)
    uri = paths[0].replace("s3://", "cos://", 1) if len(paths) == 1 else None
    if not isinstance(record.get("uri"), str) or uri != record["uri"]:
        raise ValueError("declared factor URI does not match manifest")
    factor = read_declared_cos_object(store, factor_dataset,
        params=factor_params, allow_research=True)
    if (factor.source_uri != record["uri"] or factor.content_sha256 != sha
            or factor.downloaded_bytes != size):
        raise ValueError("factor URI, content SHA256 or byte count differs from manifest")
    return BoundResearchFactor(factor_id, factor, manifest.source_uri,
        manifest.content_sha256, record["fe_dsl"], status, contains_rank, output_rank,
        _declared_lineage(record["fe_dsl"]))
