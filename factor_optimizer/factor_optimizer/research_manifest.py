"""Bind declared COS factor values to their exact research landing record."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
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


@dataclass(frozen=True)
class BoundResearchManifest:
    """One DataAccess-verified landing manifest reusable within a bounded read."""
    manifest_uri: str
    manifest_sha256: str
    factors: object
    _verification_token: object = field(repr=False, compare=False)
    _factors_sha256: str = field(repr=False)


_MANIFEST_SNAPSHOT_TOKEN = object()


def _manifest_factors_sha256(factors):
    def normalize(value):
        if isinstance(value, datetime):
            return ["datetime", value.isoformat(timespec="microseconds"), value.fold]
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("landing manifest mapping keys must be strings")
            return ["dict", [[key, normalize(item)]
                             for key, item in sorted(value.items())]]
        if type(value) is list:
            return ["list", [normalize(item) for item in value]]
        if value is None:
            return ["none"]
        if type(value) is bool:
            return ["bool", value]
        if type(value) is int:
            return ["int", value]
        if type(value) is str:
            return ["str", value]
        raise ValueError("landing manifest contains an unsupported value type")

    normalized = normalize(factors)
    try:
        payload = json.dumps(normalized, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("landing manifest factors are not canonical") from exc
    return hashlib.sha256(payload).hexdigest()


def _declared_lineage(expression):
    """Recognize a deliberately narrow complete source-declared value grammar.

    Unknown calls/branches are unresolved, never assumed to be untreated.
    This consumes a bound producer declaration, not an execution/PIT proof.
    """
    from factor_preprocess.contracts.treatment_lineage import TransformLineage, TransformStep
    root = ast.parse(expression, mode="eval").body
    steps = []

    def scalar(node):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = scalar(node.operand)
            return (-value if isinstance(node.op, ast.USub) else value) if value is not None else None
        if isinstance(node, ast.Constant):
            value = node.value
            if ((type(value) is int and value.bit_length() <= 64)
                    or (type(value) is float and math.isfinite(value))):
                return value
        return None

    def window(node):
        value = scalar(node)
        return value if type(value) is int and 1 <= value <= 10000 else None

    def ema_parts(node):
        if len(node.args) == 2 and not node.keywords:
            value, span = node.args[0], window(node.args[1])
        elif (len(node.args) == 1 and len(node.keywords) == 1
              and node.keywords[0].arg == "span"):
            value, span = node.args[0], window(node.keywords[0].value)
        else:
            return None
        return (value, span) if span is not None else None

    while (isinstance(root, ast.Call) and isinstance(root.func, ast.Name)
           and root.func.id in {"rank", "cs_rank", "fillna_const", "ts_ema"}):
        name = root.func.id
        if name in {"rank", "cs_rank"}:
            if len(root.args) != 1 or root.keywords:
                return None
            steps.append(TransformStep("CS_RANK:pct", "representation", name))
        elif name == "fillna_const":
            if len(root.args) != 2 or root.keywords or scalar(root.args[1]) is None:
                return None
            steps.append(TransformStep("FILL:constant", "missingness", name,
                                       {"value": scalar(root.args[1])}))
        else:
            parts = ema_parts(root)
            if parts is None:
                return None
            steps.append(TransformStep("SMOOTH:ewma", "temporal", name, {"span": parts[1]}))
        root = root.args[0]

    def feature(node):
        if isinstance(node, ast.Constant):
            v = node.value
            return ((type(v) is int and v.bit_length() <= 64)
                    or (type(v) is float and math.isfinite(v)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return feature(node.operand)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return False
        if node.func.id == "ts_ema":
            parts = ema_parts(node)
            return parts is not None and feature(parts[0])
        if node.keywords:
            return False
        name, args = node.func.id, node.args
        if name == "col":
            return (len(args) == 1 and isinstance(args[0], ast.Constant)
                    and isinstance(args[0].value, str) and bool(args[0].value))
        if name in {"ts_delta", "ts_sum", "ts_pct"}:
            return len(args) == 2 and feature(args[0]) and window(args[1]) is not None
        if name == 'ts_delay':
            lag = scalar(args[1]) if len(args) == 2 else None
            return type(lag) is int and 0 <= lag <= 10000 and feature(args[0])
        if name == 'ts_std':
            period = window(args[1]) if len(args) == 2 else None
            return period is not None and period >= 2 and feature(args[0])
        if name == 'flex_max':
            # Positive integer scalars mean rolling windows in FE, not clipping.
            # Admit only the verified nonpositive scalar pointwise form here.
            bound = scalar(args[1]) if len(args) == 2 else None
            return bound is not None and bound <= 0 and feature(args[0])
        arity = {"add": 2, "subtract": 2, "multiply": 2, "divide": 2,
                 "safe_div_null": 2, "neg": 1, "abs": 1, "gt": 2,
                 "lt": 2, "maximum": 2, "where": 3}.get(name)
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
    # Keep this allowlist aligned with FactorEngine's registered rank aliases.
    # These names resolve to canonical ``rank`` in FE; overlooking one can
    # misclassify an already-ranked factor and apply a second baseline rank.
    ranks = {"rank", "cs_rank", "CS_RANK", "RANK", "c_rank", "cs_rank_01"}
    is_rank = lambda n: isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ranks
    return any(is_rank(n) for n in nodes), is_rank(tree.body)


def read_bound_manifest(store, manifest_dataset, *, manifest_params=None,
                        allow_research=False):
    """Read and bind one landing manifest for reuse across its factor reads."""
    from data_access.cos.research import read_declared_cos_object
    from data_access.cos.remote import resolve_remote_paths

    if allow_research is not True:
        raise ValueError("explicit allow_research=True required")
    store.authorize_dataset(manifest_dataset)
    store._authorize_factor_params(manifest_dataset, manifest_params)
    paths = resolve_remote_paths(store.get_dataset(manifest_dataset), params=manifest_params)
    manifest = read_declared_cos_object(store, manifest_dataset,
        params=manifest_params, allow_research=True)
    uri = paths[0].replace("s3://", "cos://", 1) if len(paths) == 1 else None
    if (not isinstance(uri, str) or manifest.source_uri != uri or
            not isinstance(manifest.content_sha256, str) or
            not re.fullmatch("[0-9a-f]{64}", manifest.content_sha256)):
        raise ValueError("landing manifest URI or content identity is invalid")
    rows = manifest.table.to_pylist()
    if len(rows) != 1 or not isinstance(rows[0].get("factors"), dict):
        raise ValueError("one landing manifest with a factors mapping is required")
    factors = rows[0]["factors"]
    return BoundResearchManifest(uri, manifest.content_sha256, factors,
        _MANIFEST_SNAPSHOT_TOKEN, _manifest_factors_sha256(factors))


def verify_bound_manifest_unchanged(store, manifest_dataset, snapshot, *,
                                    manifest_params=None):
    """Require the declared manifest identity to remain stable across a pass."""
    from data_access.cos.remote import resolve_remote_paths

    if not isinstance(snapshot, BoundResearchManifest):
        raise ValueError("manifest_snapshot must be a DataAccess-bound landing manifest")
    store.authorize_dataset(manifest_dataset)
    store._authorize_factor_params(manifest_dataset, manifest_params)
    paths = resolve_remote_paths(store.get_dataset(manifest_dataset), params=manifest_params)
    uri = paths[0].replace("s3://", "cos://", 1) if len(paths) == 1 else None
    if (snapshot._verification_token is not _MANIFEST_SNAPSHOT_TOKEN or
            snapshot.manifest_uri != uri or
            not isinstance(snapshot.manifest_sha256, str) or
            not re.fullmatch("[0-9a-f]{64}", snapshot.manifest_sha256) or
            not isinstance(snapshot.factors, dict) or
            snapshot._factors_sha256 != _manifest_factors_sha256(snapshot.factors)):
        raise ValueError("landing manifest snapshot identity is invalid")
    current = read_bound_manifest(store, manifest_dataset,
        manifest_params=manifest_params, allow_research=True)
    if (current.manifest_uri != snapshot.manifest_uri or
            current.manifest_sha256 != snapshot.manifest_sha256):
        raise ValueError("landing manifest identity changed during factor reads")
    return current


def read_bound_factor(store, manifest_dataset, factor_dataset, factor_id, *,
                      manifest_params=None, factor_params=None, allow_research=False,
                      max_object_mib=64, manifest_snapshot=None):
    """Read declared datasets through DataAccess; enforce URI, bytes and SHA256.

    Only known nonblocked research landing statuses are admitted. This is not
    PIT certification, statistical acceptance, complete treatment lineage, or
    production admission. Rank flags describe syntax scope, not economic merit.
    """
    from data_access.cos.research import read_declared_cos_object
    from data_access.cos.remote import resolve_remote_paths

    if allow_research is not True:
        raise ValueError("explicit allow_research=True required")
    if type(max_object_mib) is not int or not 1 <= max_object_mib <= 128:
        raise ValueError("max_object_mib must be an integer in 1..128")
    if not isinstance(factor_id, str) or not factor_id:
        raise ValueError("factor_id is required")
    if manifest_snapshot is None:
        manifest_snapshot = read_bound_manifest(
            store, manifest_dataset, manifest_params=manifest_params,
            allow_research=True)
    if not isinstance(manifest_snapshot, BoundResearchManifest):
        raise ValueError("manifest_snapshot must be a DataAccess-bound landing manifest")
    store.authorize_dataset(manifest_dataset)
    store._authorize_factor_params(manifest_dataset, manifest_params)
    manifest_paths = resolve_remote_paths(store.get_dataset(manifest_dataset), params=manifest_params)
    manifest_uri = (manifest_paths[0].replace("s3://", "cos://", 1)
                    if len(manifest_paths) == 1 else None)
    if (manifest_snapshot._verification_token is not _MANIFEST_SNAPSHOT_TOKEN or
            manifest_snapshot.manifest_uri != manifest_uri or
            not isinstance(manifest_snapshot.manifest_sha256, str) or
            not re.fullmatch("[0-9a-f]{64}", manifest_snapshot.manifest_sha256) or
            not isinstance(manifest_snapshot.factors, dict) or
            manifest_snapshot._factors_sha256 !=
            _manifest_factors_sha256(manifest_snapshot.factors)):
        raise ValueError("landing manifest snapshot identity is invalid")
    record = manifest_snapshot.factors.get(factor_id)
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
        params=factor_params, allow_research=True,
        max_object_mib=max_object_mib)
    if (factor.source_uri != record["uri"] or factor.content_sha256 != sha
            or factor.downloaded_bytes != size):
        raise ValueError("factor URI, content SHA256 or byte count differs from manifest")
    return BoundResearchFactor(factor_id, factor, manifest_snapshot.manifest_uri,
        manifest_snapshot.manifest_sha256, record["fe_dsl"], status, contains_rank, output_rank,
        _declared_lineage(record["fe_dsl"]))
