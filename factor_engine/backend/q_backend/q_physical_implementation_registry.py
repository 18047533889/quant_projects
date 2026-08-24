"""Evidence-backed registry for executable q physical implementations."""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_PARITY_TOLERANCE = 1e-8


@dataclass(frozen=True)
class QEvidenceArtifact:
    """One immutable JSON evidence artifact, addressed relative to an approved root."""

    path: str
    sha256: str


@dataclass(frozen=True)
class QEvidenceValidationContext:
    """Runtime environment against which q production evidence is validated."""

    artifact_root: Path
    q_version: str | None
    pykx_version: str | None
    current_git_sha_provider: Callable[[], str | None]


def current_q_git_sha() -> str | None:
    """Read the live repository HEAD; inability to do so denies certification."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[3]),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def compute_q_implementation_hash(
    canonical: str,
    lowering_id: str,
    lowering_source: str,
    parameter_domain_id: str | None,
) -> str:
    """Bind certification to the actual executable lowering and parameter domain."""

    payload = json.dumps(
        {
            "canonical": canonical,
            "lowering_id": lowering_id,
            "lowering_source": lowering_source,
            "parameter_domain_id": parameter_domain_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )



def _stage_evidence_errors(
    label: str,
    stage: str,
    payload: Mapping[str, object],
) -> list[str]:
    """Validate typed executable results, not a metadata-only PASS claim."""

    errors: list[str] = []
    expected_type = f"q_{stage}_evidence/v1"
    if payload.get("evidence_type") != expected_type:
        errors.append(f"{label}: evidence_type must be {expected_type}")

    result = payload.get("result")
    if not isinstance(result, dict):
        return [*errors, f"{label}: result must be an object"]

    if stage == "compile":
        if result.get("compiled") is not True:
            errors.append(f"{label}: result.compiled must be true")
        if not _positive_int(result.get("compiled_cases")):
            errors.append(f"{label}: result.compiled_cases must be positive")
        failure_field = "failure_count"
    elif stage == "runtime":
        if result.get("executed") is not True:
            errors.append(f"{label}: result.executed must be true")
        if not _positive_int(result.get("executed_cases")):
            errors.append(f"{label}: result.executed_cases must be positive")
        failure_field = "failure_count"
    elif stage == "parity":
        if result.get("reference_backend") != "pandas":
            errors.append(f"{label}: result.reference_backend must be pandas")
        if not _positive_int(result.get("compared_cases")):
            errors.append(f"{label}: result.compared_cases must be positive")
        failure_field = "mismatch_count"
        max_abs_error = result.get("max_abs_error")
        tolerance = result.get("tolerance")
        if not _finite_nonnegative(max_abs_error):
            errors.append(f"{label}: result.max_abs_error must be finite and nonnegative")
        if not _finite_nonnegative(tolerance):
            errors.append(f"{label}: result.tolerance must be finite and nonnegative")
        elif float(tolerance) > _MAX_PARITY_TOLERANCE:
            errors.append(f"{label}: result.tolerance exceeds policy maximum")
        if (
            _finite_nonnegative(max_abs_error)
            and _finite_nonnegative(tolerance)
            and float(max_abs_error) > float(tolerance)
        ):
            errors.append(f"{label}: result.max_abs_error exceeds tolerance")
    elif stage == "null_semantics":
        if not _positive_int(result.get("checked_cases")):
            errors.append(f"{label}: result.checked_cases must be positive")
        checks = result.get("checks")
        required_checks = {"null-mask", "warmup", "all-null-window"}
        if (
            not isinstance(checks, list)
            or not checks
            or any(not isinstance(check, str) or not check for check in checks)
            or not required_checks.issubset(checks)
        ):
            errors.append(
                f"{label}: result.checks must include null-mask, warmup, and all-null-window"
            )
        failure_field = "mismatch_count"
    else:
        return [*errors, f"{label}: unsupported evidence stage"]

    failures = result.get(failure_field)
    if not _nonnegative_int(failures):
        errors.append(f"{label}: result.{failure_field} must be nonnegative")
    elif failures != 0:
        errors.append(f"{label}: result.{failure_field} must be zero")
    return errors



def _artifact_errors(
    label: str,
    artifact: QEvidenceArtifact | None,
    root: Path,
    expected: Mapping[str, str],
) -> list[str]:
    if artifact is None:
        return [f"{label}: missing"]
    if not isinstance(artifact, QEvidenceArtifact):
        return [f"{label}: expected QEvidenceArtifact"]
    if not _SHA256_RE.fullmatch(artifact.sha256):
        return [f"{label}: invalid sha256"]
    relative = Path(artifact.path)
    if relative.is_absolute():
        return [f"{label}: path must be relative"]
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return [f"{label}: path escapes artifact root"]
    try:
        if not path.is_file():
            return [f"{label}: artifact is missing or not a file"]
        raw = path.read_bytes()
    except OSError as exc:
        return [f"{label}: artifact read failed: {type(exc).__name__}"]
    if hashlib.sha256(raw).hexdigest() != artifact.sha256:
        return [f"{label}: artifact sha256 mismatch"]
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [f"{label}: invalid JSON payload"]
    if not isinstance(payload, dict):
        return [f"{label}: payload must be an object"]
    errors = []
    try:
        generated = datetime.fromisoformat(str(payload.get("generation_timestamp", "")))
        if generated.tzinfo is None or generated.utcoffset() is None:
            errors.append(f"{label}: generation_timestamp timezone is required")
    except (TypeError, ValueError):
        errors.append(f"{label}: generation_timestamp missing or malformed")
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"{label}: {key} mismatch")
    if payload.get("status") != "PASS":
        errors.append(f"{label}: status is not PASS")
    errors.extend(_stage_evidence_errors(label, expected.get("stage", ""), payload))
    return errors


def _version_errors(label: str, actual: str | None, specifier: str | None) -> list[str]:
    if not specifier:
        return [f"{label}_version_range: missing"]
    if not actual:
        return [f"{label}_version: unavailable"]
    try:
        allowed = SpecifierSet(specifier)
        version = Version(actual)
    except (InvalidSpecifier, InvalidVersion):
        return [f"{label}_version: invalid version or range"]
    if version not in allowed:
        return [f"{label}_version: outside certified range"]
    return []


@dataclass(frozen=True)
class QPhysicalImplementation:
    canonical: str
    lowering_id: str
    parameter_domain_id: str | None = None
    lowering_source: str = ""
    compile_evidence: QEvidenceArtifact | None = None
    runtime_evidence: QEvidenceArtifact | None = None
    parity_evidence: QEvidenceArtifact | None = None
    null_semantics_evidence: QEvidenceArtifact | None = None
    performance_evidence: QEvidenceArtifact | None = None
    git_sha: str | None = None
    generation_timestamp: str | None = None
    implementation_hash: str | None = None
    q_version_range: str | None = None
    pykx_version_range: str | None = None
    notes: str = ""

    @property
    def research_ready(self) -> bool:
        """Executable lowering is available for research use only."""
        return bool(self.lowering_id)

    def validation_errors(self, context: QEvidenceValidationContext) -> tuple[str, ...]:
        errors: list[str] = []
        if not _GIT_SHA_RE.fullmatch(self.git_sha or ""):
            errors.append("git_sha: missing or malformed")
        try:
            current_git_sha = context.current_git_sha_provider()
        except Exception:
            current_git_sha = None
        if not _GIT_SHA_RE.fullmatch(current_git_sha or ""):
            errors.append("current_git_sha: unavailable or malformed")
        elif self.git_sha != current_git_sha:
            errors.append("git_sha: evidence is stale")

        try:
            generated = datetime.fromisoformat(self.generation_timestamp or "")
            if generated.tzinfo is None or generated.utcoffset() is None:
                errors.append("generation_timestamp: timezone is required")
        except (TypeError, ValueError):
            errors.append("generation_timestamp: missing or malformed")

        evidence = {
            "compile_evidence": self.compile_evidence,
            "runtime_evidence": self.runtime_evidence,
            "parity_evidence": self.parity_evidence,
            "null_semantics_evidence": self.null_semantics_evidence,
        }
        paths = [artifact.path for artifact in evidence.values() if isinstance(artifact, QEvidenceArtifact)]
        if len(paths) != len(set(paths)):
            errors.append("evidence_artifacts: required artifacts must be independent")
        expected_artifact = {
            "canonical": self.canonical,
            "git_sha": self.git_sha or "",
            "implementation_hash": self.implementation_hash or "",
            "parameter_domain_id": self.parameter_domain_id or "",
            "q_version": context.q_version or "",
            "pykx_version": context.pykx_version or "",
        }
        for label, artifact in evidence.items():
            stage = label.removesuffix("_evidence")
            errors.extend(
                _artifact_errors(
                    label,
                    artifact,
                    context.artifact_root,
                    {**expected_artifact, "stage": stage},
                )
            )

        expected_hash = compute_q_implementation_hash(
            self.canonical,
            self.lowering_id,
            self.lowering_source,
            self.parameter_domain_id,
        )
        if not _SHA256_RE.fullmatch(self.implementation_hash or ""):
            errors.append("implementation_hash: missing or malformed")
        elif self.implementation_hash != expected_hash:
            errors.append("implementation_hash: lowering or parameter domain changed")
        if not self.lowering_source:
            errors.append("lowering_source: missing")
        if not self.parameter_domain_id:
            errors.append("parameter_domain: missing")

        errors.extend(_version_errors("q", context.q_version, self.q_version_range))
        errors.extend(_version_errors("pykx", context.pykx_version, self.pykx_version_range))
        return tuple(errors)


class QPhysicalImplementationRegistry:
    """Registry populated only from a compiler's executable lowering map."""

    # Admission declarations are intentionally limited to operators with an
    # executable entry in QCompiler._build_operator_map.  Unsupported names
    # are not declared native: declaring them would make the capability gap
    # look like a production admission problem and could certify no lowering.
    _DECLARED_TARGETS = frozenset({
        "abs", "add", "ceil", "cs_demean", "cs_normalize", "cs_rank", "cs_std",
        "cs_median", "cs_mean", "cs_var", "cs_zscore", "count_nonzero", "delta",
        "divide", "ema", "equal", "expm1", "exp", "ffill", "fillna", "first",
        "floor", "greater", "greater_equal", "last", "lag", "less", "less_equal",
        "log", "log1p", "max", "mean", "median", "min", "multiply", "negate",
        "not_equal", "pct_change", "power", "product", "rank", "round", "sign",
        "sma", "sqrt", "std", "subtract", "sum", "ts_corr",
        "ts_count", "ts_cov", "ts_cummax", "ts_cummin", "ts_cumprod", "ts_cumsum",
        "ts_diff", "ts_max", "ts_mean", "ts_min", "ts_returns", "ts_std", "ts_sum",
        "var", "wma", "where", "clip",
    })

    def __init__(
        self,
        *,
        lowerings: Mapping[str, str] | None = None,
        declared_targets: frozenset[str] | None = None,
        validation_context: QEvidenceValidationContext | None = None,
    ) -> None:
        self._implementations: dict[str, QPhysicalImplementation] = {}
        self._declared_targets = frozenset(
            frozenset() if declared_targets is None else declared_targets
        )
        self._validation_context = validation_context
        for canonical, lowering_id in (lowerings or {}).items():
            self.register(QPhysicalImplementation(
                canonical=canonical,
                lowering_id=lowering_id,
                lowering_source=lowering_id,
            ))

    def declared_targets(self) -> frozenset[str]:
        return self._declared_targets

    def register(self, impl: QPhysicalImplementation) -> None:
        existing = self._implementations.get(impl.canonical)
        if existing is not None and existing.lowering_id != impl.lowering_id:
            raise ValueError(
                f"Cannot register {impl.canonical}: already registered with "
                f"lowering_id={existing.lowering_id}, attempted {impl.lowering_id}"
            )
        self._implementations[impl.canonical] = impl

    def get(self, canonical: str) -> QPhysicalImplementation | None:
        return self._implementations.get(canonical)

    def evidence_errors(self, canonical: str) -> tuple[str, ...]:
        impl = self.get(canonical)
        if impl is None:
            return ("implementation: missing",)
        if self._validation_context is None:
            return ("validation_context: missing",)
        return impl.validation_errors(self._validation_context)

    def get_research_ready(self) -> frozenset[str]:
        if self.has_disagreement():
            return frozenset()
        return frozenset(name for name, impl in self._implementations.items() if impl.research_ready)

    def get_production_ready(self) -> frozenset[str]:
        if self.has_disagreement():
            return frozenset()
        return frozenset(
            name for name in self._implementations
            if self.is_production_certified(name)
        )

    def get_with_lowering(self) -> frozenset[str]:
        return frozenset(name for name, impl in self._implementations.items() if impl.lowering_id)

    def has_lowering(self, canonical: str) -> bool:
        return bool((impl := self.get(canonical)) and impl.lowering_id)

    def is_production_certified(self, canonical: str) -> bool:
        return (
            canonical in self._declared_targets
            and bool(self.get(canonical))
            and not self.evidence_errors(canonical)
        )

    def admission_disagreements(self) -> dict[str, list[str]]:
        declared = set(self.declared_targets())
        lowering = set(self.get_with_lowering())
        return {
            "declared_without_lowering": sorted(declared - lowering),
            "lowering_without_declaration": sorted(lowering - declared),
        }

    def has_disagreement(self) -> bool:
        return any(self.admission_disagreements().values())

    def get_missing_evidence(self) -> dict[str, list[str]]:
        missing: dict[str, list[str]] = {}
        for name in self._implementations:
            gaps = list(self.evidence_errors(name))
            if gaps:
                missing[name] = gaps
        return missing

    def gate_all_lowerings_have_evidence(self) -> tuple[bool, str]:
        missing = self.get_missing_evidence()
        disagreements = self.admission_disagreements()
        if self._implementations and not missing and not any(disagreements.values()):
            return True, "PASS: all lowerings have current, validated evidence"
        return False, f"FAIL: missing={list(missing)[:5]}, disagreements={disagreements}"


_REGISTRY: QPhysicalImplementationRegistry | None = None


def get_installed_q_physical_implementation_registry() -> QPhysicalImplementationRegistry | None:
    """Return an explicitly installed registry without bootstrapping one."""
    return _REGISTRY


def build_q_physical_implementation_registry(
    lowerings: Mapping[str, str],
    declared_targets: frozenset[str],
) -> QPhysicalImplementationRegistry:
    """Build without importing compiler/capability modules."""
    return QPhysicalImplementationRegistry(
        lowerings=lowerings,
        declared_targets=declared_targets,
    )


def install_q_physical_implementation_registry(
    registry: QPhysicalImplementationRegistry,
) -> QPhysicalImplementationRegistry:
    global _REGISTRY
    if _REGISTRY is not None and _REGISTRY is not registry:
        return _REGISTRY
    _REGISTRY = registry
    return registry


def get_q_physical_implementation_registry() -> QPhysicalImplementationRegistry:
    """Return compiler-derived authority, or fail closed when unavailable."""
    global _REGISTRY
    if _REGISTRY is None:
        from factor_engine.backend.q_backend.q_compiler import get_q_compiler

        compiler = get_q_compiler()
        if _REGISTRY is None:
            _REGISTRY = QPhysicalImplementationRegistry(
                lowerings=compiler.executable_lowerings(),
                declared_targets=compiler.declared_targets(),
            )
    return _REGISTRY
