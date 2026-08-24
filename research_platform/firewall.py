"""Alpha-generation firewall: a 10-stage candidate validation chain.

Pure standard-library module (``dataclasses``, ``enum``, ``hashlib``,
``typing``, ``datetime``).

Security / provenance note
--------------------------
A :class:`FactorCandidateArtifact` stores ONLY the *hypothesis* (the
research rationale), the *operations* (formula + mutation operator), and full
*provenance* (generator, prompts used, tool versions, seeds, retrieval
sources). It deliberately does NOT store chain-of-thought or any private
reasoning trace. The ``prompt_hash`` / ``system_prompt_hash`` fields let us
fingerprint *which* prompt was used without persisting its content.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .artifacts import Artifact, _normalize_dt, utcnow


@dataclass(frozen=True)
class FactorCandidateArtifact(Artifact):
    """A generated factor candidate that has not yet passed the firewall.

    Provenance-only: no chain-of-thought is stored, by design.
    """

    candidate_id: str = ""
    generator_type: str = ""
    generator_version: str = ""
    formula: str = ""
    hypothesis: str = ""  # the research rationale, NOT cot
    parent_factor_refs: tuple = ()
    mutation_operator: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    prompt_hash: str = ""
    system_prompt_hash: str = ""
    temperature: float = 0.0
    seed: int = 0
    retrieval_sources: tuple = ()
    tool_versions: tuple = ()
    proposed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposed_at", _normalize_dt(self.proposed_at))


# ---------------------------------------------------------------------------
# Firewall stages
# ---------------------------------------------------------------------------
class FirewallStage(Enum):
    GRAMMAR_LEGALITY = "grammar_legality"
    TYPE_DIMENSION = "type_dimension"
    FIELD_AVAILABILITY = "field_availability"
    PIT_FUTURE_LEAKAGE_STATIC = "pit_future_leakage_static"
    COMPLEXITY_BUDGET = "complexity_budget"
    FORMULA_EXACT_DUPLICATE = "formula_exact_duplicate"
    RESULT_IDENTITY_DUPLICATE = "result_identity_duplicate"
    CHEAP_FINGERPRINT_NOVELTY = "cheap_fingerprint_novelty"
    COMPUTE_COST_ESTIMATE = "compute_cost_estimate"
    FE_COMPILATION = "fe_compilation"


@dataclass(frozen=True)
class FirewallResult:
    """Outcome of a single firewall stage for one candidate."""

    stage: FirewallStage
    passed: bool
    reason: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def stage_name(self) -> str:
        return self.stage.value


class _ValidationContext:
    """Opaque context handed to firewall stages.

    Carries the raw JSON-ish dict a caller provides plus lazily-derived
    auxiliary maps (known formulas, known factor ids, available fields, ...).
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data: Dict[str, Any] = data or {}
        self._derived: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.data


@dataclass
class AlphaGenerationFirewall:
    """Configurable 10-stage validation chain for candidate factors.

    Stages run in order and all are evaluated (not short-circuited) so a
    candidate gets a full report. The final stage (``fe_compilation``) defaults
    to passing True but can be injected with a real compiler predicate via the
    ``compile_check`` callable.
    """

    # (optional) known state used by duplicate/novelty stages.
    known_formulas: Optional[set] = None
    known_fingerprints: Optional[set] = None
    known_result_identities: Optional[set] = None
    available_fields: Optional[set] = None
    complexity_budget: int = 64
    compute_cost_budget: float = 1e6
    # default True; inject to run a real compile
    compile_checker: Optional[Callable[[str], bool]] = None
    compile_pass: bool = True  # default compile result if no checker injected

    def check(
        self,
        candidate: FactorCandidateArtifact,
        context: Optional[_ValidationContext] = None,
    ) -> List[FirewallResult]:
        """Run all 10 stages and return the full report (not short-circuited)."""
        ctx = context or _ValidationContext()
        ctx.data.setdefault("candidate", candidate)
        results: List[FirewallResult] = [
            self._grammar_legality(candidate),
            self._type_dimension(candidate),
            self._field_availability(candidate, ctx),
            self._pit_future_leakage_static(candidate),
            self._complexity_budget(candidate),
            self._formula_exact_duplicate(candidate, ctx),
            self._result_identity_duplicate(candidate, ctx),
            self._cheap_fingerprint_novelty(candidate, ctx),
            self._compute_cost_estimate(candidate, ctx),
            self._fe_compilation(candidate),
        ]
        return results

    def check_and_report(self, candidate: FactorCandidateArtifact,
                         context: Optional[_ValidationContext] = None) -> Dict[str, Any]:
        results = self.check(candidate, context)
        passed_all = all(r.passed for r in results)
        return {
            "passed": passed_all,
            "passed_stages": [r.stage_name for r in results if r.passed],
            "failed_stages": [r.stage_name for r in results if not r.passed],
            "results": [r.__dict__ | {"stage": r.stage_name} for r in results],
        }

    # ------------------------------------------------------------------
    # Individual stages
    # ------------------------------------------------------------------
    @staticmethod
    def _grammar_legality(cand: FactorCandidateArtifact) -> FirewallResult:
        ok = bool(cand.formula.strip()) and _balanced_parens(cand.formula)
        reason = "empty formula or unbalanced parentheses"
        return FirewallResult(FirewallStage.GRAMMAR_LEGALITY, ok,
                              "" if ok else reason)

    @staticmethod
    def _type_dimension(cand: FactorCandidateArtifact) -> FirewallResult:
        # We only check that a formula exists and isn't trivially a scalar.
        # A real type checker is out of scope for the skeleton; documented.
        ok = "=>" not in cand.formula and len(cand.formula) >= 3
        return FirewallResult(FirewallStage.TYPE_DIMENSION, ok,
                              "" if ok else "formula too short or ill-typed")

    def _field_availability(self, cand, ctx) -> FirewallResult:
        missing = []
        if self.available_fields:
            for f in _referenced_fields(cand.formula):
                if f not in self.available_fields:
                    missing.append(f)
        ok = not missing
        return FirewallResult(FirewallStage.FIELD_AVAILABILITY, ok,
                              f"missing fields: {missing}" if missing else "")

    @staticmethod
    def _pit_future_leakage_static(cand) -> FirewallResult:
        # Static heuristic: reject clearly forward-looking tokens in formula.
        leak_tokens = ("shift", "-1", "future", "fwd", "lead")
        found = [t for t in leak_tokens if t in cand.formula.lower()]
        ok = not found
        return FirewallResult(FirewallStage.PIT_FUTURE_LEAKAGE_STATIC, ok,
                              f"suspicious tokens: {found}" if found else "")

    def _complexity_budget(self, cand) -> FirewallResult:
        cost = _structural_complexity(cand.formula)
        ok = cost <= self.complexity_budget
        return FirewallResult(FirewallStage.COMPLEXITY_BUDGET, ok,
                              f"complexity {cost} > {self.complexity_budget}" if not ok else "")

    def _formula_exact_duplicate(self, cand, ctx) -> FirewallResult:
        known = self.known_formulas if hasattr(self, "known_formulas") else (
            ctx.get("known_formulas") or set()
        )
        if known is None:
            return FirewallResult(FirewallStage.FORMULA_EXACT_DUPLICATE, True,
                                  "no known-formula set provided; assumed novel")
        ok = cand.formula not in known
        return FirewallResult(FirewallStage.FORMULA_EXACT_DUPLICATE, ok,
                              "" if ok else "exact formula duplicate")

    def _result_identity_duplicate(self, cand, ctx) -> FirewallResult:
        known = self.known_result_identities if hasattr(self, "known_result_identities") else (
            ctx.get("known_result_identities") or set()
        )
        ident = _result_identity(cand)
        if known is None:
            return FirewallResult(FirewallStage.RESULT_IDENTITY_DUPLICATE, True,
                                  "no known identities; assumed novel")
        ok = ident not in known
        return FirewallResult(FirewallStage.RESULT_IDENTITY_DUPLICATE, ok,
                              "" if ok else "behaviorally identical to an admitted factor")

    def _cheap_fingerprint_novelty(self, cand, ctx) -> FirewallResult:
        known = self.known_fingerprints if hasattr(self, "known_fingerprints") else (
            ctx.get("known_fingerprints") or set()
        )
        fp = _fingerprint(cand)
        if known is None:
            return FirewallResult(FirewallStage.CHEAP_FINGERPRINT_NOVELTY, True,
                                  "no fingerprint set; assumed novel")
        ok = fp not in known
        return FirewallResult(FirewallStage.CHEAP_FINGERPRINT_NOVELTY, ok,
                              "" if ok else "cheap fingerprint already seen")

    def _compute_cost_estimate(self, cand, ctx) -> FirewallResult:
        est = _estimate_cost(cand.formula)
        ok = est <= self.compute_cost_budget
        return FirewallResult(FirewallStage.COMPUTE_COST_ESTIMATE, ok,
                              f"estimated cost {est} > budget {self.compute_cost_budget}"
                              if not ok else "")

    def _fe_compilation(self, cand) -> FirewallResult:
        if self.compile_checker is not None:
            try:
                ok = bool(self.compile_checker(cand.formula))
                return FirewallResult(FirewallStage.FE_COMPILATION, ok,
                                      "" if ok else "fe compiler rejected formula")
            except Exception as exc:  # noqa: BLE001
                return FirewallResult(FirewallStage.FE_COMPILATION, False,
                                      f"fe compilation error: {exc}")
        return FirewallResult(FirewallStage.FE_COMPILATION, self.compile_pass,
                              "compile checker not injected; defaulted")


# ---------------------------------------------------------------------------
# Small helpers (documented as heuristics, not full checks)
# ---------------------------------------------------------------------------
def _balanced_parens(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _referenced_fields(formula: str) -> List[str]:
    # naive: every "f_..."-style token is treated as a field reference
    out: List[str] = []
    for tok in _tokens(formula):
        if tok.startswith("f_"):
            out.append(tok)
    return out


def _tokens(formula: str) -> List[str]:
    import re
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)


def _result_identity(cand: FactorCandidateArtifact) -> str:
    return hashlib.sha256(
        (cand.generator_type + "|" + cand.formula).encode("utf-8")
    ).hexdigest()


def _fingerprint(cand: FactorCandidateArtifact) -> str:
    return hashlib.sha256(
        (cand.formula + "|" + cand.mutation_operator).encode("utf-8")
    ).hexdigest()


def _estimate_compute(formula: str) -> float:
    return float(len(formula))


_estimate_cost = _estimate_compute


def _structural_complexity(formula: str) -> int:
    """Cheap structural-complexity heuristic: token count + nesting depth."""
    depth = 0
    max_depth = 0
    tokens = 1
    for ch in formula:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch in " +-*/,.":
            tokens += 1
    return tokens + max_depth
