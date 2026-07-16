# -*- coding: utf-8
"""数学变换 domain / overflow 契约。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ExpOverflowPolicy = Literal["to_inf", "to_null", "clip"]
InverseZeroPolicy = Literal["null", "inf", "default"]
LogAbsZeroPolicy = Literal["negative_inf", "null", "log_epsilon"]
SignedLogFormula = Literal["sign_times_log_abs_plus_eps", "sign_times_log1p_abs"]
PowerDomainPolicy = Literal["backend_default_with_null_guard"]


@dataclass(frozen=True)
class PowerSpec:
    negative_base_non_integer_is_null: bool = True
    zero_negative_exponent_is_null: bool = True
    domain: PowerDomainPolicy = "backend_default_with_null_guard"


@dataclass(frozen=True)
class InverseSpec:
    zero: InverseZeroPolicy = "null"


@dataclass(frozen=True)
class LogAbsSpec:
    zero: LogAbsZeroPolicy = "negative_inf"


@dataclass(frozen=True)
class SignedLogSpec:
    """当前生产公式：``sign(x) * log(abs(x) + 1e-10)``（保留向后兼容）。"""

    formula: SignedLogFormula = "sign_times_log_abs_plus_eps"
    epsilon: float = 1e-10


@dataclass(frozen=True)
class ExpSpec:
    overflow: ExpOverflowPolicy = "to_inf"
    underflow_to_zero: bool = True


@dataclass(frozen=True)
class LogSpec:
    non_positive: Literal["null", "negative_inf"] = "null"
    domain_outside: Literal["null"] = "null"


@dataclass(frozen=True)
class OverflowPolicy:
    """定义域外 / overflow 统一策略。"""

    domain_outside: Literal["null"] = "null"
    overflow: ExpOverflowPolicy = "to_inf"
    underflow: Literal["zero", "null"] = "zero"


OVERFLOW_POLICY = OverflowPolicy()
LOG_SPEC = LogSpec()


POWER_SPEC = PowerSpec()
INVERSE_SPEC = InverseSpec()
LOG_ABS_SPEC = LogAbsSpec()
SIGNED_LOG_SPEC = SignedLogSpec()
EXP_SPEC = ExpSpec()


def signed_log_epsilon() -> float:
    return SIGNED_LOG_SPEC.epsilon


def signed_log_formula_name() -> str:
    return SIGNED_LOG_SPEC.formula


def inverse_zero_is_null() -> bool:
    return INVERSE_SPEC.zero == "null"


def exp_overflow_to_inf() -> bool:
    return EXP_SPEC.overflow == "to_inf"


def log_domain_outside_is_null() -> bool:
    return LOG_SPEC.domain_outside == "null"


def power_domain_outside_is_null() -> bool:
    return POWER_SPEC.negative_base_non_integer_is_null and POWER_SPEC.zero_negative_exponent_is_null
