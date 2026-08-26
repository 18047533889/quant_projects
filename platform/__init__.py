"""Quant Research Platform — platform package.

DRAFT. The platform layer must NOT import domain packages (factor_engine,
quant_evaluator, factor_assets, …), and domain packages must never import
platform. The contracts DTO layer lives in ``platform.app.contracts``.
"""
