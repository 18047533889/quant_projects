"""A-share LQTP platform client kit.

Typical usage::

    from ashare_lqtp import LqtpTokenManager, run_factor_formula, DEFAULT_SERVER

    mgr = LqtpTokenManager.login(DEFAULT_SERVER, user, password)
    resp = run_factor_formula(
        token=mgr.token,
        formula="close / delay(close, 1) - 1",
        begin_date=20240102,
        end_date=20240131,
    )
"""

from ashare_lqtp.client import (  # noqa: F401
    DEFAULT_BACKTEST_CASH,
    DEFAULT_SERVER,
    GRPC_CHANNEL_OPTIONS,
    LqtpAuth,
    LqtpTokenManager,
    analysis_to_dict,
    diagnose_backtest_nav_health,
    env_password,
    env_username,
    factor_values_to_long_df,
    fetch_lqtp_universe,
    login,
    login_with_retry,
    make_channel,
    require_credentials,
    run_backtest_from_weights,
    run_factor_formula,
    run_topk_backtest_for_long_df,
    save_backtest_rows,
    summarize_backtest,
    top_quantile_weights,
)
from ashare_lqtp.dsl_compat import (  # noqa: F401
    eval_route_for_entry,
    is_lqtp_native_dsl,
)

__all__ = [
    "DEFAULT_BACKTEST_CASH",
    "DEFAULT_SERVER",
    "GRPC_CHANNEL_OPTIONS",
    "LqtpAuth",
    "LqtpTokenManager",
    "analysis_to_dict",
    "diagnose_backtest_nav_health",
    "env_password",
    "env_username",
    "eval_route_for_entry",
    "factor_values_to_long_df",
    "fetch_lqtp_universe",
    "is_lqtp_native_dsl",
    "login",
    "login_with_retry",
    "make_channel",
    "require_credentials",
    "run_backtest_from_weights",
    "run_factor_formula",
    "run_topk_backtest_for_long_df",
    "save_backtest_rows",
    "summarize_backtest",
    "top_quantile_weights",
]

__version__ = "0.1.1"
