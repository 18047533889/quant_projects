"""Legacy class-based Gateway implementation.

This nested package is retained for backward compatibility with the original
``GatewayCore`` API.  It intentionally performs no eager wildcard imports:
submodules use their own relative dependencies, while the repository-level
``gateway`` package exposes compatibility facades.
"""

__version__ = "1.0.0"
