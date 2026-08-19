"""Setup for factor_assets package."""
from setuptools import setup


setup(
    name="factor_assets",
    version="0.1.0",
    description="Factor asset identity, registry, lifecycle, and governance",
    author="Quant Platform Team",
    python_requires=">=3.10",
    install_requires=[
        "typing-extensions>=4.5.0",
        "dataclasses-json>=0.5.13",
    ],
    package_dir={"factor_assets": "."},
    packages=[
        "factor_assets",
        "factor_assets.adapters",
        "factor_assets.aggregation",
        "factor_assets.assembly",
        "factor_assets.campaigns",
        "factor_assets.clustering",
        "factor_assets.contracts",
        "factor_assets.graph",
        "factor_assets.identity",
        "factor_assets.lifecycle",
        "factor_assets.novelty",
        "factor_assets.optimizer",
        "factor_assets.registry",
        "factor_assets.seen_index",
        "factor_assets.selection",
        "factor_assets.similarity",
    ],
)
