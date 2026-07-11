#!/usr/bin/env python3
"""Evaluate the external GTJA185 factor pack with AutoFactorEvaluation."""
from __future__ import annotations

import sys

from evaluation.batch import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--provider",
                "autofactor.provider:load_pack",
                "--output-dir",
                "gtja191/output/autofactor",
                *sys.argv[1:],
            ]
        )
    )
