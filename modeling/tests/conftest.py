"""Source-tree test bootstrap for the optional sibling adapter dependency."""

import sys
from pathlib import Path


# The standalone wheel deliberately does not vendor factor_preprocess.  When
# running this repository's tests directly, expose the sibling source package
# just as the integration test suite does; installed-package users must still
# install the ``preprocess`` extra explicitly.
_sibling_root = Path(__file__).resolve().parents[2] / "factor_preprocess"
if _sibling_root.is_dir():
    sys.path.insert(0, str(_sibling_root))
