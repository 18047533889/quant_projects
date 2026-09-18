"""Source-linked documentation coverage."""
import re
import runpy
from pathlib import Path


def test_api_reference_matches_source():
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / "scripts" / "build_api_reference.py"))
    assert namespace["OUTPUT"].read_text() == namespace["render"]()


def test_functional_guide_exists_and_is_linked():
    root = Path(__file__).resolve().parents[1]
    guide = root / "docs" / "FUNCTIONAL_GUIDE.md"
    assert guide.is_file()
    assert len(guide.read_text()) > 5000
    assert "FUNCTIONAL_GUIDE.md" in (root / "README.md").read_text()


def test_transform_catalog_matches_default_registry():
    """The readable table must cover the conditional runtime registry."""
    from factor_preprocess.registry.transforms import create_default_registry

    root = Path(__file__).resolve().parents[1]
    catalog = (root / "docs" / "TRANSFORM_CATALOG.md").read_text()
    documented = set(re.findall(r"^\| `([a-z0-9_]+)` \|", catalog, re.MULTILINE))
    registered = {meta.name for meta in create_default_registry().all_transforms()}
    optional = {"wavelet_decompose", "wavelet_smooth", "wavelet_denoise"}

    assert optional <= documented
    assert registered <= documented
    assert registered - optional == documented - optional
