"""Source-linked documentation coverage."""
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
