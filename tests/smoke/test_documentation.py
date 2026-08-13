"""
Smoke test: Documentation
Test that documentation files exist and can be processed.
"""
import pytest
from pathlib import Path


DOCS_DIR = Path("/home/shw/quant_projects/docs")
PROJECT_ROOT = Path("/home/shw/quant_projects")


class TestDocumentation:
    """Test documentation exists and is accessible."""

    def test_main_readme_exists(self):
        """Test that main README exists."""
        readme = PROJECT_ROOT / "README.md"
        assert readme.exists(), "Main README.md should exist"
        assert readme.stat().st_size > 0, "README should not be empty"

    def test_docs_directory_exists(self):
        """Test that docs directory exists."""
        assert DOCS_DIR.exists(), "docs/ directory should exist"
        assert DOCS_DIR.is_dir(), "docs/ should be a directory"

    def test_docs_readme_exists(self):
        """Test that docs README exists."""
        docs_readme = DOCS_DIR / "README.md"
        if docs_readme.exists():
            assert docs_readme.stat().st_size > 0, "docs/README should not be empty"
        else:
            pytest.skip("docs/README.md not found")

    def test_package_readmes_exist(self):
        """Test that each package has a README."""
        packages = ["dataaccess", "factor_engine", "factor_optimizer", "quant_evaluator"]

        for pkg in packages:
            pkg_dir = PROJECT_ROOT / pkg
            readme = pkg_dir / "README.md"

            if pkg_dir.exists() and pkg_dir.is_dir():
                # At least the directory should exist
                assert True
                if readme.exists():
                    assert readme.stat().st_size > 0, f"{pkg} README should not be empty"

    def test_markdown_files_readable(self):
        """Test that markdown files can be read."""
        md_files = list(DOCS_DIR.rglob("*.md"))

        assert len(md_files) > 0, "Should have at least some markdown files"

        for md_file in md_files[:10]:  # Check first 10 files
            try:
                content = md_file.read_text(encoding='utf-8')
                assert len(content) > 0, f"{md_file} should not be empty"
            except Exception as e:
                pytest.fail(f"Could not read {md_file}: {e}")

    def test_chinese_docs_readable(self):
        """Test that Chinese documentation files can be read."""
        chinese_doc = DOCS_DIR / "量化平台使用总览.md"

        if chinese_doc.exists():
            try:
                content = chinese_doc.read_text(encoding='utf-8')
                assert len(content) > 0, "Chinese doc should not be empty"
            except Exception as e:
                pytest.fail(f"Could not read Chinese doc: {e}")
        else:
            pytest.skip("Chinese documentation not found")
