from pathlib import Path

import blasphemous


ROOT = Path(__file__).resolve().parents[1]


def test_package_version_is_v040():
    assert blasphemous.__version__ == "0.4.0"


def test_readme_does_not_reference_removed_branch():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "run_unified" not in readme
    assert "blasphemous.unified" not in readme
    assert "unified dynamic mode selection system" not in readme


def test_release_docs_exist():
    assert (ROOT / "docs" / "quickstart.md").exists()
    assert (ROOT / "docs" / "benchmark.md").exists()
    assert (ROOT / "docs" / "release-checklist.md").exists()
