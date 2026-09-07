"""The public validation job must never masquerade as a private release."""

from pathlib import Path


def test_windows_ci_keeps_private_resources_and_fixture_executables_out_of_artifacts():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/windows-validation.yml").read_text()
    assert "contents: read" in workflow
    assert "runs-on: windows-2022" in workflow
    assert "persist-credentials: false" in workflow
    assert "*.exe" not in workflow
    assert "private-fonts" not in workflow and "investor-demo" not in workflow
    assert "NOT-release" in workflow
    script = (root / "tools/windows_installer_qa.py").read_text()
    assert 'os.environ.get("GITHUB_ACTIONS") != "true"' in script
    assert '"release_ready": False' in script
    assert 'if not runtime_present:' in script
    assert '"installer_probe_passed_not_release"' in script
