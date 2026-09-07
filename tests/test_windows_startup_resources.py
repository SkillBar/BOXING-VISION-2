from pathlib import Path


def test_safehttpx_runtime_version_file_is_included_in_frozen_app():
    root = Path(__file__).resolve().parents[1]
    spec = (root / "packaging/windows/boxing_vision.spec").read_text()
    assert 'collect_data_files("safehttpx")' in spec


def test_window_title_alone_does_not_pass_installed_app_qa():
    root = Path(__file__).resolve().parents[1]
    script = (root / "tools/private_windows_build.py").read_text()
    assert 'class_name() == "#32770"' in script
    assert 'state.get("videoReady", 0) < 2' in script
    assert 'state.get("pid") != process_id' in script
    desktop = (root / "boxing_vision/desktop.py").read_text()
    assert "На Windows требуется Microsoft Edge WebView2 Runtime. Подробности" not in desktop
    workflow = (root / ".github/workflows/private-windows-build.yml").read_text()
    assert "python -m tools.private_windows_build " in workflow
