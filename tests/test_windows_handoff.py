from pathlib import Path

import pytest

from boxing_vision.desktop import DesktopSetupError
from tools.build_windows import create_recipient_folder


def test_recipient_folder_contains_only_installer_and_plain_instructions(tmp_path):
    # A fake executable checks packaging logic, never Windows runtime readiness.
    installer = tmp_path / "fixture.exe"
    installer.write_bytes(b"MZ" + b"0" * 2048)
    output = create_recipient_folder(installer, tmp_path / "handoff")
    assert {p.name for p in output.iterdir()} == {"BoxingVision-Setup.exe", "Начать здесь.txt"}
    assert (output / "BoxingVision-Setup.exe").read_bytes() == installer.read_bytes()
    text = (output / "Начать здесь.txt").read_text(encoding="utf-8-sig")
    assert "Команды вводить не нужно" in text
    assert "не отключайте защиту" in text
    assert "pip install" not in text
    with pytest.raises(FileExistsError):
        create_recipient_folder(installer, output)


@pytest.mark.parametrize("data", [None, b"", b"source code" * 300, b"MZshort"])
def test_missing_or_invalid_installer_does_not_create_fake_ready_folder(tmp_path, data):
    source = tmp_path / "missing.exe"
    if data is not None:
        source.write_bytes(data)
    output = tmp_path / "handoff"
    with pytest.raises(DesktopSetupError):
        create_recipient_folder(source, output)
    assert not output.exists()


def test_installer_prepares_offline_runtime_before_launch_without_global_python():
    root = Path(__file__).resolve().parents[1]
    script = (root / "packaging/windows/installer.iss").read_text()
    assert "function PrepareToInstall" in script
    assert "if WebView2Installed then Exit" in script
    assert "RuntimeVersionPresent(HKLM32) or RuntimeVersionPresent(HKCU)" in script
    assert "F3017226-FE2A-4295-8BDF-00C3A9A7E4C5" in script
    assert "'/silent /install'" in script
    assert "ewWaitUntilTerminated" in script
    assert "NeedsRestart := True" in script
    assert "not WebView2Installed" in script
    assert "pip install" not in script and "python.exe" not in script
    assert script.index('Flags: dontcopy') < script.index('recursesubdirs')
    spec = (root / "packaging/windows/boxing_vision.spec").read_text()
    assert '"prerequisites" not in item.relative_to(payload).parts' in spec
