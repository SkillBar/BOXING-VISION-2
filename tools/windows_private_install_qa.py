"""Real full-payload install wizard automation on the disposable Windows runner.

All evidence is private: callers encrypt the screenshots and installation log.
Do not run this against a personal Windows installation.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def install_and_capture(installer: Path, expected_executable: Path, output: Path) -> Path:
    if sys.platform != "win32" or os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Disposable Windows CI only")
    from pywinauto import Application, Desktop

    output.mkdir(parents=True, exist_ok=False)
    install_dir = Path(os.environ["LOCALAPPDATA"]) / "Programs/BoxingVision"
    if install_dir.exists():
        raise RuntimeError("Refusing to overwrite a pre-existing installation")
    command = subprocess.list2cmdline([
        str(installer), "/LANG=russian", "/NORESTART", "/CURRENTUSER",
        f"/DIR={install_dir}", f"/LOG={output / 'install.log'}",
    ])
    Application(backend="win32").start(command)
    wizard = Desktop(backend="win32").window(title_re=".*Boxing Vision.*", class_name="TWizardForm")
    wizard.wait("visible", timeout=60)
    pages = []
    finished = False
    deadline = time.monotonic() + 420
    try:
        while time.monotonic() < deadline:
            time.sleep(0.8)
            wrapper = wizard.wrapper_object()
            controls = [control for control in wrapper.descendants() if control.is_visible()]
            texts = [control.window_text() for control in controls if control.window_text()]
            # Keep one screenshot per wizard page, not hundreds of progress ticks.
            heading = texts[0] if texts else ""
            if not pages or heading != pages[-1]["heading"]:
                filename = f"step-{len(pages) + 1:02d}.png"
                wrapper.capture_as_image().save(output / filename)
                pages.append({"file": filename, "heading": heading, "text": texts})
            buttons = {control.window_text().replace("&", "").strip(): control
                       for control in controls if control.class_name() == "TNewButton" and control.is_enabled()}
            finish = buttons.get("Завершить") or buttons.get("Finish")
            if finish:
                finish.click()
                finished = True
                break
            forward = next((buttons[name] for name in ("Далее", "Далее >", "Next", "Next >", "Установить", "Install") if name in buttons), None)
            if forward:
                forward.click()
        if not finished:
            raise RuntimeError("Full installer did not reach completion within 7 minutes")
        installed = install_dir / "BoxingVision.exe"
        if hashlib.sha256(installed.read_bytes()).digest() != hashlib.sha256(expected_executable.read_bytes()).digest():
            raise RuntimeError("Installed EXE hash mismatch")
        return installed
    finally:
        (output / "steps.json").write_text(json.dumps({
            "real_windows_installer": True, "full_payload": True,
            "finished": finished, "pages": pages,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
