"""Native installer QA on a disposable Windows Actions runner, NOT a release.

Compiles the production Inno script with an inert executable in place of the
private application payload. Screenshots show real Windows wizard controls.
They do not establish that model inference, offline WebView installation, or
the investor application works. No test executable is published by CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32" or os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("This install/uninstall test requires a disposable Windows Actions runner")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    report: dict = {
        "status": "running", "release_ready": False,
        "payload": "inert QA executable, not Boxing Vision runtime",
        "screenshots": [], "limitations": [
            "No private fonts, model weights or user video were uploaded",
            "Offline WebView2 prerequisite installation is not tested here",
            "Application launch, inference and clean consumer Windows QA remain required",
        ],
    }
    (output / "READ-ME.txt").write_text(
        "Real Windows installer screenshots; inert QA payload, NOT the investor release.\n"
        "The production installer script is compiled unchanged. No application EXE is distributed here.\n",
        encoding="utf-8",
    )
    try:
        import winreg

        from pywinauto import Application, Desktop

        key = r"Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
        runtime_present = False
        for hive, flags in [(winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY),
                            (winreg.HKEY_CURRENT_USER, 0)]:
            try:
                with winreg.OpenKey(hive, key, 0, winreg.KEY_READ | flags) as opened:
                    version = winreg.QueryValueEx(opened, "pv")[0]
                    runtime_present |= bool(version and version != "0.0.0.0")
            except FileNotFoundError:
                pass
        if not runtime_present:
            raise RuntimeError("Runner lacks WebView2; QA must not execute a fake prerequisite installer")
        report["webview2_already_present"] = True
        compiler = Path(os.environ["ProgramFiles(x86)"]) / "Inno Setup 6" / "ISCC.exe"
        if not compiler.is_file():
            raise RuntimeError("Expected runner-provided Inno Setup 6 compiler is missing")
        work = Path(tempfile.mkdtemp(prefix="boxing-installer-qa-"))
        bundle = work / "probe-bundle"
        bundle.mkdir()
        fixture = bundle / "BoxingVision.exe"
        source = work / "Probe.cs"
        source.write_text("class Probe { static void Main() {} }", encoding="utf-8")
        csc = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
        subprocess.run([str(csc), "/nologo", "/target:winexe", f"/out:{fixture}", str(source)], check=True)
        compile_result = subprocess.run([
            str(compiler), f"/DBundleDir={bundle}", f"/DWebView2Installer={fixture}",
            f"/O{work}", str(root / "packaging/windows/installer.iss"),
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (output / "compiler.log").write_text(compile_result.stdout, encoding="utf-8")
        compile_result.check_returncode()
        report["compiled"] = True
        installer = work / "BoxingVision-Setup-x64.exe"
        install_dir = work / "installed"
        log_path = output / "install.log"
        command = subprocess.list2cmdline([
            str(installer), "/LANG=russian", "/NORESTART", "/CURRENTUSER",
            f"/DIR={install_dir}", f"/LOG={log_path}",
        ])
        Application(backend="win32").start(command)
        # Inno's setup loader spawns a child; select the actual wizard by title.
        wizard = Desktop(backend="win32").window(title_re=".*Boxing Vision.*", class_name="TWizardForm")
        wizard.wait("visible", timeout=40)
        for step in range(1, 9):
            wizard.wait("ready", timeout=30)
            time.sleep(0.5)
            wrapper = wizard.wrapper_object()
            controls = wrapper.descendants()
            texts = [c.window_text() for c in controls if c.is_visible() and c.window_text()]
            shot = f"{step:02d}-installer.png"
            wrapper.capture_as_image().save(output / shot)
            report["screenshots"].append({"file": shot, "visible_text": texts})
            buttons = [c for c in controls if c.class_name() == "TNewButton" and c.is_visible() and c.is_enabled()]
            def label(button):
                return button.window_text().replace("&", "").strip()
            finish = next((b for b in buttons if label(b) in {"Завершить", "Finish"}), None)
            if finish:
                report["reached_finish"] = True
                finish.click()
                break
            forward = next((b for b in buttons if label(b) in {"Далее >", "Next >", "Установить", "Install"}), None)
            if forward is None:
                raise RuntimeError(f"No forward wizard action: {texts}")
            installing = label(forward) in {"Установить", "Install"}
            forward.click()
            if installing:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if (install_dir / "unins000.exe").is_file():
                        time.sleep(1)
                        break
                    time.sleep(0.2)
        if not report.get("reached_finish"):
            raise RuntimeError("Installer did not reach its completion page")
        installed = install_dir / "BoxingVision.exe"
        if hashlib.sha256(installed.read_bytes()).digest() != hashlib.sha256(fixture.read_bytes()).digest():
            raise RuntimeError("Installed payload differs from input")
        report["installed_payload_matches"] = True
        uninstaller = install_dir / "unins000.exe"
        result = subprocess.run([str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                                 f"/LOG={output / 'uninstall.log'}"], timeout=60, check=False)
        result.check_returncode()
        if installed.exists():
            raise RuntimeError("Uninstaller left the application executable")
        report["uninstalled"] = True
        report["status"] = "installer_probe_passed_not_release"
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
