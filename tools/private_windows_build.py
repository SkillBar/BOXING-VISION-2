"""Assemble an explicitly authorized private Windows evaluation, never publish it.

Input fonts/video/weights arrive encrypted and are not discovered from the host.
Only public Windows media prerequisites are fetched here. Their provenance and
the font/model rights limitations remain in the resulting manifest/notices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


def download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=90) as source, target.open("xb") as output:
        shutil.copyfileobj(source, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Windows only")
    inputs = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    document = json.loads((inputs / "inputs.json").read_text(encoding="utf-8"))
    if document.get("delivery_scope") != "private_evaluation":
        raise ValueError("Explicit private evaluation manifest required")
    downloads = inputs / "windows-prerequisites"
    downloads.mkdir()
    ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.1-essentials_build.zip"
    archive = downloads / "ffmpeg.zip"
    download(ffmpeg_url, archive)
    sha_file = downloads / "ffmpeg.sha256"
    download(ffmpeg_url + ".sha256", sha_file)
    expected = sha_file.read_text().split()[0].lower()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("FFmpeg publisher archive hash mismatch")

    def register(role, path, source_url, license_text):
        document["files"].append({
            "role": role, "source": path.relative_to(inputs).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_url": source_url, "license": license_text,
            "redistribution_approved": False, "private_build_authorized": True,
            "rights_note": "Private evaluation requested by the user; not an assertion of general redistribution rights.",
        })
    with zipfile.ZipFile(archive) as package:
        for role in ("ffmpeg", "ffprobe"):
            member = next(name for name in package.namelist() if name.endswith(f"/bin/{role}.exe"))
            binary = downloads / f"{role}.exe"
            binary.write_bytes(package.read(member))
            register(role, binary, ffmpeg_url, "GPL-3.0; static Gyan FFmpeg distribution; see included license and upstream source/build links")
        license_member = next(name for name in package.namelist() if name.lower().endswith("/license"))
        notice = downloads / "FFmpeg-LICENSE.txt"
        notice.write_bytes(package.read(license_member))
        register("ffmpeg_license", notice, ffmpeg_url, "GPL-3.0 license text")
    runtime = downloads / "MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
    runtime_url = "https://go.microsoft.com/fwlink/?linkid=2124701"
    download(runtime_url, runtime)
    register("webview2", runtime, runtime_url, "Microsoft WebView2 Runtime redistribution terms")
    final_input = inputs / "windows-inputs.json"
    final_input.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    compiler = Path(os.environ["ProgramFiles(x86)"]) / "Inno Setup 6/ISCC.exe"
    subprocess.run([
        sys.executable, "-m", "tools.build_windows", "--inputs", str(final_input),
        "--private-evaluation", "--demo", str(inputs / "demo"), "--iscc", str(compiler),
        "--output", str(output / "build"),
    ], check=True, timeout=2400)
    builds = list((output / "build").glob("boxing-vision-build-*"))
    if len(builds) != 1:
        raise ValueError("Expected one isolated build")
    shutil.copytree(builds[0] / "handoff/BoxingVision-Windows", output / "delivery/BoxingVision-Windows")
    shutil.copyfile(builds[0] / "build-report.json", output / "delivery/build-report.json")
    # Capture the real native application, not a browser titlebar mockup.
    # These contain private footage and are returned only in encrypted delivery.
    from pywinauto import Desktop

    executable = builds[0] / "dist/BoxingVision/BoxingVision.exe"
    process = subprocess.Popen([str(executable)], cwd=executable.parent)
    qa = output / "delivery/screenshots"
    qa.mkdir()
    try:
        # WinForms class suffix depends on runtime; title is exact and unique in this isolated VM.
        window = Desktop(backend="win32").window(title="Boxing Vision", visible_only=True)
        window.wait("visible", timeout=120)
        time.sleep(15)
        for width, height in [(1440, 900), (1280, 800), (1024, 768)]:
            wrapper = window.wrapper_object()
            wrapper.move_window(x=0, y=0, width=width, height=height, repaint=True)
            time.sleep(3)
            wrapper.capture_as_image().save(qa / f"windows-app-{width}x{height}.png")
        (qa / "README.txt").write_text(
            "Real native Windows window from CI, not a mockup. Inspect actual video/font rendering visually.\n"
            "Requested sizes may be limited by the runner display. Full inference and consumer-PC QA remain required.\n",
            encoding="utf-8",
        )
    finally:
        # No user work exists in this build VM; end only this application process tree.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False)


if __name__ == "__main__":
    main()
