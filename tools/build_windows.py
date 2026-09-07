"""Validate an explicit asset allowlist and build on Windows x64 only.

No videos, analyses, user fonts or model caches are discovered automatically.
Use --validate-only to inspect a supplied input manifest without building.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from boxing_vision.desktop import (
    RTM_MODEL_REQUIREMENTS,
    DesktopSetupError,
    safe_bundle_file,
    sha256_file,
)

REQUIRED_ROLES = {
    "ffmpeg",
    "ffprobe",
    "detector",
    "pose",
    "sf_regular",
    "sf_medium",
    "sf_semibold",
    "sf_bold",
    "druk_medium",
    "druk_bold",
}
FONT_NAMES = {
    "sf_regular": "SF-Pro-Text-Regular",
    "sf_medium": "SF-Pro-Text-Medium",
    "sf_semibold": "SF-Pro-Text-Semibold",
    "sf_bold": "SF-Pro-Text-Bold",
    "druk_medium": "DrukCyr-Medium",
    "druk_bold": "DrukCyr-Bold",
}
PUNCH_FILES = {
    "punch_model": "model.onnx",
    "punch_manifest": "manifest.json",
    "punch_license": "LICENSE",
}


def windows_pe_x64(path: Path) -> bool:
    """Reject renamed macOS executables and the wrong Windows architecture."""
    with path.open("rb") as handle:
        if handle.read(2) != b"MZ":
            return False
        handle.seek(0x3C)
        offset_bytes = handle.read(4)
        if len(offset_bytes) != 4:
            return False
        offset = struct.unpack("<I", offset_bytes)[0]
        if offset > path.stat().st_size - 6:
            return False
        handle.seek(offset)
        return handle.read(6) == b"PE\x00\x00\x64\x86"


def validate_public_model_manifest(document: Any) -> None:
    """Do not distribute local training/source paths inside optional metadata.

    URLs, model hashes and architecture conventions are retained verbatim. An
    offending field is rejected, never silently removed from a signed/hash input.
    """
    private_fields = {
        "source_path",
        "local_path",
        "checkpoint_path",
        "dataset_path",
        "training_path",
        "user_dir",
        "data_dir",
        "cache_dir",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if any(str(key).lower() in private_fields for key in value):
                raise DesktopSetupError(
                    "Punch manifest contains a private local-path field"
                )
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and (
            PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or bool(PureWindowsPath(value).drive)
        ):
            raise DesktopSetupError("Punch manifest contains an absolute local path")

    if not isinstance(document, dict):
        raise DesktopSetupError("Punch manifest must be a JSON object")
    visit(document)


def validated_inputs(manifest_path: Path) -> list[dict[str, Any]]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("platform") != "windows-x64"
    ):
        raise DesktopSetupError(
            "Input manifest must declare schema_version=1 and platform=windows-x64"
        )
    files = manifest.get("files", [])
    if not isinstance(files, list) or not all(isinstance(item, dict) for item in files):
        raise DesktopSetupError("Input files must be a list of resource records")
    roles = [record.get("role") for record in files]
    if (
        not all(isinstance(role, str) for role in roles)
        or len(roles) != len(set(roles))
        or not REQUIRED_ROLES.issubset(roles)
    ):
        raise DesktopSetupError(
            "Provide exactly one input for every required binary, model and SF Pro/Druk weight"
        )
    allowed = REQUIRED_ROLES | set(PUNCH_FILES) | {"webview2"}
    if any(role not in allowed for role in roles):
        raise DesktopSetupError(
            "Unexpected input role: only explicit runtime assets are permitted"
        )
    if set(roles).intersection(PUNCH_FILES) and not set(PUNCH_FILES).issubset(roles):
        raise DesktopSetupError(
            "An optional punch bundle requires ONNX, manifest and LICENSE together"
        )
    result = []
    for record in files:
        role = record["role"]
        if not isinstance(record.get("source"), str) or not record["source"]:
            raise DesktopSetupError(f"Missing explicit source path: {role}")
        source = Path(record["source"]).expanduser()
        if not source.is_absolute():
            source = manifest_path.parent / source
        source = source.resolve(strict=True)
        if not source.is_file() or source.stat().st_size == 0:
            raise DesktopSetupError(f"Missing input file: {role}")
        expected = str(record.get("sha256", "")).lower()
        if len(expected) != 64 or sha256_file(source) != expected:
            raise DesktopSetupError(f"Input SHA256 mismatch: {role}")
        if (
            record.get("redistribution_approved") is not True
            or not record.get("license")
            or not record.get("source_url")
        ):
            raise DesktopSetupError(
                f"Explicit provenance and permission for this build are required: {role}"
            )
        if role == "webview2":
            # The signed installer container can be x86 even for the x64 runtime.
            # Validate publisher on the Windows build host, not just its name.
            with source.open("rb") as handle:
                executable_header = handle.read(2)
            if source.suffix.lower() != ".exe" or executable_header != b"MZ":
                raise DesktopSetupError("WebView2 must be the official Evergreen Standalone x64 installer")
            destination = "prerequisites/WebView2RuntimeInstaller.exe"
        elif role in {"ffmpeg", "ffprobe"}:
            if not windows_pe_x64(source):
                raise DesktopSetupError(
                    f"{role} must be a Windows x64 PE executable, not a renamed Mac binary"
                )
            destination = f"bin/{role}.exe"
        elif role in RTM_MODEL_REQUIREMENTS:
            filename, registered_hash = RTM_MODEL_REQUIREMENTS[role]
            if expected != registered_hash:
                raise DesktopSetupError(
                    f"Unregistered {role} weights; compare against the current pipeline pins"
                )
            destination = f"models/rtmlib/{filename}"
        elif role in FONT_NAMES:
            with source.open("rb") as handle:
                if handle.read(4) not in {b"OTTO", b"\x00\x01\x00\x00", b"true"}:
                    raise DesktopSetupError(f"{role} must be a genuine OTF/TTF font")
            extension = source.suffix.lower()
            if extension not in {".otf", ".ttf"} or (
                role.startswith("druk_") and extension != ".ttf"
            ):
                raise DesktopSetupError(f"Unsupported font format for {role}")
            destination = f"fonts/{FONT_NAMES[role]}{extension}"
        else:
            destination = f"models/acm40960-lstm-v1/{PUNCH_FILES[role]}"
            if role == "punch_manifest":
                validate_public_model_manifest(
                    json.loads(source.read_text(encoding="utf-8"))
                )
        result.append(
            {
                "source": str(source),
                "path": destination,
                "role": role,
                "sha256": expected,
                "license": str(record["license"]),
                "source_url": str(record["source_url"]),
                "redistribution_approved": True,
            }
        )
    return result


def stage_inputs(records: list[dict[str, Any]], stage: Path) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    for record in records:
        target = safe_bundle_file(stage, record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record["source"], target)
        if sha256_file(target) != record["sha256"]:
            raise DesktopSetupError(f"Staging SHA256 mismatch: {record['role']}")
    if any(record["role"] == "punch_model" for record in records):
        from boxing_vision.model_registry import validate_acm_bundle

        validate_acm_bundle(stage / "models" / "acm40960-lstm-v1")
    payload = {
        "schema_version": 1,
        "platform": "windows-x64",
        "files": [
            {key: value for key, value in row.items() if key != "source"}
            for row in records if row["role"] != "webview2"
        ],
    }
    (stage / "bundle-manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def create_recipient_folder(installer: Path, output: Path) -> Path:
    """Two recipient-facing files only, after a real installer was produced."""
    if not installer.is_file() or installer.stat().st_size < 1024:
        raise DesktopSetupError("Installer missing: cannot create a recipient package from source code")
    with installer.open("rb") as handle:
        if handle.read(2) != b"MZ":
            raise DesktopSetupError("Expected a Windows installer, not a renamed archive")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(installer, output / "BoxingVision-Setup.exe")
    (output / "Начать здесь.txt").write_text(
        "Boxing Vision — установка\n\n"
        "1. Скачайте всю папку на компьютер с Windows 10/11 x64.\n"
        "2. Откройте BoxingVision-Setup.exe и нажмите «Установить».\n"
        "3. Запустите Boxing Vision с рабочего стола.\n\n"
        "Python, библиотеки и модели уже входят в приложение. Команды вводить не нужно.\n"
        "При необходимости установщик подготовит Microsoft WebView2 автоматически.\n"
        "Если в комплект включён демо-разбор, он откроется сразу. Для своего видео нажмите «Новый анализ».\n\n"
        "Если Windows или антивирус блокирует установку, не отключайте защиту: передайте сообщение отправителю.\n",
        encoding="utf-8-sig",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist/windows"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--demo", type=Path, help="Explicit prepared read-only demo directory; never scan local runs")
    parser.add_argument(
        "--iscc",
        type=Path,
        help="Explicit path to Inno Setup 6 ISCC.exe; omit for onedir only",
    )
    args = parser.parse_args()
    if not args.validate_only and (
        sys.platform != "win32"
        or platform.machine().lower() not in {"amd64", "x86_64"}
        or struct.calcsize("P") != 8
    ):
        parser.error(
            "A real Windows x64 Python host is required; PyInstaller is not a Windows cross-compiler on macOS"
        )
    records = validated_inputs(args.inputs)
    if args.iscc and not any(row["role"] == "webview2" for row in records):
        parser.error("The simple installer requires an approved offline WebView2 input (role=webview2)")
    if args.demo:
        from boxing_vision.demo_bundle import verify_demo
        verify_demo(args.demo)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "inputs_verified",
                    "files": len(records),
                    "executable_built": False,
                },
                indent=2,
            )
        )
        return 0
    root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Unique build directories preserve all earlier runs and release outputs.
    work = Path(tempfile.mkdtemp(prefix="boxing-vision-build-", dir=output))
    stage = work / "payload"
    stage_inputs(records, stage)
    if args.demo:
        from boxing_vision.demo_bundle import stage_demo
        demo_record = stage_demo(args.demo, stage)
        bundle_path = stage / "bundle-manifest.json"
        manifest = json.loads(bundle_path.read_text(encoding="utf-8"))
        manifest["files"].append(demo_record)
        bundle_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import os

    environment = dict(
        os.environ,
        BOXING_VISION_BUILD_PAYLOAD=str(stage),
        BOXING_VISION_BUILD_ROOT=str(root),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--distpath",
            str(work / "dist"),
            "--workpath",
            str(work / "pyinstaller"),
            str(root / "packaging/windows/boxing_vision.spec"),
        ],
        check=True,
        cwd=root,
        env=environment,
    )
    executable = work / "dist" / "BoxingVision" / "BoxingVision.exe"
    if not executable.is_file() or not windows_pe_x64(executable):
        raise DesktopSetupError("PyInstaller did not produce a Windows x64 executable")
    subprocess.run([str(executable), "--check"], check=True, cwd=executable.parent)
    if args.iscc:
        compiler = args.iscc.resolve(strict=True)
        runtime = next(row for row in records if row["role"] == "webview2")
        # Avoid interpolating paths into PowerShell source. Env carries a literal
        # path; an unsigned or non-Microsoft installer never gets embedded.
        signature_env = dict(environment, BOXING_VISION_PREREQUISITE=str(safe_bundle_file(stage, runtime["path"])))
        subprocess.run([
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            ("$s = Get-AuthenticodeSignature -LiteralPath $env:BOXING_VISION_PREREQUISITE; "
             "if ($s.Status -ne 'Valid' -or $s.SignerCertificate.Subject -notmatch '(?:^|, )O=Microsoft Corporation(?:,|$)') { exit 1 }"),
        ], check=True, env=signature_env)
        subprocess.run(
            [
                str(compiler),
                f"/DBundleDir={executable.parent}",
                f"/DWebView2Installer={safe_bundle_file(stage, runtime['path'])}",
                f"/O{work / 'installer'}",
                str(root / "packaging/windows/installer.iss"),
            ],
            check=True,
            cwd=root,
        )
        create_recipient_folder(work / "installer" / "BoxingVision-Setup-x64.exe",
                                work / "handoff" / "BoxingVision-Windows")
    (work / "build-report.json").write_text(
        json.dumps(
            {
                "executable": str(executable),
                "sha256": sha256_file(executable),
                "status": "built_needs_windows_visual_and_video_QA",
                "python": sys.version,
                "platform": platform.platform(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Built: {executable}\nWindows playback, full analysis and clean-machine QA remain required."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
