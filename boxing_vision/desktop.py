"""Native desktop host for the existing local Gradio/ML application.

Importing this module does not import Gradio, ONNX, pywebview or create files.
The frozen application configures writable paths before importing the UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

RTM_MODEL_REQUIREMENTS = {
    "detector": (
        "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx",
        "ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250",
    ),
    "pose": (
        "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx",
        "9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d",
    ),
}


class DesktopSetupError(RuntimeError):
    """A missing or invalid desktop resource, before model initialization."""


@dataclass(frozen=True, slots=True)
class DesktopPaths:
    bundle: Path
    data: Path
    fonts: Path

    @property
    def runs(self) -> Path:
        return self.data / "runs"

    @property
    def cache(self) -> Path:
        return self.data / "cache"

    @property
    def logs(self) -> Path:
        return self.data / "logs"


def resolve_paths(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    bundle: Path | None = None,
) -> DesktopPaths:
    env = os.environ if environ is None else environ
    target = sys.platform if platform is None else platform
    root = bundle or Path(
        getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)
    )
    if env.get("BOXING_VISION_DATA_DIR"):
        data = Path(env["BOXING_VISION_DATA_DIR"]).expanduser().resolve()
    elif target == "win32":
        local = env.get("LOCALAPPDATA")
        if not local:
            raise DesktopSetupError(
                "Windows не предоставил LOCALAPPDATA для данных Boxing Vision"
            )
        data = Path(local) / "BoxingVision"
    elif target == "darwin":
        data = Path.home() / "Library" / "Application Support" / "BoxingVision"
    else:
        data = (
            Path(env.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
            / "BoxingVision"
        )
    fonts = (
        Path(env["BOXING_VISION_FONT_DIR"]).expanduser()
        if env.get("BOXING_VISION_FONT_DIR")
        else root / "fonts"
    )
    root, data, fonts = root.resolve(), data.resolve(), fonts.resolve()
    if data == root or data.is_relative_to(root):
        raise DesktopSetupError(
            "Данные и кэш должны находиться отдельно от каталога приложения"
        )
    return DesktopPaths(root, data, fonts)


def prepare_environment(
    paths: DesktopPaths, *, environ: MutableMapping[str, str] | None = None
) -> None:
    env = os.environ if environ is None else environ
    for directory in (
        paths.data,
        paths.runs,
        paths.cache,
        paths.logs,
        paths.cache / "gradio",
        paths.cache / "rtmlib" / "hub" / "checkpoints",
        paths.cache / "webview",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "BOXING_VISION_DATA_DIR": str(paths.data),
            "BOXING_VISION_DESKTOP": "1",
            "BOXING_VISION_FONT_DIR": str(paths.fonts),
            "GRADIO_TEMP_DIR": str(paths.cache / "gradio"),
            "GRADIO_ANALYTICS_ENABLED": "False",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TORCH_HOME": str(paths.cache / "rtmlib"),
            "XDG_CACHE_HOME": str(paths.cache),
            "PYWEBVIEW_GUI": "edgechromium"
            if sys.platform == "win32"
            else env.get("PYWEBVIEW_GUI", ""),
        }
    )
    if (paths.bundle / "bin").is_dir():
        env["PATH"] = str(paths.bundle / "bin") + os.pathsep + env.get("PATH", "")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_bundle_file(root: Path, relative: str) -> Path:
    posix, windows = PurePosixPath(relative), PureWindowsPath(relative)
    if (
        not relative
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise DesktopSetupError("Некорректный путь ресурса приложения")
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DesktopSetupError("Ресурс выходит за пределы каталога приложения")
    return path


def resolve_saved_run(paths: DesktopPaths, value: str) -> Path:
    """Validate an explicitly selected local result, without importing the ML/UI.

    A bare ID is relative to this application's runs directory; paths must be
    absolute. Reject remote paths before touching the filesystem, and reject
    linked artifacts because the UI can subsequently serve clips and previews.
    No result is discovered or selected automatically.
    """
    if not value or not value.strip() or "\x00" in value:
        raise DesktopSetupError("Укажите ID или полный локальный путь сохранённого анализа")
    value = value.strip()
    windows = PureWindowsPath(value)
    if value.startswith(("//", "\\\\")) or "://" in value:
        raise DesktopSetupError("Открытие анализа по сетевому адресу не поддерживается")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        if windows.drive or len(PurePosixPath(value).parts) != 1 or len(windows.parts) != 1 or value in {".", ".."}:
            raise DesktopSetupError("Укажите ID анализа или полный путь внутри каталога runs приложения")
        candidate = paths.runs / candidate
    try:
        root = paths.runs.resolve(strict=True)
        run = candidate.resolve(strict=True)
        if run == root or not run.is_relative_to(root):
            raise DesktopSetupError("Анализ должен находиться внутри каталога runs приложения")
        if not run.is_dir():
            raise DesktopSetupError("Сохранённый анализ должен быть каталогом")
        # Do not follow symlinks or Windows directory junctions when restoring
        # a run: neither media files nor later review/cache inputs may escape.
        pending = [run]
        while pending:
            for artifact in pending.pop().iterdir():
                if artifact.is_symlink() or getattr(artifact, "is_junction", lambda: False)():
                    raise DesktopSetupError("Сохранённый анализ содержит ссылки на файлы или каталоги; используйте локальную копию")
                resolved = artifact.resolve(strict=True)
                if not resolved.is_relative_to(run):
                    raise DesktopSetupError("Файл сохранённого анализа выходит за пределы его каталога")
                if artifact.is_dir():
                    pending.append(artifact)
                elif not artifact.is_file():
                    raise DesktopSetupError("Сохранённый анализ содержит неподдерживаемый тип файла")
        for filename in ("annotated.mp4", "events.json", "summary.json"):
            artifact = run / filename
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise DesktopSetupError(f"Сохранённый анализ неполон: нужен непустой {filename}")
        events = json.loads((run / "events.json").read_text(encoding="utf-8"))
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if not isinstance(events, list) or not all(isinstance(event, dict) for event in events) or not isinstance(summary, dict):
            raise DesktopSetupError("Повреждён формат events.json или summary.json сохранённого анализа")
    except DesktopSetupError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise DesktopSetupError(f"Не удалось открыть сохранённый анализ: {exc}") from exc
    return run


def prepare_saved_run(
    paths: DesktopPaths, value: str, *, environ: MutableMapping[str, str] | None = None
) -> Path:
    """Reuse the existing result loader only after validating an explicit run."""
    run = resolve_saved_run(paths, value)
    env = os.environ if environ is None else environ
    env["BOXING_VISION_DEMO_RUN"] = str(run)
    env["BOXING_VISION_OPEN_SAVED_RUN"] = "1"
    return run


def prepare_bundled_demo(paths: DesktopPaths, manifest: Mapping[str, Any]) -> Path | None:
    """Open only an explicitly bundled example, never an arbitrary recent run."""
    record = next((row for row in manifest.get("files", []) if row.get("role") == "demo_manifest"), None)
    if record is None:
        return None
    if record.get("path") != "demo/demo-manifest.json":
        raise DesktopSetupError("Некорректный путь демонстрационного комплекта")
    from .demo_bundle import verify_demo

    root = paths.bundle / "demo"
    verify_demo(root)
    os.environ["BOXING_VISION_DEMO_RUN"] = str(root.resolve())
    os.environ.pop("BOXING_VISION_OPEN_SAVED_RUN", None)
    return root


def verify_bundle(paths: DesktopPaths, *, required: bool = True) -> dict[str, Any]:
    """Offline preflight validates the exact build allowlist, never downloads."""
    manifest_path = paths.bundle / "bundle-manifest.json"
    if not manifest_path.is_file():
        if required:
            raise DesktopSetupError(
                "Отсутствует bundle-manifest.json. Нужна полная Windows-сборка с моделями, FFmpeg и шрифтами"
            )
        return {"status": "development", "offline_ready": False, "files": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("platform") != "windows-x64"
    ):
        raise DesktopSetupError("Неподдерживаемый манифест сборки")
    records = manifest.get("files", [])
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        raise DesktopSetupError("Некорректный список ресурсов сборки")
    required_roles = {
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
    roles = [item.get("role") for item in records]
    if not all(isinstance(role, str) for role in roles) or len(roles) != len(
        set(roles)
    ):
        raise DesktopSetupError("Роли ресурсов сборки должны быть уникальны")
    if not required_roles.issubset(roles):
        raise DesktopSetupError(
            "В сборке не хватает FFmpeg, моделей или согласованной типографики SF Pro/Druk"
        )
    for record in records:
        if not isinstance(record.get("path"), str):
            raise DesktopSetupError("Не указан путь ресурса сборки")
        source = safe_bundle_file(paths.bundle, record["path"])
        expected = str(record.get("sha256", ""))
        registered_model = RTM_MODEL_REQUIREMENTS.get(record.get("role"))
        if registered_model and (source.name, expected.lower()) != registered_model:
            raise DesktopSetupError(
                "Модели сборки не совпадают с зарегистрированными YOLOX/RTMPose; автономный запуск не подтверждён"
            )
        if (
            len(expected) != 64
            or not source.is_file()
            or sha256_file(source) != expected.lower()
        ):
            raise DesktopSetupError(
                f"Ресурс отсутствует или повреждён: {record.get('role', source.name)}. Переустановите полную сборку"
            )
        if record.get("role") == "demo_manifest":
            if record.get("path") != "demo/demo-manifest.json":
                raise DesktopSetupError("Некорректный путь демонстрационного комплекта")
            from .demo_bundle import verify_demo
            verify_demo(paths.bundle / "demo")
    return {**manifest, "status": "verified", "offline_ready": True}


def seed_model_cache(paths: DesktopPaths, manifest: Mapping[str, Any]) -> None:
    """Copy only SHA-verified detector/pose weights into the writable RTMLib cache."""
    for record in manifest.get("files", []):
        if record.get("role") not in {"detector", "pose"}:
            continue
        source = safe_bundle_file(paths.bundle, record["path"])
        destination = paths.cache / "rtmlib" / "hub" / "checkpoints" / source.name
        expected = record["sha256"].lower()
        if destination.is_file() and sha256_file(destination) == expected:
            continue
        # The sole replace target is this exact app-owned model cache file.
        temporary = destination.with_suffix(destination.suffix + ".staging")
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != expected:
            raise DesktopSetupError(
                "Ошибка контрольной суммы при подготовке кэша моделей"
            )
        os.replace(temporary, destination)


def verify_media_tools(
    paths: DesktopPaths, *, runner: Callable[..., Any] = subprocess.run
) -> None:
    """Catch missing DLLs/codecs in a supplied Windows FFmpeg distribution."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

    def run(arguments: list[str]) -> Any:
        try:
            return runner(
                arguments,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                creationflags=flags,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DesktopSetupError(
                "Не удалось запустить комплект FFmpeg/FFprobe"
            ) from exc

    for name in ("ffmpeg", "ffprobe"):
        result = run(
            [str(paths.bundle / "bin" / f"{name}.exe"), "-version"],
        )
        if result.returncode != 0 or name not in result.stdout.lower():
            raise DesktopSetupError(
                f"{name} не запускается. Проверьте Windows x64 DLL-комплект"
            )
    codecs = run(
        [str(paths.bundle / "bin" / "ffmpeg.exe"), "-hide_banner", "-encoders"],
    )
    if (
        codecs.returncode != 0
        or "libx264" not in codecs.stdout
        or " aac " not in codecs.stdout
    ):
        raise DesktopSetupError("FFmpeg должен поддерживать кодирование libx264 и AAC")


def loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def is_loopback_url(url: str, port: int) -> bool:
    parsed = urlsplit(url)
    return (
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and parsed.port == port
        and not parsed.username
        and not parsed.password
    )


def window_geometry(screens: list[Any]) -> dict[str, Any]:
    screen = screens[0] if screens else None
    width = min(1440, max(320, int(getattr(screen, "width", 1504)) - 64))
    height = min(900, max(300, int(getattr(screen, "height", 964)) - 64))
    return {
        "width": width,
        "height": height,
        "min_size": (min(640, width), min(480, height)),
    }


class DesktopSession:
    def __init__(self, app: Any, cancel: Callable[[], Any] = lambda: None) -> None:
        self.app, self.cancel = app, cancel
        self._closed = False
        self._lock = threading.Lock()

    def close(self, *_: Any) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.cancel()
        finally:
            self.app.close(verbose=False)


class LogStream:
    """Windowed PyInstaller has no stdout/stderr; keep print/tqdm safe."""

    encoding = "utf-8"

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def write(self, value: str) -> int:
        if value.strip():
            self.logger.info(value.rstrip())
        return len(value)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def launch_desktop(
    paths: DesktopPaths,
    *,
    app_factory: Callable[[], Any] | None = None,
    webview_module: Any = None,
    cancel: Callable[[], Any] | None = None,
    port_factory: Callable[[], int] = loopback_port,
) -> None:
    if app_factory is None:
        # Environment is prepared before this import; do not move to module scope.
        from .ui import _request_cancel, build_app

        app_factory = build_app
        cancel = cancel or _request_cancel
    if webview_module is None:
        try:
            import webview as webview_module
        except ImportError as exc:
            raise DesktopSetupError(
                "Не установлен desktop-компонент pywebview. Используйте Windows-установщик"
            ) from exc
    app = app_factory()
    session = DesktopSession(app, cancel or (lambda: None))
    try:
        port = port_factory()
        _, url, share = app.queue(default_concurrency_limit=1, max_size=4).launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            inbrowser=False,
            prevent_thread_lock=True,
            show_error=True,
            show_api=False,
            quiet=True,
            allowed_paths=[
                str(paths.bundle / "boxing_vision" / "static"),
                str(paths.runs),
                str(paths.bundle / "demo"),
            ],
        )
        if share or not is_loopback_url(str(url), port):
            raise DesktopSetupError(
                "Приложение не смогло запустить изолированный локальный сервер"
            )
        webview_module.settings.update(
            {
                "ALLOW_DOWNLOADS": True,
                "ALLOW_FILE_URLS": False,
                "OPEN_EXTERNAL_LINKS_IN_BROWSER": True,
                "OPEN_DEVTOOLS_IN_DEBUG": False,
            }
        )
        window = webview_module.create_window(
            "Boxing Vision",
            url,
            js_api=None,
            resizable=True,
            frameless=False,
            easy_drag=False,
            background_color="#050608",
            text_select=True,
            zoomable=True,
            confirm_close=True,
            **window_geometry(list(webview_module.screens)),
        )
        window.events.closed += session.close
        webview_module.start(
            gui="edgechromium" if sys.platform == "win32" else None,
            debug=False,
            private_mode=False,
            storage_path=str(paths.cache / "webview"),
        )
    finally:
        session.close()


def show_startup_error(message: str) -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, "Boxing Vision", 0x10)
    elif sys.stderr is not None:
        print(message, file=sys.stderr)


def smoke_inference(paths: DesktopPaths) -> dict[str, Any]:
    """One real detector/pose call from the frozen bundle; not an accuracy eval."""
    import time

    import cv2
    import numpy as np

    from .pose import RTMLibPoseBackend

    video = paths.bundle / "demo" / "workspace-preview.mp4"
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, 49583)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise DesktopSetupError("Не удалось декодировать кадр встроенного видео для проверки ML")
    start = time.monotonic()
    poses = RTMLibPoseBackend(mode="lightweight", device="cpu").infer(frame)
    if not poses:
        raise DesktopSetupError("Проверка ML не обнаружила людей на контрольном кадре")
    report = {"status": "inference_smoke_passed", "accuracy_validated": False,
              "detected_people": len(poses), "seconds": round(time.monotonic() - start, 3)}
    model_path = paths.bundle / "models/acm40960-lstm-v1/model.onnx"
    if model_path.is_file():
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 25, 16), dtype=np.float32)})
        if not all(np.isfinite(output).all() for output in outputs):
            raise DesktopSetupError("Проверка классификатора вернула некорректные числа")
        report["classifier_execution"] = "passed_on_synthetic_input_not_accuracy"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Boxing Vision — локальное desktop-приложение"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check", action="store_true", help="Проверить комплект без открытия окна"
    )
    action.add_argument("--smoke-inference", action="store_true", help="Проверить исполнение моделей на кадре встроенного демо")
    action.add_argument(
        "--open-run", metavar="ID_OR_PATH",
        help="Открыть сохранённый анализ по ID или полному пути внутри каталога runs приложения",
    )
    args = parser.parse_args()
    try:
        paths = resolve_paths()
        prepare_environment(paths)
        logger = logging.getLogger("boxing_vision.desktop")
        logger.setLevel(logging.INFO)
        logger.addHandler(
            RotatingFileHandler(
                paths.logs / "desktop.log",
                maxBytes=2_000_000,
                backupCount=3,
                encoding="utf-8",
            )
        )
        if sys.stdout is None:
            sys.stdout = LogStream(logger)
        if sys.stderr is None:
            sys.stderr = LogStream(logger)
        if args.open_run is not None:
            prepare_saved_run(paths, args.open_run)
        manifest = verify_bundle(paths, required=bool(getattr(sys, "frozen", False)))
        seed_model_cache(paths, manifest)
        if manifest["offline_ready"]:
            verify_media_tools(paths)
        if args.smoke_inference:
            report = smoke_inference(paths)
            (paths.logs / "inference-smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return 0
        if args.check:
            if sys.stdout is not None:
                print(
                    json.dumps(
                        {
                            "status": manifest["status"],
                            "offline_ready": manifest["offline_ready"],
                        },
                        ensure_ascii=False,
                    )
                )
            return 0 if manifest["offline_ready"] else 2
        if args.open_run is None:
            prepare_bundled_demo(paths, manifest)
        launch_desktop(paths)
        return 0
    except (DesktopSetupError, ImportError, OSError, RuntimeError, ValueError) as exc:
        logging.getLogger("boxing_vision.desktop").exception("Desktop startup failed")
        message = f"Boxing Vision не запущен: {exc}\nНа Windows требуется Microsoft Edge WebView2 Runtime. Подробности — в logs/desktop.log."
        # Build/preflight probes must never wait for a native dialog click.
        if not (args.check or args.smoke_inference):
            show_startup_error(message)
        elif sys.stderr is not None:
            print(message, file=sys.stderr)
        return 1
