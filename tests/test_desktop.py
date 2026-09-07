"""Native-shell/build contracts; no Windows host or GUI is simulated as proof."""

from __future__ import annotations

import json
import logging
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from boxing_vision import desktop
from tools import build_windows


def _paths(tmp_path: Path) -> desktop.DesktopPaths:
    bundle = tmp_path / "application"
    bundle.mkdir(exist_ok=True)
    return desktop.DesktopPaths(bundle, tmp_path / "appdata", bundle / "fonts")


def _pe(machine: int = 0x8664) -> bytes:
    data = bytearray(256)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 128)
    data[128:134] = b"PE\x00\x00" + struct.pack("<H", machine)
    return bytes(data)


def _input_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    records = []
    pins = {}
    for role in sorted(build_windows.REQUIRED_ROLES):
        if role in {"ffmpeg", "ffprobe"}:
            source, data = inputs / f"{role}.exe", _pe()
        elif role in {"detector", "pose"}:
            source, data = inputs / f"{role}.onnx", role.encode()
        else:
            source, data = inputs / f"{role}.ttf", b"\x00\x01\x00\x00" + role.encode()
        source.write_bytes(data)
        digest = desktop.sha256_file(source)
        if role in {"detector", "pose"}:
            pins[role] = (source.name, digest)
        records.append(
            {
                "source": f"inputs/{source.name}",
                "role": role,
                "sha256": digest,
                "license": "Test fixture only",
                "source_url": "https://example.test/runtime",
                "redistribution_approved": True,
            }
        )
    monkeypatch.setattr(desktop, "RTM_MODEL_REQUIREMENTS", pins)
    monkeypatch.setattr(build_windows, "RTM_MODEL_REQUIREMENTS", pins)
    manifest = tmp_path / "inputs.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "platform": "windows-x64", "files": records})
    )
    return manifest


def _mutate_manifest(path: Path, mutate) -> None:
    document = json.loads(path.read_text())
    mutate(document)
    path.write_text(json.dumps(document))


def test_offline_webview_installer_is_setup_only_not_an_application_dependency(tmp_path, monkeypatch):
    inputs = _input_fixture(tmp_path, monkeypatch)
    runtime = tmp_path / "WebView2Standalone.exe"
    runtime.write_bytes(_pe(machine=0x014c))  # Container architecture != runtime payload architecture.
    _mutate_manifest(inputs, lambda doc: doc["files"].append({
        "source": str(runtime), "role": "webview2", "sha256": desktop.sha256_file(runtime),
        "license": "Test fixture only", "source_url": "https://developer.microsoft.com/microsoft-edge/webview2/",
        "redistribution_approved": True,
    }))
    records = build_windows.validated_inputs(inputs)
    paths = _paths(tmp_path)
    build_windows.stage_inputs(records, paths.bundle)
    assert (paths.bundle / "prerequisites/WebView2RuntimeInstaller.exe").is_file()
    bundled = json.loads((paths.bundle / "bundle-manifest.json").read_text())
    assert "webview2" not in {r["role"] for r in bundled["files"]}
    assert desktop.verify_bundle(paths)["offline_ready"] is True


class _Event:
    def __init__(self):
        self.callbacks = []

    def __iadd__(self, callback):
        self.callbacks.append(callback)
        return self


class _App:
    def __init__(self, url="http://127.0.0.1:43210/", share=None):
        self.url, self.share, self.closed = url, share, 0

    def queue(self, **kwargs):
        self.queue_kwargs = kwargs
        return self

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs
        return None, self.url, self.share

    def close(self, **kwargs):
        assert kwargs == {"verbose": False}
        self.closed += 1


class _Webview:
    def __init__(self, error=None):
        self.settings = {}
        self.screens = [SimpleNamespace(width=1024, height=768)]
        self.window = SimpleNamespace(events=SimpleNamespace(closed=_Event()))
        self.error = error

    def create_window(self, title, url, **kwargs):
        self.title, self.url, self.window_kwargs = title, url, kwargs
        return self.window

    def start(self, **kwargs):
        self.start_kwargs = kwargs
        if self.error:
            raise self.error
        for callback in self.window.events.closed.callbacks:
            callback()


def test_windows_data_and_fonts_are_separate_from_read_only_application(tmp_path):
    paths = desktop.resolve_paths(
        environ={"LOCALAPPDATA": str(tmp_path / "local")},
        platform="win32",
        bundle=tmp_path / "app",
    )
    assert paths.data == tmp_path / "local" / "BoxingVision"
    assert paths.fonts == tmp_path / "app" / "fonts"
    assert not paths.data.exists()  # Resolving paths has no write side effect.


def test_explicit_private_paths_take_precedence(tmp_path):
    paths = desktop.resolve_paths(
        environ={
            "BOXING_VISION_DATA_DIR": str(tmp_path / "data"),
            "BOXING_VISION_FONT_DIR": str(tmp_path / "private-fonts"),
        },
        platform="win32",
        bundle=tmp_path / "app",
    )
    assert paths.data == tmp_path / "data"
    assert paths.fonts == tmp_path / "private-fonts"


@pytest.mark.parametrize("suffix", ["", "/runs"])
def test_refuses_writable_data_inside_application(tmp_path, suffix):
    with pytest.raises(desktop.DesktopSetupError):
        desktop.resolve_paths(
            environ={"BOXING_VISION_DATA_DIR": str(tmp_path / "app") + suffix},
            bundle=tmp_path / "app",
        )


def test_windows_without_localappdata_has_actionable_failure(tmp_path):
    with pytest.raises(desktop.DesktopSetupError, match="LOCALAPPDATA"):
        desktop.resolve_paths(environ={}, platform="win32", bundle=tmp_path / "app")


def test_environment_is_ready_before_lazy_ui_import(tmp_path):
    paths, env = _paths(tmp_path), {"PATH": "original-path"}
    (paths.bundle / "bin").mkdir()
    desktop.prepare_environment(paths, environ=env)
    assert env["BOXING_VISION_DESKTOP"] == "1"
    assert env["BOXING_VISION_FONT_DIR"] == str(paths.fonts)
    assert env["BOXING_VISION_DATA_DIR"] == str(paths.data)
    assert env["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert env["TORCH_HOME"] == str(paths.cache / "rtmlib")
    assert env["PATH"].startswith(str(paths.bundle / "bin"))
    assert (paths.cache / "rtmlib/hub/checkpoints").is_dir()
    assert paths.runs.is_dir() and paths.logs.is_dir()
    assert not paths.fonts.exists()  # No font discovery, copying or synthesis.


@pytest.mark.parametrize(
    "relative",
    [
        "",
        "../video.mp4",
        "/video.mp4",
        "C:\\private\\model.onnx",
        "C:model.onnx",
        "fonts/../../secret",
        "..\\secret",
    ],
)
def test_resource_paths_cannot_escape_bundle(tmp_path, relative):
    with pytest.raises(desktop.DesktopSetupError):
        desktop.safe_bundle_file(tmp_path, relative)


def test_resource_symlink_cannot_escape_bundle(tmp_path):
    paths = _paths(tmp_path)
    (paths.bundle / "escape").symlink_to(tmp_path)
    with pytest.raises(desktop.DesktopSetupError):
        desktop.safe_bundle_file(paths.bundle, "escape/secret")


def test_known_model_pins_match_current_runtime():
    from boxing_vision.pose import _LIGHTWEIGHT_MODEL_HASHES

    assert set(desktop.RTM_MODEL_REQUIREMENTS.values()) == set(
        _LIGHTWEIGHT_MODEL_HASHES.values()
    )


def test_complete_allowlist_verifies_and_seeds_only_two_model_files(
    tmp_path, monkeypatch
):
    inputs = _input_fixture(tmp_path, monkeypatch)
    paths = _paths(tmp_path)
    desktop.prepare_environment(paths, environ={})
    (tmp_path / "private-video.mp4").write_bytes(b"private video is not a build asset")
    build_windows.stage_inputs(build_windows.validated_inputs(inputs), paths.bundle)
    manifest = desktop.verify_bundle(paths)
    assert manifest["offline_ready"] is True
    assert str(tmp_path) not in (paths.bundle / "bundle-manifest.json").read_text()
    desktop.seed_model_cache(paths, manifest)
    cache = paths.cache / "rtmlib/hub/checkpoints"
    assert sorted(path.name for path in cache.iterdir()) == [
        "detector.onnx",
        "pose.onnx",
    ]
    assert not (paths.bundle / "private-video.mp4").exists()
    prior_mtime = (cache / "pose.onnx").stat().st_mtime_ns
    desktop.seed_model_cache(paths, manifest)
    assert (cache / "pose.onnx").stat().st_mtime_ns == prior_mtime
    (cache / "pose.onnx").write_bytes(b"broken cached copy")
    desktop.seed_model_cache(paths, manifest)
    assert (cache / "pose.onnx").read_bytes() == b"pose"


def test_corrupted_resource_blocks_offline_claim(tmp_path, monkeypatch):
    inputs = _input_fixture(tmp_path, monkeypatch)
    paths = _paths(tmp_path)
    build_windows.stage_inputs(build_windows.validated_inputs(inputs), paths.bundle)
    (paths.bundle / "fonts/SF-Pro-Text-Regular.ttf").write_bytes(b"wrong")
    with pytest.raises(desktop.DesktopSetupError, match="повреждён"):
        desktop.verify_bundle(paths)


@pytest.mark.parametrize(
    "document",
    [
        [],
        {"schema_version": 2},
        {"schema_version": 1, "platform": "windows-x64", "files": None},
        {"schema_version": 1, "platform": "windows-x64", "files": [None]},
    ],
)
def test_malformed_bundle_fails_with_setup_error(tmp_path, document):
    paths = _paths(tmp_path)
    (paths.bundle / "bundle-manifest.json").write_text(json.dumps(document))
    with pytest.raises(desktop.DesktopSetupError):
        desktop.verify_bundle(paths)


def test_missing_bundle_is_not_presented_as_offline_ready(tmp_path):
    paths = _paths(tmp_path)
    assert desktop.verify_bundle(paths, required=False)["offline_ready"] is False
    with pytest.raises(desktop.DesktopSetupError, match="bundle-manifest"):
        desktop.verify_bundle(paths)


def test_native_window_has_no_python_bridge_and_only_loopback_server(tmp_path):
    paths, app, webview, cancellations = _paths(tmp_path), _App(), _Webview(), []
    desktop.launch_desktop(
        paths,
        app_factory=lambda: app,
        webview_module=webview,
        cancel=lambda: cancellations.append(True),
        port_factory=lambda: 43210,
    )
    assert app.launch_kwargs["server_name"] == "127.0.0.1"
    assert app.launch_kwargs["server_port"] == 43210
    assert app.launch_kwargs["share"] is False
    assert app.launch_kwargs["inbrowser"] is False
    assert app.launch_kwargs["prevent_thread_lock"] is True
    assert webview.window_kwargs["js_api"] is None
    assert webview.window_kwargs["frameless"] is False
    assert webview.window_kwargs["zoomable"] is True
    assert webview.settings["ALLOW_DOWNLOADS"] is True
    assert webview.settings["ALLOW_FILE_URLS"] is False
    assert webview.start_kwargs["debug"] is False
    assert webview.start_kwargs["storage_path"] == str(paths.cache / "webview")
    assert app.closed == 1 and cancellations == [True]


@pytest.mark.parametrize(
    "url",
    [
        "https://example.test/",
        "http://0.0.0.0:43210/",
        "http://127.0.0.1:80/",
        "http://user@127.0.0.1:43210/",
    ],
)
def test_remote_or_wrong_port_cannot_become_native_content(tmp_path, url):
    app = _App(url=url)
    with pytest.raises(desktop.DesktopSetupError):
        desktop.launch_desktop(
            _paths(tmp_path),
            app_factory=lambda: app,
            webview_module=_Webview(),
            port_factory=lambda: 43210,
        )
    assert app.closed == 1


def test_webview_failure_does_not_leave_gradio_running(tmp_path):
    app = _App()
    with pytest.raises(RuntimeError, match="WebView2"):
        desktop.launch_desktop(
            _paths(tmp_path),
            app_factory=lambda: app,
            webview_module=_Webview(RuntimeError("WebView2 missing")),
            port_factory=lambda: 43210,
        )
    assert app.closed == 1


def test_cancellation_failure_still_closes_server_once():
    app = _App()

    def fail():
        raise RuntimeError("cancel failure")

    session = desktop.DesktopSession(app, fail)
    with pytest.raises(RuntimeError):
        session.close()
    session.close()
    assert app.closed == 1


@pytest.mark.parametrize(
    "width,height", [(1920, 1080), (1440, 900), (1024, 768), (768, 900), (512, 384)]
)
def test_native_geometry_fits_small_and_scaled_monitor(width, height):
    geometry = desktop.window_geometry([SimpleNamespace(width=width, height=height)])
    assert geometry["width"] <= width and geometry["height"] <= height
    assert geometry["min_size"][0] <= geometry["width"]
    assert geometry["min_size"][1] <= geometry["height"]


def test_windowed_console_replacement_handles_print_and_progress(caplog):
    stream = desktop.LogStream(logging.getLogger("boxing_vision.desktop.test"))
    with caplog.at_level(logging.INFO):
        assert stream.write("Progress 25%\n") == len("Progress 25%\n")
        stream.flush()
    assert stream.encoding == "utf-8" and stream.isatty() is False
    assert "Progress 25%" in caplog.text


def test_media_preflight_requires_h264_and_aac(tmp_path):
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        assert kwargs["timeout"] == 10 and kwargs["check"] is False
        output = (
            " V....D libx264 encoder\n A..... aac native encoder\n"
            if "-encoders" in args
            else Path(args[0]).stem + " version 8"
        )
        return SimpleNamespace(returncode=0, stdout=output)

    desktop.verify_media_tools(_paths(tmp_path), runner=runner)
    assert len(calls) == 3


@pytest.mark.parametrize(
    "error", [OSError("missing DLL"), subprocess.TimeoutExpired("ffmpeg", 10)]
)
def test_media_tool_errors_produce_setup_message(tmp_path, error):
    def runner(*args, **kwargs):
        raise error

    with pytest.raises(desktop.DesktopSetupError, match="FFmpeg"):
        desktop.verify_media_tools(_paths(tmp_path), runner=runner)


def test_missing_aac_encoder_blocks_preflight(tmp_path):
    def runner(args, **kwargs):
        output = "libx264" if "-encoders" in args else Path(args[0]).stem + " version 8"
        return SimpleNamespace(returncode=0, stdout=output)

    with pytest.raises(desktop.DesktopSetupError, match="AAC"):
        desktop.verify_media_tools(_paths(tmp_path), runner=runner)


@pytest.mark.parametrize(
    "content,expected",
    [
        (_pe(), True),
        (_pe(0x014C), False),
        (b"\xcf\xfa\xed\xfe", False),
        (b"MZshort", False),
    ],
)
def test_build_rejects_mac_binary_renamed_as_exe(tmp_path, content, expected):
    path = tmp_path / "ffmpeg.exe"
    path.write_bytes(content)
    assert build_windows.windows_pe_x64(path) is expected


@pytest.mark.parametrize(
    "problem",
    ["missing_font", "permission", "hash", "extra_video", "duplicate", "partial_punch"],
)
def test_build_inputs_fail_closed_before_copying(tmp_path, monkeypatch, problem):
    inputs = _input_fixture(tmp_path, monkeypatch)

    def mutate(document):
        records = document["files"]
        if problem == "missing_font":
            document["files"] = [row for row in records if row["role"] != "sf_regular"]
        elif problem == "permission":
            records[0]["redistribution_approved"] = False
        elif problem == "hash":
            records[0]["sha256"] = "0" * 64
        elif problem == "extra_video":
            records.append({"role": "user_video", "source": "private.mp4"})
        elif problem == "duplicate":
            records.append(records[0])
        else:
            records.append({"role": "punch_model", "source": "model.onnx"})

    _mutate_manifest(inputs, mutate)
    with pytest.raises(desktop.DesktopSetupError):
        build_windows.validated_inputs(inputs)
    assert not (tmp_path / "dist").exists()


def test_mac_build_is_explicitly_not_a_windows_cross_compile(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(build_windows.sys, "platform", "darwin")
    monkeypatch.setattr(
        build_windows.sys,
        "argv",
        ["build_windows", "--inputs", str(tmp_path / "missing.json")],
    )
    with pytest.raises(SystemExit) as error:
        build_windows.main()
    assert error.value.code == 2
    assert "not a Windows cross-compiler" in capsys.readouterr().err


@pytest.mark.parametrize(
    "document",
    [
        {"source_path": "relative/private-training.mp4"},
        {"metadata": [{"source_path": "something"}]},
        {"extra": "/Users/private/training.mp4"},
        {"extra": "C:\\Users\\Private\\training.mp4"},
    ],
)
def test_optional_model_manifest_cannot_leak_local_paths(document):
    with pytest.raises(desktop.DesktopSetupError, match="path"):
        build_windows.validate_public_model_manifest(document)


def test_public_provenance_urls_and_hashes_remain_unchanged():
    document = {
        "source_url": "https://github.com/example/model",
        "sha256": "a" * 64,
        "input": {"shape": [1, 25, 16]},
    }
    before = json.dumps(document)
    build_windows.validate_public_model_manifest(document)
    assert json.dumps(document) == before


def _saved_run(paths: desktop.DesktopPaths, name: str = "20260904-demo") -> Path:
    run = paths.runs / name
    run.mkdir(parents=True)
    (run / "annotated.mp4").write_bytes(b"saved-video-fixture-not-decoded")
    (run / "events.json").write_text("[]", encoding="utf-8")
    (run / "summary.json").write_text(
        '{"metadata":{"duration_s":12,"export_stale":true}}', encoding="utf-8"
    )
    return run


@pytest.fixture(autouse=True)
def close_test_desktop_log_handlers():
    """Do not leave Windows test directories held by repeated main() checks."""
    logger = logging.getLogger("boxing_vision.desktop")
    existing = set(logger.handlers)
    yield
    for handler in list(logger.handlers):
        if handler not in existing:
            logger.removeHandler(handler)
            handler.close()


@pytest.mark.parametrize("absolute", [False, True])
def test_explicit_saved_run_accepts_id_or_absolute_local_path(tmp_path, absolute):
    paths = _paths(tmp_path)
    run = _saved_run(paths, "Бой с пробелами")
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    env = {"UNCHANGED": "value"}
    selected = desktop.prepare_saved_run(paths, str(run) if absolute else run.name, environ=env)
    assert selected == run.resolve()
    assert env == {
        "UNCHANGED": "value",
        "BOXING_VISION_DEMO_RUN": str(run.resolve()),
        "BOXING_VISION_OPEN_SAVED_RUN": "1",
    }
    assert {path.name: path.read_bytes() for path in run.iterdir()} == before


@pytest.mark.parametrize("value", ["", " ", ".", "..", "../outside", "nested/run", "nested\\run", "C:run", "bad\x00run"])
def test_saved_run_rejects_ambiguous_relative_paths_without_environment_change(tmp_path, value):
    paths = _paths(tmp_path)
    _saved_run(paths)
    env = {"BOXING_VISION_DEMO_RUN": "previous-selection"}
    with pytest.raises(desktop.DesktopSetupError):
        desktop.prepare_saved_run(paths, value, environ=env)
    assert env == {"BOXING_VISION_DEMO_RUN": "previous-selection"}


@pytest.mark.parametrize("value", ["https://example.test/run", "file:///run", "//server/share/run", "\\\\server\\share\\run"])
def test_saved_run_rejects_remote_locations_before_filesystem_access(tmp_path, monkeypatch, value):
    paths = _paths(tmp_path)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Remote run validation must not access the filesystem")

    monkeypatch.setattr(Path, "resolve", unexpected)
    with pytest.raises(desktop.DesktopSetupError, match="сетевому"):
        desktop.resolve_saved_run(paths, value)


def test_saved_run_rejects_outside_root_and_root_itself(tmp_path):
    paths = _paths(tmp_path)
    _saved_run(paths)
    outside = tmp_path / "outside"
    outside.mkdir()
    for path in (outside, paths.runs):
        with pytest.raises(desktop.DesktopSetupError, match="внутри"):
            desktop.resolve_saved_run(paths, str(path))


def test_saved_run_rejects_run_directory_link_escaping_allowed_root(tmp_path):
    paths = _paths(tmp_path)
    _saved_run(paths)
    outside = tmp_path / "outside"
    outside.mkdir()
    (paths.runs / "linked-run").symlink_to(outside, target_is_directory=True)
    with pytest.raises(desktop.DesktopSetupError, match="внутри"):
        desktop.resolve_saved_run(paths, "linked-run")


@pytest.mark.parametrize("kind", ["file", "directory", "dangling", "loop"])
def test_saved_run_rejects_linked_media_and_cache_artifacts(tmp_path, kind):
    paths = _paths(tmp_path)
    run = _saved_run(paths)
    if kind == "file":
        (run / "workspace-preview.mp4").symlink_to(run / "annotated.mp4")
    elif kind == "directory":
        (run / "clips").symlink_to(tmp_path, target_is_directory=True)
    elif kind == "dangling":
        (run / "preview.webp").symlink_to(tmp_path / "missing.webp")
    else:
        (run / "loop").symlink_to(run, target_is_directory=True)
    with pytest.raises(desktop.DesktopSetupError, match="ссылки"):
        desktop.resolve_saved_run(paths, run.name)


@pytest.mark.parametrize("missing", ["annotated.mp4", "events.json", "summary.json"])
@pytest.mark.parametrize("empty", [False, True])
def test_saved_run_requires_nonempty_result_artifacts(tmp_path, missing, empty):
    paths = _paths(tmp_path)
    run = _saved_run(paths)
    if empty:
        (run / missing).write_bytes(b"")
    else:
        (run / missing).unlink()
    with pytest.raises(desktop.DesktopSetupError, match=missing.replace(".", r"\.")):
        desktop.resolve_saved_run(paths, run.name)


@pytest.mark.parametrize("filename,content", [
    ("events.json", "{"), ("events.json", "{}"), ("events.json", "[null]"),
    ("summary.json", "{"), ("summary.json", "[]"), ("summary.json", "null"),
])
def test_saved_run_rejects_broken_json_before_loading_ui(tmp_path, filename, content):
    paths = _paths(tmp_path)
    run = _saved_run(paths)
    (run / filename).write_text(content, encoding="utf-8")
    with pytest.raises(desktop.DesktopSetupError):
        desktop.resolve_saved_run(paths, run.name)


def test_saved_run_preparation_happens_before_native_ui_launch(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    run = _saved_run(paths)
    monkeypatch.setattr(desktop.sys, "argv", ["BoxingVision.exe", "--open-run", run.name])
    monkeypatch.setattr(desktop, "resolve_paths", lambda: paths)
    env = {}
    monkeypatch.setattr(desktop.os, "environ", env)
    calls = []

    def launch(selected_paths):
        assert selected_paths == paths
        assert env["BOXING_VISION_DEMO_RUN"] == str(run.resolve())
        assert env["BOXING_VISION_OPEN_SAVED_RUN"] == "1"
        calls.append("launch")

    monkeypatch.setattr(desktop, "launch_desktop", launch)
    assert desktop.main() == 0
    assert calls == ["launch"]
    assert not (run / ".render_cache").exists()  # Opening needs no model/cached observations.


@pytest.mark.parametrize("selection", ["missing", ""])
def test_saved_run_invalid_input_never_launches_native_ui(tmp_path, monkeypatch, selection):
    paths = _paths(tmp_path)
    monkeypatch.setattr(desktop.sys, "argv", ["BoxingVision.exe", "--open-run", selection])
    monkeypatch.setattr(desktop, "resolve_paths", lambda: paths)
    monkeypatch.setattr(desktop.os, "environ", {})

    def unexpected(*_args, **_kwargs):
        raise AssertionError("An invalid saved run must not launch UI/inference")

    monkeypatch.setattr(desktop, "launch_desktop", unexpected)
    assert desktop.main() == 1


def test_open_run_cannot_be_combined_with_resource_check(monkeypatch):
    monkeypatch.setattr(desktop.sys, "argv", ["BoxingVision.exe", "--check", "--open-run", "run"])
    with pytest.raises(SystemExit) as result:
        desktop.main()
    assert result.value.code == 2


def test_no_run_argument_does_not_autoselect_existing_result(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _saved_run(paths)
    env = {}
    monkeypatch.setattr(desktop.os, "environ", env)
    monkeypatch.setattr(desktop.sys, "argv", ["BoxingVision.exe"])
    monkeypatch.setattr(desktop, "resolve_paths", lambda: paths)

    def launch(_paths):
        assert "BOXING_VISION_DEMO_RUN" not in env
        assert "BOXING_VISION_OPEN_SAVED_RUN" not in env

    monkeypatch.setattr(desktop, "launch_desktop", launch)
    assert desktop.main() == 0


@pytest.mark.parametrize("restored,expected_demo", [("1", False), ("0", True)])
def test_reopening_saved_result_does_not_assign_demo_portraits(tmp_path, monkeypatch, restored, expected_demo):
    from boxing_vision import ui

    run = tmp_path / "saved-run"
    monkeypatch.setenv("BOXING_VISION_DEMO_RUN", str(run))
    monkeypatch.setenv("BOXING_VISION_OPEN_SAVED_RUN", restored)
    captured = []

    def payload(_events, summary, _duration, **_kwargs):
        captured.append(summary)
        return {}

    monkeypatch.setattr(ui, "build_presentation_payload", payload)
    monkeypatch.setattr(ui, "render_workspace_shell", lambda _payload: "workspace")
    assert ui._workspace([], {"metadata": {}}, 12, run) == "workspace"
    assert captured[0]["metadata"]["demo_portraits"] is expected_demo
