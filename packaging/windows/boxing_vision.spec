# PyInstaller one-directory build. The payload is an explicit validated allowlist.
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

root = Path(os.environ["BOXING_VISION_BUILD_ROOT"]).resolve()
payload = Path(os.environ["BOXING_VISION_BUILD_PAYLOAD"]).resolve()
if not (payload / "bundle-manifest.json").is_file():
    raise RuntimeError("Run tools.build_windows with validated explicit inputs")

datas = [(str(root / "boxing_vision/static"), "boxing_vision/static")]
datas.append((str(root / "THIRD_PARTY_NOTICES.md"), "notices"))
binaries = []
hiddenimports = []
for package in ("gradio", "gradio_client", "rtmlib", "onnxruntime", "trackers", "supervision", "scenedetect", "webview"):
    package_data, package_binaries, package_hidden = collect_all(package)
    datas += package_data
    binaries += package_binaries
    hiddenimports += package_hidden
for distribution in ("gradio", "gradio_client", "rtmlib", "onnxruntime", "trackers", "supervision", "scenedetect", "pywebview"):
    datas += copy_metadata(distribution, recursive=True)
# Package WebView2 loader/.NET resources; Setup supplies the missing OS runtime.
datas += collect_data_files("pythonnet")
# safehttpx reads version.txt during import; metadata alone does not include it.
datas += collect_data_files("safehttpx")
datas += collect_data_files("groovy")
for item in payload.rglob("*"):
    if item.is_file() and "prerequisites" not in item.relative_to(payload).parts:
        datas.append((str(item), str(item.parent.relative_to(payload))))

a = Analysis(
    [str(root / "desktop_app.py")], pathex=[str(root)], binaries=binaries,
    datas=datas, hiddenimports=hiddenimports,
    hookspath=[], runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "cefpython3", "torch", "tensorflow", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="BoxingVision",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="BoxingVision")
