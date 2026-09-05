# Boxing Vision for Windows x64

This is a **build framework**, not a Windows executable produced or verified on
the current Mac. The Gradio/ML application is hosted in a native pywebview window
with Windows title-bar controls, without browser tabs, address bar or toolbar.
Native mode hides the Gradio presentation footer. It does not replace the ML
pipeline or claim that tracking quality gates have passed.

## Architecture

- `desktop_app.py` is the frozen entrypoint; `boxing_vision.desktop` imports the
  UI only after configuring app-owned writable paths.
- One Gradio server listens on a random `127.0.0.1` port. Sharing is explicitly
  off. No Python object is exposed as a JavaScript API to the webview.
- EdgeChromium is selected explicitly on Windows. No Internet Explorer fallback.
- Closing the window requests analysis cancellation and closes the Gradio
  server. Downloads use the native WebView2 download flow.
- The bundle is read-only; runs, uploads, model cache, WebView state and rotated
  logs live under `%LOCALAPPDATA%\BoxingVision`.
- The UI uses supplied SF Pro Text regular/medium/semibold/bold files and Druk
  Medium/Bold for the existing numerical style. No Mac system fonts are silently
  substituted, downloaded, or copied. Missing required font inputs block a
  release build rather than pretending SF Pro is available.
- Window dimensions adapt to the primary monitor's **logical** size, with zoom
  enabled. Existing responsive UI handles narrow windows.

pywebview's [API](https://pywebview.flowrl.com/api/) defines native windows,
downloads, sizing and lifecycle callbacks. Its
[freezing guide](https://pywebview.flowrl.com/guide/freezing.html) recommends
PyInstaller for Windows. [PyInstaller is not a cross-compiler](https://pyinstaller.org/en/stable/):
a Windows executable must be built on a real Windows host.

## Required Windows environment

1. Windows 10/11 x64 with x64 Python **3.12**, a clean virtual environment, and
   Microsoft [Edge WebView2 Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/).
   pywebview's [engine requirements](https://pywebview.flowrl.com/guide/web_engine.html)
   also specify .NET Framework 4.6.2 or newer. These are prerequisites; this
   repository does not silently download or install them.
2. Windows x64 FFmpeg and ffprobe executables, with their required DLLs already
   resolved. For the first package use a self-contained/static distribution:
   only the two explicitly supplied executables are copied. FFmpeg must have
   **libx264 and AAC encoders**. A Mach-O binary renamed to `.exe` is rejected.
3. The two exact registered YOLOX/RTMPose ONNX weights from
   `inputs.example.json`. Optional punch classification additionally needs all
   three files from the validated `acm40960-lstm-v1` bundle: `model.onnx`,
   `manifest.json`, `LICENSE`.
4. The four supplied SF Pro Text and two Druk font files. A font-family name or
   macOS's `-apple-system` fallback is not a substitute for Windows font files.
5. Review permission to distribute **each** supplied binary, font and checkpoint;
   record it in the manifest. The validator checks your explicit approval, not
   the legal sufficiency of a license. Include applicable full notices/source
   offers before sharing an installer. Code licensing is not weight/data licensing.
6. Optional Inno Setup 6 with an explicit `ISCC.exe` path, for an installer.

## Build

### Optional preloaded investor example

Prepare a **new**, portable read-only snapshot from one explicitly chosen run:

```sh
python -m tools.prepare_investor_demo --run runs/CHOSEN_RUN --output packaging/investor-demo/sparring-v1 --start-ms 49583
```

Pass `-Demo packaging/investor-demo/sparring-v1` to the PowerShell build script
(or `--demo` to the Python builder). On launch the app opens the saved video,
timeline and statistics, paused at the prepared frame; no analysis or network
download is needed to view it. `--open-run` takes precedence over the example.
The example is read-only; **New analysis** opens the ordinary upload workflow.
Review counts, confidence and result restrictions remain unchanged. The snapshot
includes only the current tracking-only MP4, event/statistics JSON and preview
sprites, not source media, caches, old exports, logs or local filesystem paths.
Every included file has a SHA256 manifest. Existing runs are never overwritten.

The repository does not include the private sparring footage. Transfer the
prepared example privately to the Windows builder, and ensure permission for
each intended audience before sharing an installer containing the footage.

### Windows build host

In PowerShell from the repository root, on the Windows build host:

```powershell
py -3.12 -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv-win\Scripts\python.exe -m pip install -r packaging\windows\requirements.txt
```

Copy `packaging/windows/inputs.example.json` to your own input directory, fill in
the exact file paths, SHA256 values, provenance and permission decisions.
Relative `source` paths resolve relative to that manifest, not the current shell.
Use `Get-FileHash -Algorithm SHA256 <explicit-file>` to obtain a file's hash.
The example deliberately contains placeholders and `redistribution_approved:
false`; it is not an approved redistributable model/font bundle.

```powershell
.\tools\build_windows.ps1 -Python .\.venv-win\Scripts\python.exe `
  -Inputs C:\BoxingVisionInputs\inputs.json -ValidateOnly

.\tools\build_windows.ps1 -Python .\.venv-win\Scripts\python.exe `
  -Inputs C:\BoxingVisionInputs\inputs.json `
  -ISCC "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
```

The Python equivalent is `python -m tools.build_windows --inputs <manifest>`.
Each build gets a **new** subdirectory under `dist/windows`; old build outputs,
videos, `runs`, home directories and developer caches are never swept or copied.
No network CI or publishing job is started by these scripts.

The PyInstaller spec collects application static assets, Gradio metadata/templates,
RTMLib, ONNX Runtime, tracker dependencies and pywebview runtime resources. Qt,
CEF, PyTorch and TensorFlow are excluded: the shipping inference pipeline uses
ONNX, not the training environments. The explicit payload allowlist is copied
into the onedir bundle; input source paths are removed from its runtime manifest.

Outputs:

- `dist/windows/boxing-vision-build-*/dist/BoxingVision/BoxingVision.exe`
- the accompanying `_internal` directory — **do not send the exe alone**;
- optional `installer/BoxingVision-Setup-x64.exe`;
- `build-report.json`, recording hash and status `built_needs_windows_visual_and_video_QA`.

At first launch, verified detector/pose files are copied into the writable RTMLib
cache without network access. Integrity/resource preflight runs before UI/model
initialization. `BoxingVision.exe --check` validates the bundle and FFmpeg codecs;
it does not replace a real analysis/inference test. A missing complete bundle
fails clearly instead of claiming an offline-ready application.

## Reopen a saved analysis

Closing the window preserves analyses in `%LOCALAPPDATA%\BoxingVision\runs`.
There is no GUI history chooser yet, and the application does not automatically
open the latest run. Launch with an explicit run-directory ID or its full local
path, in PowerShell:

```powershell
& "$env:LOCALAPPDATA\Programs\BoxingVision\BoxingVision.exe" --open-run "YOUR_RUN_DIRECTORY_ID"
& "$env:LOCALAPPDATA\Programs\BoxingVision\BoxingVision.exe" --open-run "$env:LOCALAPPDATA\BoxingVision\runs\YOUR_RUN_DIRECTORY_ID"
```

For an onedir build, use that build's `BoxingVision.exe` instead. If explicitly
configured, `BOXING_VISION_DATA_DIR` changes the data root; the permitted results
remain inside its `runs` subdirectory. Network locations, outside paths, linked
artifacts and incomplete results are rejected. A saved result needs nonempty
`annotated.mp4`, `events.json` and `summary.json`. Cached observations are also
needed for render-only rebuilds or identity correction; simply opening the saved
video/statistics does not require a new ML pass or alter the saved result.

`--open-run` and `--check` are separate, mutually exclusive actions. Opening a
saved run does not substitute demo boxer portraits for its actual profiles.

## Release gate — still required on Windows

- Clean Windows account, no Python installed: launch from installer and from
  onedir, including a Unicode/spaced path; verify WebView2 missing-runtime error.
- Disconnect the network: open a local MP4, calibrate, process, review, export,
  restart with `--open-run <saved-run-ID>` and reopen the saved run. Check both
  model loading and FFmpeg DLLs.
- Play H.264/AAC, test upload and native download dialogs, seek, fullscreen and
  close/cancel during processing. Verify the loopback server exits with the app.
- SF Pro and Druk actually load at 100/125/150/200% Windows scaling; validate
  1920×1080, 1440×900, 1366×768 and a 768-pixel-wide window.
- No run, source video, token, user home path or private photo from the developer
  machine appears in the installer. Review the bundle manifest and onedir files.
- Measure the full ML acceptance/performance gate separately. Add code signing
  before wider distribution; this framework does not sign or upload installers.
- Uninstall must preserve `%LOCALAPPDATA%\BoxingVision` user data. It intentionally
  does not promise to erase analyses or remove user files.

Current status: launcher/build logic can be unit-tested on Mac; actual Windows
packaging, model execution, WebView playback and installer QA are **pending**.
