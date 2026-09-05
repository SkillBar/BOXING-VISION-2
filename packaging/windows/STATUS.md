# Windows delivery status

This branch contains the Windows application source and build tooling, **not a
verified Windows executable**. No Windows host was available for the local work.

Implemented:

- Native pywebview/EdgeChromium shell, app-owned writable data directories,
  close/cancel handling, Windows-safe media subprocesses.
- PyInstaller onedir build and optional Inno Setup installer, explicit resource
  SHA256/provenance allowlist, saved-run opening.
- Optional preloaded read-only demo with an initial paused video frame, timeline,
  statistics and preview sprites. No inference is needed to view the example.
- Non-blocking identity preflight: temporary missing fighters no longer abort
  analysis. Unknown/predicted poses still do not generate confirmed events.
- Shared full tracking rectangles, measured upper-body joints and bounded
  prediction; local identity recovery and cached review/render workflow.
- Tablet player height reserves space for timeline controls; wheel scrolling at
  FIT no longer intentionally captures page scrolling.

Required before calling a release ready:

1. Supply approved Windows FFmpeg, ONNX weights and font resources. The official
   SF Pro download does not confer permission for Windows embedding; approval
   remains unset. Do not mark resource rights approved simply to bypass a check.
2. Transfer an authorized demo privately to the Windows build machine, or get
   explicit permission to distribute that footage publicly. It is not in Git.
3. Build on Windows x64 and run `BoxingVision.exe --check`, then clean-machine
   installation, WebView2 playback, DPI/responsive and full inference tests.
4. Complete expert review of identity/event accuracy. The current measured
   algorithmic pair coverage is below the target; the presence of more boxes or
   candidates is not proof of correct identities or punches.

Local tests on macOS are regression evidence, not a Windows or ML quality gate.
The branch removes previously tracked environments, caches and run media from
its current tree. This does **not** erase files from older Git history or main.
