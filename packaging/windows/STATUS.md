# Windows delivery status

This branch contains the Windows application source and build tooling, **not a
verified Windows executable**. Local development is on macOS; the Windows
Actions validation workflow now runs native regression and installer checks.
Consult the actual run result, not the presence of the workflow, for evidence.

Implemented:

- Native pywebview/EdgeChromium shell, app-owned writable data directories,
  close/cancel handling, Windows-safe media subprocesses.
- PyInstaller onedir build and optional Inno Setup installer, explicit resource
  SHA256/provenance allowlist, saved-run opening.
- Installer source now includes an offline WebView2 prerequisite check/install,
  Microsoft publisher validation on the build host, and a two-file recipient
  folder generated only after compilation. Python/libraries are embedded, not
  installed globally. These installer paths still need native Windows testing.
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

2026-09-07: the production Inno script compiled on Windows Server 2022, its
Russian wizard reached completion, the installed inert test payload matched
its input, and uninstall succeeded (Actions run 34124692104). Five real wizard
screenshots were retained. This is installer-probe evidence, not application QA.

The user subsequently explicitly authorized encrypted transfer of the prepared
demo and requested SF Pro/Druk in a private investor build. The explicit
`--private-evaluation` path preserves `redistribution_approved: false`, the
authorization scope, original license and rights limitations. It does not
convert private use into a license grant or authorize public distribution.
The separate private build workflow returns only AES-GCM ciphertext; no release
EXE, unencrypted fonts or footage is uploaded publicly. Final app/ML QA is pending.

Local tests on macOS are regression evidence, not a Windows or ML quality gate.
The CI installer probe compiles the unchanged installer script with an inert
test payload, exercises the real wizard, and retains screenshots/logs only.
It does not publish an EXE and is not an investor build. A runner with WebView2
already installed does not test the missing-runtime/offline installation path.
The branch removes previously tracked environments, caches and run media from
its current tree. This does **not** erase files from older Git history or main.
