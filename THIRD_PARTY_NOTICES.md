# Third-party notices

Boxing Vision is an investor prototype and does not redistribute model weights or FFmpeg.

- RTMLib 0.0.16 — Apache-2.0. The application uses its `Body(lightweight)` pipeline.
- YOLOX Tiny detector checkpoint — downloaded on first inference from OpenMMLab via RTMLib. Expected file SHA-256: `ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250`.
- RTMPose-S body checkpoint — downloaded on first inference from OpenMMLab via RTMLib. Expected file SHA-256: `9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d`.
- ONNX Runtime — MIT.
- OpenCV Python — Apache-2.0.
- Gradio — Apache-2.0.

The local Homebrew FFmpeg build used during development includes `libx264` and may be GPL-enabled. It is an external prerequisite, not bundled with this repository. Before commercial distribution, legal counsel should review the exact FFmpeg build, model-weight terms, all transitive dependencies and required attributions.
