# Third-party notices

Boxing Vision is an investor prototype and does not redistribute model weights or FFmpeg.

- RTMLib 0.0.16 — Apache-2.0. The application uses its `Body(lightweight)` pipeline.
- YOLOX Tiny detector checkpoint — downloaded on first inference from OpenMMLab via RTMLib. Expected file SHA-256: `ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250`.
- RTMPose-S body checkpoint — downloaded on first inference from OpenMMLab via RTMLib. Expected file SHA-256: `9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d`.
- ONNX Runtime — MIT.
- OpenCV Python — Apache-2.0.
- Gradio — Apache-2.0.
- Roboflow Trackers 2.6.0 — Apache-2.0; BoT-SORT is shot-local and does not itself identify fighters.
- PySceneDetect 0.7.1 — BSD-3-Clause; AdaptiveDetector.
- ACM40960/Boxing — MIT, copyright (c) 2025 Jayaganeshan Thanga Kumar, Kabilesh Sekar. Local isolated adapter only, not the upstream Ultralytics application. Pinned source, SHA256 and original MIT notice are preserved in `models/acm40960-lstm-v1/manifest.json` and `LICENSE`. Training-data and weight commercial approval remain unverified. The bundle is ignored by Git and is not a distributable commercial approval.
- RF-DETR Nano — comparative benchmark only, not enabled in the application. See `models/rfdetr-nano/manifest.json` and `qa/ml-baseline/detector-comparison-20260904.json`; no quality winner without labelled ground truth.

## Demo portraits

These are interface-only demo portraits, not an assertion about who appears in an uploaded video. Custom portraits take precedence. They are never input to identity matching.

- `static/assets/demo-bivol.jpg`: Dmitry Bivol in 2023. Вячеслав Евдокимов / ФК Зенит, [source](https://commons.wikimedia.org/wiki/File:Dmitry_Bivol_in_2023.jpg), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Original downloaded unchanged; cropped by CSS for display.
- `static/assets/demo-usyk.jpg`: Oleksandr Usyk at TIFF 2025. Gabriel Hutchinson / WikiPortraits, [source](https://commons.wikimedia.org/wiki/File:Oleksandr_Usyk_at_TIFF_2025.jpg), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Original downloaded unchanged; cropped by CSS for display.

The registered body-map v3 layers were derived from one built-in ImageGen segmentation master. No Combat IQ artwork or video is included in the distributed app assets. `tools/build_body_atlas.py` records the extraction; all three layers share one 1024×1536 pixel grid.

The local Homebrew FFmpeg build used during development includes `libx264` and may be GPL-enabled. It is an external prerequisite, not bundled with this repository. Before commercial distribution, legal counsel should review the exact FFmpeg build, model-weight terms, all transitive dependencies and required attributions.
