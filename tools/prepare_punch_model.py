"""Prepare the pinned ACM LSTM in an isolated torch/onnx environment.

Example: models/.export-env/bin/python tools/prepare_punch_model.py
The runtime application requires only ONNX Runtime, not torch or Ultralytics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boxing_vision.model_registry import (
    ACM_BUNDLE_ID,
    ACM_CHECKPOINT_SHA256,
    ACM_CHECKPOINT_URL,
    ACM_JOINTS,
    ACM_LABELS,
    ACM_LICENSE_URL,
    ACM_REPOSITORY,
    ACM_REVISION,
    sha256_file,
)


def download(url: str, destination: Path, max_bytes: int) -> None:
    """Stage a bounded HTTPS download before replacing the model artifact."""
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Model download exceeded expected size")
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".download")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(destination: Path, *, offline: bool = False) -> dict:
    import onnx
    import onnxruntime as ort
    import torch
    from torch import nn

    class ActionClassifierLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(16, 128, 2, batch_first=True, dropout=.5)
            self.fc = nn.Linear(128, 4)

        def forward(self, inputs):
            hidden = torch.zeros(2, 1, 128, device=inputs.device)
            cell = torch.zeros(2, 1, 128, device=inputs.device)
            output, _ = self.lstm(inputs, (hidden, cell))
            return self.fc(output[:, -1, :])

    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "punch_classifier.pth"
    if not checkpoint.exists():
        if offline:
            raise FileNotFoundError(checkpoint)
        download(ACM_CHECKPOINT_URL, checkpoint, 1024 * 1024)
    if sha256_file(checkpoint) != ACM_CHECKPOINT_SHA256:
        raise ValueError("Source checkpoint failed pinned SHA256 verification")
    license_path = destination / "LICENSE"
    if not license_path.exists():
        if offline:
            raise FileNotFoundError(license_path)
        download(ACM_LICENSE_URL, license_path, 16 * 1024)
    if "MIT License" not in license_path.read_text(encoding="utf-8"):
        raise ValueError("Source MIT license missing")

    torch.set_num_threads(1)
    model = ActionClassifierLSTM().eval()
    # weights_only=True prohibits arbitrary pickle globals; only this exact,
    # independently pinned checkpoint is accepted.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model_path = destination / "model.onnx"
    temporary_path = destination / "model.pending.onnx"
    example = torch.zeros(1, 25, 16, dtype=torch.float32)
    torch.onnx.export(
        model, (example,), str(temporary_path), input_names=["poses"],
        output_names=["logits"], opset_version=17, dynamo=False,
        do_constant_folding=True,
    )
    graph = onnx.load(str(temporary_path), load_external_data=False)
    if any(value.data_location == onnx.TensorProto.EXTERNAL for value in graph.graph.initializer):
        raise ValueError("External ONNX data is not supported")
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(temporary_path), options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    max_error = 0.0
    timings = []
    with torch.no_grad():
        for _ in range(32):
            inputs = rng.normal(size=(1, 25, 16)).astype(np.float32)
            reference = model(torch.from_numpy(inputs)).numpy()
            start = time.perf_counter()
            output = session.run(None, {"poses": inputs})[0]
            timings.append(1000 * (time.perf_counter() - start))
            max_error = max(max_error, float(np.max(np.abs(reference - output))))
            np.testing.assert_allclose(reference, output, rtol=1e-4, atol=1e-4)
    os.replace(temporary_path, model_path)
    manifest = {
        "version": 1, "bundle_id": ACM_BUNDLE_ID,
        "source_repository": ACM_REPOSITORY, "source_revision": ACM_REVISION,
        "source_url": ACM_CHECKPOINT_URL, "source_sha256": ACM_CHECKPOINT_SHA256,
        "onnx_sha256": sha256_file(model_path), "labels": list(ACM_LABELS),
        "code_license": "MIT", "license_url": ACM_LICENSE_URL,
        "weight_license": "repository MIT; separate weight/data grant not documented",
        "training_datasets": "Author-collected boxing actions; full provenance not published",
        "commercial_approval": False, "status": "local_research_candidate",
        "calibration": "none; softmax scores are not calibrated probabilities",
        "purpose": "technique refinement of existing verified punch candidates only",
        "limitations": [
            "No background, hand or contact outcome head",
            "RTMPose input distribution differs from source YOLO11 poses",
            "Broadcast accuracy has not been measured on labeled held-out fights",
            "Sparse interpolated input is diagnostic only",
        ],
        "input": {"shape": [1, 25, 16], "fps": 30, "joints": list(ACM_JOINTS), "preprocessing": "acm-shoulder-width-v1"},
        "export": {
            "torch": torch.__version__, "onnx": onnx.__version__,
            "onnxruntime": ort.__version__, "opset": 17,
            "cpu_parity_max_absolute_error": max_error,
            "synthetic_inference_median_ms": float(np.median(timings)),
            "synthetic_inference_p95_ms": float(np.percentile(timings, 95)),
        },
    }
    manifest_path = destination / "manifest.json"
    temporary_manifest = destination / "manifest.pending.json"
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models") / ACM_BUNDLE_ID)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, offline=args.offline), ensure_ascii=False, indent=2))
