"""Pinned, opt-in third-party model provenance. No runtime downloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ACM_BUNDLE_ID = "acm40960-lstm-v1"
ACM_REVISION = "c9d94ffa0ae24b8065c6c40f5f9f8dc453619676"
ACM_REPOSITORY = "https://github.com/ACM40960/Boxing"
ACM_CHECKPOINT_SHA256 = "bd5b039c5fa04abc45b8f6fc1681dd10317ecd0b599a303748d22db3e510c4f2"
ACM_CHECKPOINT_URL = (
    f"https://raw.githubusercontent.com/ACM40960/Boxing/{ACM_REVISION}/app/punch_classifier.pth"
)
ACM_LICENSE_URL = f"https://raw.githubusercontent.com/ACM40960/Boxing/{ACM_REVISION}/LICENSE"
ACM_LABELS = ("jab", "cross", "hook", "uppercut")
ACM_JOINTS = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_acm_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Validate the registered input/output convention and exported file hash."""
    root = Path(bundle_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("bundle_id") != ACM_BUNDLE_ID:
        raise ValueError("Unknown punch model bundle")
    if manifest.get("source_revision") != ACM_REVISION:
        raise ValueError("Unregistered source revision")
    if manifest.get("source_sha256") != ACM_CHECKPOINT_SHA256:
        raise ValueError("Unregistered source checkpoint")
    if manifest.get("labels") != list(ACM_LABELS):
        raise ValueError("Punch label order does not match the checkpoint")
    expected_input = {
        "shape": [1, 25, 16], "fps": 30,
        "joints": list(ACM_JOINTS), "preprocessing": "acm-shoulder-width-v1",
    }
    if manifest.get("input") != expected_input:
        raise ValueError("Punch input convention does not match the checkpoint")
    model_path = root / "model.onnx"
    if not model_path.is_file() or model_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Punch ONNX model missing or unexpectedly large")
    if sha256_file(model_path) != manifest.get("onnx_sha256"):
        raise ValueError("Punch ONNX model failed SHA256 verification")
    if not (root / "LICENSE").is_file():
        raise ValueError("Third-party license missing from punch model bundle")
    return manifest
