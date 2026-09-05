"""Isolated RF-DETR export and reproducible person-detector latency comparison.

Export with models/.export-env (rfdetr[onnx]==1.10.0); benchmark with the
application's ONNX-only environment. No installation or download occurs here.
No ground truth is inferred from detections; counts are not accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RFDETR_SHA256 = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"
RFDETR_URL = (
    "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def export_rfdetr(directory: Path) -> dict:
    """Uses the upstream safe loader (trust_checkpoint=False), pinned bytes."""
    import importlib.metadata

    if importlib.metadata.version("rfdetr") != "1.10.0":
        raise ValueError("Export requires the isolated rfdetr==1.10.0 environment")
    checkpoint = directory / "rf-detr-nano.pth"
    if digest(checkpoint) != RFDETR_SHA256:
        raise ValueError(
            "RF-DETR checkpoint SHA256 differs from pinned official weights"
        )
    import torch
    from rfdetr import RFDETRNano

    torch.set_num_threads(4)
    model = RFDETRNano(
        pretrain_weights=str(checkpoint.resolve()), device="cpu", trust_checkpoint=False
    )
    started = time.perf_counter()
    output = Path(
        model.export(
            output_dir=str(directory.resolve()),
            batch_size=1,
            dynamic_batch=False,
            shape=(384, 384),
            verbose=False,
            fp16=False,
            output_name="rfdetr-nano",
        )
    )
    manifest = {
        "model_id": "rfdetr-nano-coco-1.10.0",
        "package_version": "1.10.0",
        "source_url": RFDETR_URL,
        "checkpoint_sha256": RFDETR_SHA256,
        "onnx_filename": output.name,
        "onnx_sha256": digest(output),
        "code_license": "Apache-2.0",
        "weights_license": "Apache-2.0",
        "license_source": "https://rfdetr.roboflow.com/learn/pretrained/",
        "training_datasets": ["COCO"],
        "commercial_approval": False,
        "approval_note": "Upstream permissive pretrained model; project legal approval and input-video rights remain separate.",
        "input": {
            "shape": [1, 3, 384, 384],
            "rgb": True,
            "resize": "float32 bilinear square, antialias=False",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "output": {
            "boxes": "dets, normalized cxcywh",
            "logits": "labels",
            "person_class_id": 1,
        },
        "export_seconds": time.perf_counter() - started,
        "status": "benchmark_candidate_not_application_default",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class RFDetrPeople:
    def __init__(self, directory: Path, threshold: float, threads: int):
        import onnxruntime as ort

        self.manifest = json.loads((directory / "manifest.json").read_text())
        model = directory / self.manifest["onnx_filename"]
        if digest(model) != self.manifest["onnx_sha256"]:
            raise ValueError("RF-DETR ONNX hash mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.threshold = threshold

    def __call__(self, frame):
        import cv2

        # Upstream 1.10.0 uses float tensor bilinear with antialias=False.
        # Resize float RGB to avoid uint8-rounding and PIL-antialias divergence.
        image = cv2.resize(
            frame[:, :, ::-1].astype(np.float32) / 255.0,
            (384, 384),
            interpolation=cv2.INTER_LINEAR,
        )
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        image = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        outputs = dict(
            zip(
                [item.name for item in self.session.get_outputs()],
                self.session.run(None, {self.session.get_inputs()[0].name: image}),
                strict=True,
            )
        )
        logits, boxes = outputs["labels"][0], outputs["dets"][0]
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        # Same global top-query selection as upstream, then COCO91 person class 1.
        flat = probabilities.ravel()
        top = np.argsort(flat)[-min(len(boxes), flat.size) :][::-1]
        selected = top[(top % logits.shape[1] == 1) & (flat[top] >= self.threshold)]
        scores = flat[selected]
        boxes = boxes[selected // logits.shape[1]].copy()
        corners = np.column_stack(
            (
                boxes[:, :2] - np.maximum(boxes[:, 2:], 0) / 2,
                boxes[:, :2] + np.maximum(boxes[:, 2:], 0) / 2,
            )
        )
        corners *= np.array([frame.shape[1], frame.shape[0]] * 2)
        return corners, scores


class YoloXPeople:
    def __init__(self, threshold: float, threads: int):
        import onnxruntime as ort

        from boxing_vision.pose import RTMLibPoseBackend

        self.model = RTMLibPoseBackend(device="cpu")._ensure_model().det_model
        # RTMLib owns preprocessing; replace only its ORT session to fix threads.
        original = self.model.session
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.model.session = ort.InferenceSession(
            original._model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.model_hash = digest(Path(original._model_path))
        self.threshold = threshold

    def __call__(self, frame):
        from boxing_vision.pose import decode_yolox_people

        image, ratio = self.model.preprocess(frame)
        outputs = self.model.inference(image)
        return decode_yolox_people(
            outputs[0],
            self.model.model_input_size,
            ratio,
            score_threshold=self.threshold,
            nms_threshold=0.60,
        )


def benchmark(
    video: Path,
    rfdetr_dir: Path,
    start_s: float,
    duration_s: float,
    sample_fps: float,
    threshold: float,
    threads: int,
) -> dict:
    import cv2

    if not (
        0 < duration_s <= 60
        and 0 < sample_fps <= 5
        and 0 <= threshold <= 1
        and threads >= 1
    ):
        raise ValueError(
            "Bounded benchmark: duration <=60 s, sample_fps <=5, threshold [0,1], threads >=1"
        )
    report = {
        "schema_version": 1,
        "video_filename": video.name,
        "video_sha256": digest(video),
        "hardware": platform.machine(),
        "system": platform.platform(),
        "start_s": start_s,
        "duration_s": duration_s,
        "sample_fps": sample_fps,
        "provider": "CPUExecutionProvider",
        "threads": threads,
        "threshold": threshold,
        "video_rights": "unknown_not_a_training_dataset",
        "ground_truth": None,
        "accuracy": None,
        "winner": None,
        "notes": [
            "Latency includes preprocessing/inference/postprocessing, excludes seek/decode.",
            "Thresholds are uncalibrated and not directly comparable across models.",
            "Counts do not measure precision, recall, boxing identity, or occlusion robustness.",
            "No warm thermal or full pipeline performance gate is established.",
        ],
        "models": {},
    }
    for name, factory in (
        ("yolox_tiny_humanart", lambda: YoloXPeople(threshold, threads)),
        ("rfdetr_nano_coco", lambda: RFDetrPeople(rfdetr_dir, threshold, threads)),
    ):
        detector = factory()
        capture = cv2.VideoCapture(str(video))
        rows, times = [], []
        try:
            for timestamp in np.arange(start_s, start_s + duration_s, 1 / sample_fps):
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000))
                ok, frame = capture.read()
                if not ok:
                    break
                if not rows:
                    for _ in range(3):
                        detector(frame)
                begin = time.perf_counter()
                boxes, scores = detector(frame)
                elapsed = (time.perf_counter() - begin) * 1000
                times.append(elapsed)
                rows.append(
                    {
                        "timestamp_ms": round(timestamp * 1000),
                        "latency_ms": elapsed,
                        "people_count": len(boxes),
                        "boxes": boxes.tolist(),
                        "scores": scores.tolist(),
                    }
                )
        finally:
            capture.release()
        if not rows:
            raise ValueError("No video frames decoded")
        report["models"][name] = {
            "frames": rows,
            "sample_count": len(rows),
            "latency_median_ms": float(np.median(times)),
            "latency_p95_ms": float(np.percentile(times, 95)),
            "mean_people_count": float(np.mean([row["people_count"] for row in rows])),
        }
        del detector
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "benchmark"))
    parser.add_argument("--rfdetr-dir", type=Path, default=Path("models/rfdetr-nano"))
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start-s", type=float, default=0)
    parser.add_argument("--duration-s", type=float, default=12)
    parser.add_argument("--sample-fps", type=float, default=2)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "export":
        result = export_rfdetr(args.rfdetr_dir)
    else:
        if args.video is None:
            parser.error("--video required for benchmark")
        result = benchmark(
            args.video,
            args.rfdetr_dir,
            args.start_s,
            args.duration_s,
            args.sample_fps,
            args.threshold,
            args.threads,
        )
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
