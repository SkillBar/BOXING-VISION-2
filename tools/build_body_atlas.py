"""Extract registered technical layers from one ImageGen segmentation master.

No hand-drawn anatomy: each mask is extracted from the same generated pixels.
The generator's baked white checkerboard is removed by foreground connectivity.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("boxing_vision/static/assets"))
    args = parser.parse_args()
    source = np.asarray(Image.open(args.source).convert("RGB"))
    # Background is neutral white/checkerboard; all saturated target pixels and
    # graphite foreground are disjoint from it. Keep only the central component.
    maximum, minimum = source.max(axis=2), source.min(axis=2)
    binary = ((maximum < 190) | ((maximum.astype(int) - minimum) > 65)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count < 2:
        raise ValueError("No foreground silhouette found")
    foreground = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    alpha = foreground.astype(np.uint8) * 255
    r, g, b = (source[..., i].astype(float) for i in range(3))
    head = foreground & (r > g * 1.7) & (b > g * 1.7) & (r > 100)
    body = foreground & (g > r * 1.5) & (g > b * 1.5) & (g > 90)
    gray = .299 * r + .587 * g + .114 * b
    gray[head] = 66 + maximum[head] / 255 * 32
    gray[body] = 57 + maximum[body] / 255 * 38
    neutral = np.stack([gray * .94, gray * .97, gray], axis=2).clip(0, 255).astype(np.uint8)
    neutral[~foreground] = 0
    args.output.mkdir(parents=True, exist_ok=True)
    white = np.full_like(source, 255)
    paths = {"base": "body-map-v3-base.png", "head_mask": "body-map-v3-head.png", "body_mask": "body-map-v3-body.png"}
    for key, rgb, a in (("base", neutral, alpha), ("head_mask", white, head.astype(np.uint8)*255),
                        ("body_mask", white, body.astype(np.uint8)*255)):
        Image.fromarray(np.dstack((rgb, a))).save(args.output / paths[key], optimize=True)
    manifest = {"version": 3, **paths, "canvas_size": [source.shape[1], source.shape[0]],
                "coordinate_space": "defender_front_canonical_v1",
                "viewport_crop": {"x": .18, "y": 0, "width": .64, "height": .72},
                "source": args.source.name, "generator": "built-in ImageGen",
                "registration_error_px": 0, "alpha_min": int(alpha.min()), "alpha_max": int(alpha.max()),
                "head_pixels": int(head.sum()), "body_pixels": int(body.sum()),
                "zones": {"head": {"default_point": {"x": .5, "y": .10}},
                          "body": {"default_point": {"x": .5, "y": .30}}}}
    (args.output / "body-map-v3-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
