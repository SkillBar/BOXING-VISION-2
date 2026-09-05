"""Build aligned alpha masks for the Boxing Vision target silhouette.

The source illustration remains the visual base.  These masks deliberately
expose only the two target classes supported by the model: head and torso.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "boxing_vision" / "static" / "assets"
SOURCE = ASSETS / "body-map-source.png"
OUTPUT_SIZE = (512, 768)


def _rgba(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    packed = np.dstack((np.clip(rgb, 0, 255).astype(np.uint8), alpha.astype(np.uint8)))
    return Image.fromarray(packed, "RGBA").resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)


def main() -> None:
    source = np.asarray(Image.open(SOURCE).convert("RGBA"), dtype=np.uint8)
    red = source[..., 0].astype(np.int16)
    green = source[..., 1].astype(np.int16)
    blue = source[..., 2].astype(np.int16)
    source_alpha = source[..., 3].astype(np.int16)

    # Remove the low-alpha halo produced around the render while retaining a
    # narrow anti-aliased edge on the actual mannequin.
    alpha = np.clip((source_alpha - 185) * 4, 0, 255).astype(np.uint8)
    solid = alpha > 72

    # The source is intentionally a segmentation template: magenta is HEAD,
    # green is BODY. Channel-dominance thresholds reject colored halo pixels.
    head_zone = solid & (red > 155) & (blue > 115) & (red > green * 2) & (blue > green * 1.6)
    body_zone = solid & (green > 145) & (green > red * 1.7) & (green > blue * 1.7)

    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    neutral = np.clip(luminance * 0.82 + 20, 62, 166)
    # Normalize the two target plates so they read as clean interface zones,
    # not painted anatomy, while preserving a small amount of 3D form.
    for zone in (head_zone, body_zone):
        if np.any(zone):
            median = float(np.median(luminance[zone]))
            neutral[zone] = np.clip(124 + (luminance[zone] - median) * 0.22, 108, 142)
    base_rgb = np.stack((neutral * 0.94, neutral * 0.97, neutral), axis=-1)

    white = np.full((*alpha.shape, 3), 255, dtype=np.uint8)
    head_alpha = np.where(head_zone, alpha, 0).astype(np.uint8)
    body_alpha = np.where(body_zone, alpha, 0).astype(np.uint8)

    _rgba(base_rgb, alpha).save(ASSETS / "body-map-boxer.png", optimize=True)
    _rgba(white, head_alpha).save(ASSETS / "body-map-head-mask.png", optimize=True)
    _rgba(white, body_alpha).save(ASSETS / "body-map-body-mask.png", optimize=True)


if __name__ == "__main__":
    main()
