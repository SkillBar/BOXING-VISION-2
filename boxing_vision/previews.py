from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import BinaryIO

from PIL import Image

from .artifacts import atomic_write_json
from .media_process import hidden_process_kwargs


def _read_frame(stream: BinaryIO, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def generate_hover_previews(
    video_path: str | os.PathLike[str],
    run_dir: str | os.PathLike[str],
    *,
    interval_s: float = 2.0,
    tile_width: int = 160,
    tile_height: int = 90,
    columns: int = 10,
    rows: int = 10,
    padding: int = 2,
    margin: int = 2,
) -> Path:
    """Create compact WebP sheets and a manifest for timeline hover previews.

    FFmpeg performs the deterministic sampling, scaling and letterboxing. Pillow
    only packs the raw frames into bounded-size WebP sheets because the local
    FFmpeg build does not necessarily ship a WebP encoder.
    """

    if interval_s <= 0:
        raise ValueError("preview interval must be positive")
    if min(tile_width, tile_height, columns, rows) <= 0:
        raise ValueError("preview geometry must be positive")
    if min(padding, margin) < 0:
        raise ValueError("preview padding and margin cannot be negative")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to generate timeline previews")
    source = Path(video_path).expanduser().resolve()
    destination_root = Path(run_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Preview source was not found: {source}")

    sheets_dir = destination_root / "previews"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = destination_root / "preview_manifest.json"
    filter_graph = (
        f"fps=1/{interval_s:.6f},"
        f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
        f"pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2:color=0x050608"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        filter_graph,
        "-an",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **hidden_process_kwargs(),
    )
    assert process.stdout is not None and process.stderr is not None

    frame_bytes = tile_width * tile_height * 3
    frames_per_sheet = columns * rows
    sheet_width = margin * 2 + columns * tile_width + max(0, columns - 1) * padding
    sheet_height = margin * 2 + rows * tile_height + max(0, rows - 1) * padding
    frames: list[dict[str, int]] = []
    sheet_names: list[str] = []
    sheet = Image.new("RGB", (sheet_width, sheet_height), "#050608")
    sheet_index = 0
    frame_index = 0

    def flush_sheet() -> None:
        nonlocal sheet, sheet_index
        if frame_index <= sheet_index * frames_per_sheet:
            return
        sheet_index += 1
        name = f"sprite-{sheet_index:03d}.webp"
        destination = sheets_dir / name
        temporary = sheets_dir / f".{name}.{uuid.uuid4().hex}.tmp.webp"
        try:
            sheet.save(temporary, format="WEBP", quality=72, method=5)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        sheet_names.append(f"previews/{name}")
        sheet = Image.new("RGB", (sheet_width, sheet_height), "#050608")

    try:
        while True:
            raw = _read_frame(process.stdout, frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError("FFmpeg returned a partial preview frame")
            local_index = frame_index % frames_per_sheet
            column = local_index % columns
            row = local_index // columns
            x = margin + column * (tile_width + padding)
            y = margin + row * (tile_height + padding)
            image = Image.frombytes("RGB", (tile_width, tile_height), raw)
            sheet.paste(image, (x, y))
            frames.append(
                {
                    "time_ms": round(frame_index * interval_s * 1000),
                    "sheet": sheet_index,
                    "x": x,
                    "y": y,
                }
            )
            frame_index += 1
            if frame_index % frames_per_sheet == 0:
                flush_sheet()

        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg preview generation failed: {stderr or return_code}")
        flush_sheet()
    except BaseException:
        process.kill()
        process.wait()
        raise

    manifest = {
        "version": 1,
        "interval_ms": round(interval_s * 1000),
        "tile_width": tile_width,
        "tile_height": tile_height,
        "sheet_width": sheet_width,
        "sheet_height": sheet_height,
        "columns": columns,
        "rows": rows,
        "padding": padding,
        "margin": margin,
        "sheets": sheet_names,
        "frames": frames,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest_path


__all__ = ["generate_hover_previews"]
