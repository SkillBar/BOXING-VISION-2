"""Render review contact sheets from actual cached tracks, without running ML.

The pictures are sparse diagnostic samples, not identity-precision measurements.
Each sample warms the real stateful renderer for one second at output FPS.
"""
from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_right
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxing_vision.contracts import RenderFrameContext
from boxing_vision.display_tracking import DisplayTrackSampler
from boxing_vision.pipeline import _read_observation_cache, _sample_pose_tracks
from boxing_vision.render import FrameRenderer
from boxing_vision.tracking_artifacts import read_display_cache


def inspect_run(run_dir: Path, output: Path, times: list[float]) -> dict:
    run_dir = run_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = run_dir / ".render_cache"
    config = json.loads((cache / "config.json").read_text())
    first_pass = json.loads((cache / "first_pass.json").read_text())
    frames = sorted(first_pass["frame_states"], key=lambda item: item["timestamp_ms"])
    frame_times = [int(frame["timestamp_ms"]) for frame in frames]
    display = DisplayTrackSampler(read_display_cache(cache / "display_tracks.jsonl.gz"))
    pose_tracks = {}
    for observation in _read_observation_cache(cache / "observations.jsonl.gz"):
        pose_tracks.setdefault(observation.fighter_id, []).append(observation)
    for track in pose_tracks.values():
        track.sort(key=lambda item: item.timestamp_ms)
    capture = cv2.VideoCapture(str(cache / "normalized.mp4"))
    if not capture.isOpened():
        raise ValueError("Cannot open normalized video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    duration_s = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    samples = []
    images = []
    try:
        for target_s in times:
            if not 0 <= target_s < duration_s:
                raise ValueError(f"Sample {target_s} outside video ({duration_s:.3f}s)")
            target_index = round(target_s * fps)
            start_index = max(0, target_index - round(fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
            renderer = FrameRenderer(
                {"fighter_a": config.get("fighter_a_name", "Боксёр A"),
                 "fighter_b": config.get("fighter_b_name", "Боксёр B")},
                hud_mode="compact", tracking_overlay_style="full",
                pose_threshold=config.get("pose_score_threshold", 0.25),
            )
            indices = {key: 0 for key in pose_tracks}
            for frame_index in range(start_index, target_index + 1):
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode frame {frame_index}")
                timestamp_ms = round(frame_index / fps * 1000)
                state = frames[max(0, bisect_right(frame_times, timestamp_ms) - 1)]
                context = RenderFrameContext(
                    timestamp_ms=timestamp_ms, shot_id=int(state["shot_id"]),
                    scene_state=state["scene_state"],
                    is_scene_cut=bool(state.get("is_scene_cut"))
                    and abs(timestamp_ms - state["timestamp_ms"]) < 1000 / fps,
                )
                poses = _sample_pose_tracks(pose_tracks, indices, timestamp_ms)
                sampled = display.sample(timestamp_ms, context)
                rendered = renderer.draw(
                    image, poses, timestamp_ms=timestamp_ms, frame_context=context,
                    include_hud=False, display_tracks=sampled,
                )
            name = f"frame-{timestamp_ms:06d}.png"
            rgb = Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
            rgb.save(output / name)
            images.append(rgb)
            samples.append({
                "requested_s": target_s, "timestamp_ms": timestamp_ms,
                "frame_path": str((output / name).resolve()),
                "shot_id": context.shot_id, "scene_state": str(context.scene_state),
                "display_tracks": [item.to_dict() for item in sampled],
                "analytical_fighters": [item.fighter_id for item in poses],
            })
    finally:
        capture.release()
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    sheets = []
    tile_width = 768
    image_height = round(images[0].height * tile_width / images[0].width)
    tile_height = image_height + 76
    for start in range(0, len(images), 4):
        sheet = Image.new("RGB", (tile_width * 2, tile_height * 2), "#0D1014")
        draw = ImageDraw.Draw(sheet)
        for slot, (picture, sample) in enumerate(zip(images[start:start + 4], samples[start:start + 4])):
            x, y = (slot % 2) * tile_width, (slot // 2) * tile_height
            sheet.paste(picture.resize((tile_width, image_height)), (x, y + 76))
            tracks = sample["display_tracks"]
            line = " | ".join(
                f"src {item['source_track_id']} {item['identity_state']} {item['display_state']}"
                for item in tracks
            ) or "NO DISPLAY TRACKS"
            draw.text((x + 12, y + 9), f"{sample['timestamp_ms'] / 1000:.3f}s | shot {sample['shot_id']} | {sample['scene_state']}", font=font, fill="#F5F7FA")
            draw.text((x + 12, y + 35), line, font=font, fill="#C2C8D0")
            draw.text((x + 12, y + 55), "actual cache · tracking-only renderer · sparse QA sample", font=font, fill="#939EAA")
        path = output / f"contact-sheet-{start // 4 + 1}.png"
        sheet.save(path)
        sheets.append(str(path.resolve()))
    report = {"run_dir": str(run_dir), "evaluation_kind": "sparse_visual_review_not_precision",
              "fps": fps, "duration_s": duration_s, "samples": samples, "contact_sheets": sheets}
    (output / "samples.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("qa/recovery"))
    parser.add_argument("--times", nargs="+", type=float,
                        default=[10, 12.4, 15.1, 20, 35, 50, 70, 100, 140, 170, 185])
    args = parser.parse_args()
    report = inspect_run(args.run_dir, args.output, args.times)
    print(json.dumps({"samples": len(report["samples"]), "contact_sheets": report["contact_sheets"]}, indent=2))


if __name__ == "__main__":
    main()
