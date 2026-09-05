from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .contracts import PunchEvent
from .media_process import hidden_process_kwargs

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class JobArtifacts:
    """Filesystem contract for one analysis run."""

    job_id: str
    run_dir: Path
    silent_video: Path
    annotated_video: Path
    events_path: Path
    summary_path: Path
    log_path: Path
    clips_dir: Path
    work_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def _new_job_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _validate_safe_id(value: str, *, label: str = "job_id") -> str:
    if not _SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            f"{label} must contain only letters, digits, '.', '_' or '-' and "
            "must not contain a path"
        )
    return value


def create_job_artifacts(
    runs_dir: str | os.PathLike[str],
    job_id: str | None = None,
    *,
    exist_ok: bool = False,
) -> JobArtifacts:
    """Create all directories and stable paths used by one analysis job.

    The job directory is deliberately created in one operation.  By default an
    existing explicit job id is rejected so a new run cannot silently overwrite
    investor-demo evidence.
    """

    safe_job_id = _validate_safe_id(_new_job_id() if job_id is None else job_id)
    root = Path(runs_dir).expanduser().resolve()
    run_dir = root / safe_job_id
    run_dir.mkdir(parents=True, exist_ok=exist_ok)

    work_dir = run_dir / ".work"
    clips_dir = run_dir / "clips"
    work_dir.mkdir(exist_ok=exist_ok)
    clips_dir.mkdir(exist_ok=exist_ok)

    return JobArtifacts(
        job_id=safe_job_id,
        run_dir=run_dir,
        silent_video=work_dir / "annotated_silent.mp4",
        annotated_video=run_dir / "annotated.mp4",
        events_path=run_dir / "events.json",
        summary_path=run_dir / "summary.json",
        log_path=run_dir / "analysis.log",
        clips_dir=clips_dir,
        work_dir=work_dir,
    )


# A shorter name is convenient in code that treats run/job as synonyms.
create_run_artifacts = create_job_artifacts


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: object,
    *,
    indent: int | None = 2,
) -> Path:
    """Durably replace a JSON file without exposing a partially written file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=indent,
                default=_json_default,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary, destination)
        temporary = None

        # Best effort: syncing the containing directory makes the rename durable
        # across a power loss on filesystems that support directory fsync.
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


write_json_atomic = atomic_write_json


def _resolve_executable(executable: str | os.PathLike[str]) -> str:
    value = os.fspath(executable)
    if os.path.sep in value:
        resolved = Path(value).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"FFmpeg executable was not found: {resolved}")
        return str(resolved)

    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(
            f"FFmpeg executable '{value}' was not found. Install ffmpeg and retry."
        )
    return resolved


def _run_ffmpeg(
    arguments: Sequence[str | os.PathLike[str]],
    *,
    ffmpeg_bin: str | os.PathLike[str] = "ffmpeg",
    cancel_callback: Callable[[], bool] | None = None,
) -> subprocess.CompletedProcess[str]:
    executable = _resolve_executable(ffmpeg_bin)
    command = [executable, *[os.fspath(argument) for argument in arguments]]
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **hidden_process_kwargs(),
        )
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if cancel_callback is not None and cancel_callback():
                    raise RuntimeError("FFmpeg operation cancelled")
        if process.returncode != 0:
            detail = (stderr or stdout or "unknown FFmpeg error").strip()
            raise RuntimeError(f"FFmpeg failed: {detail}")
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=3)
        raise


def _temporary_media_path(destination: Path) -> Path:
    suffix = destination.suffix or ".mp4"
    return destination.parent / f".{destination.stem}.{uuid.uuid4().hex}.tmp{suffix}"


def persist_fighter_portrait(
    source_path: str | os.PathLike[str] | None,
    run_dir: str | os.PathLike[str],
    fighter_id: str,
    *,
    size: int = 256,
) -> str | None:
    """Store a privacy-safe square portrait and return its run-relative path.

    The original upload name and absolute path never enter the artifact contract.
    A fixed, validated fighter id determines the destination filename, making a
    repeated call an atomic replacement rather than a source-name disclosure.
    """

    if source_path is None or not os.fspath(source_path).strip():
        return None
    if size < 64 or size > 2048:
        raise ValueError("portrait size must be between 64 and 2048 pixels")

    safe_fighter_id = _validate_safe_id(str(fighter_id), label="fighter_id")
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Portrait image was not found: {source}")

    destination_dir = Path(run_dir).expanduser().resolve() / "profiles"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{safe_fighter_id}.webp"
    temporary = destination_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp.webp"
    try:
        try:
            with Image.open(source) as uploaded:
                normalized = ImageOps.exif_transpose(uploaded).convert("RGB")
                portrait = ImageOps.fit(
                    normalized,
                    (size, size),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.4),
                )
                portrait.save(
                    temporary,
                    format="WEBP",
                    quality=88,
                    method=6,
                )
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Файл портрета не является поддерживаемым изображением") from exc
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return destination.relative_to(Path(run_dir).expanduser().resolve()).as_posix()


def finalize_h264_video(
    silent_video: str | os.PathLike[str],
    source_video: str | os.PathLike[str] | None,
    output_path: str | os.PathLike[str],
    *,
    ffmpeg_bin: str | os.PathLike[str] = "ffmpeg",
    crf: int = 21,
    preset: str = "fast",
    cancel_callback: Callable[[], bool] | None = None,
) -> Path:
    """Encode annotated frames as browser-safe H.264 and retain source audio.

    The source audio mapping is optional (``?`` in FFmpeg's map expression), so
    an input without an audio stream follows the exact same successful path.
    The output is atomically installed only after FFmpeg has completed.
    """

    silent = Path(silent_video)
    source = Path(source_video) if source_video is not None else None
    destination = Path(output_path)
    if not silent.is_file():
        raise FileNotFoundError(f"Annotated silent video was not found: {silent}")
    if source is not None and not source.is_file():
        raise FileNotFoundError(f"Source video was not found: {source}")
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_media_path(destination)
    arguments: list[str | os.PathLike[str]] = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    arguments.extend(["-i", silent])
    if source is not None:
        arguments.extend(["-i", source])

    arguments.extend(["-map", "0:v:0"])
    if source is not None:
        arguments.extend(["-map", "1:a:0?"])
    arguments.extend(["-map_metadata", "-1", "-map_chapters", "-1"])
    arguments.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )
    if source is not None:
        from .video import probe_video

        # Infinite apad + -shortest can overshoot by hundreds of milliseconds
        # depending on mux scheduling. Bound padding by the rendered VIDEO,
        # not by source audio/container duration (which can be shorter/longer).
        duration = probe_video(silent).duration_s
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Не удалось определить длительность видео для синхронизации звука")
        duration_text = f"{duration:.9f}"
        arguments.extend(["-c:a", "aac", "-b:a", "160k", "-af",
                          f"apad=whole_dur={duration_text},atrim=end={duration_text}", "-t", duration_text])
    arguments.append(temporary)

    try:
        _run_ffmpeg(
            arguments,
            ffmpeg_bin=ffmpeg_bin,
            cancel_callback=cancel_callback,
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _event_value(event: PunchEvent | Mapping[str, Any], name: str) -> Any:
    if isinstance(event, Mapping):
        return event[name]
    return getattr(event, name)


def extract_event_clips(
    source_video: str | os.PathLike[str],
    events: Iterable[PunchEvent | Mapping[str, Any]],
    clips_dir: str | os.PathLike[str],
    *,
    padding_s: float = 2.0,
    ffmpeg_bin: str | os.PathLike[str] = "ffmpeg",
    update_events: bool = True,
    cancel_callback: Callable[[], bool] | None = None,
) -> dict[str, Path]:
    """Extract H.264 clips around events and return ``event_id -> path``.

    ``start_ms - padding`` is clamped to zero.  FFmpeg naturally clamps the end
    at EOF. Dataclass events are updated with their resulting path by default;
    mapping inputs are never mutated.
    """

    source = Path(source_video)
    if not source.is_file():
        raise FileNotFoundError(f"Source video was not found: {source}")
    if padding_s < 0:
        raise ValueError("padding_s cannot be negative")

    destination_dir = Path(clips_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: dict[str, Path] = {}

    for event in events:
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("Clip extraction cancelled")
        event_id = _validate_safe_id(
            str(_event_value(event, "event_id")), label="event_id"
        )
        start_ms = int(_event_value(event, "start_ms"))
        end_ms = int(_event_value(event, "end_ms"))
        if start_ms < 0 or end_ms < start_ms:
            raise ValueError(f"Invalid time range for event {event_id}")

        clip_start_s = max(0.0, start_ms / 1000.0 - padding_s)
        clip_end_s = end_ms / 1000.0 + padding_s
        duration_s = max(0.05, clip_end_s - clip_start_s)
        destination = destination_dir / f"{event_id}.mp4"
        temporary = _temporary_media_path(destination)
        arguments: list[str | os.PathLike[str]] = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{clip_start_s:.3f}",
            "-i",
            source,
            "-t",
            f"{duration_s:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            temporary,
        ]

        try:
            _run_ffmpeg(
                arguments,
                ffmpeg_bin=ffmpeg_bin,
                cancel_callback=cancel_callback,
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

        clip_paths[event_id] = destination
        if update_events and isinstance(event, PunchEvent):
            event.clip_path = (Path("clips") / destination.name).as_posix()

    return clip_paths


__all__ = [
    "JobArtifacts",
    "atomic_write_json",
    "create_job_artifacts",
    "create_run_artifacts",
    "extract_event_clips",
    "finalize_h264_video",
    "persist_fighter_portrait",
    "write_json_atomic",
]
