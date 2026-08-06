"""Video I/O utilities used by the offline analysis pipeline.

The module deliberately keeps FFmpeg behind a small, testable boundary.  It
does not open an input with OpenCV until :func:`validate_video` has established
that the file really contains a supported video stream.
"""

from __future__ import annotations

import json
import math
import selectors
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v"}


class VideoError(RuntimeError):
    """Base error for input inspection and transcoding failures."""


class VideoValidationError(VideoError):
    """Raised when a user supplied file cannot be analysed safely."""


class VideoNormalizationError(VideoError):
    """Raised when FFmpeg cannot create the normalized working copy."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: Path
    duration_s: float
    width: int
    height: int
    fps: float
    frame_count: int | None
    video_codec: str
    has_audio: bool
    audio_codec: str | None
    rotation: int
    size_bytes: int

    @property
    def display_width(self) -> int:
        return self.height if abs(self.rotation) % 180 == 90 else self.width

    @property
    def display_height(self) -> int:
        return self.width if abs(self.rotation) % 180 == 90 else self.height

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True, slots=True)
class VideoFrame:
    frame_index: int
    timestamp_ms: int
    image: np.ndarray


def _parse_fraction(value: object) -> float:
    if value in (None, "", "N/A", "0/0"):
        return 0.0
    text = str(value)
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _parse_rotation(stream: dict[str, object]) -> int:
    tags = stream.get("tags") or {}
    if isinstance(tags, dict) and tags.get("rotate") is not None:
        try:
            return int(float(str(tags["rotate"]))) % 360
        except (TypeError, ValueError):
            pass
    side_data = stream.get("side_data_list") or []
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict) and item.get("rotation") is not None:
                try:
                    return int(float(str(item["rotation"]))) % 360
                except (TypeError, ValueError):
                    continue
    return 0


def probe_video(path: str | Path, *, ffprobe_path: str = "ffprobe") -> VideoMetadata:
    """Return trustworthy stream metadata using ``ffprobe`` JSON output."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise VideoValidationError(f"Видео не найдено: {source}")
    executable = shutil.which(ffprobe_path)
    if executable is None:
        raise VideoValidationError("FFprobe не найден. Установите FFmpeg и повторите запуск.")

    command = [
        executable,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VideoValidationError(f"Не удалось прочитать видео: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "неизвестная ошибка"
        raise VideoValidationError(f"FFprobe не смог прочитать файл: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise VideoValidationError("FFprobe вернул повреждённые метаданные") from exc

    streams = payload.get("streams") or []
    if not isinstance(streams, list):
        streams = []
    video_stream = next(
        (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise VideoValidationError("В файле нет видеопотока")
    audio_stream = next(
        (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"),
        None,
    )
    format_data = payload.get("format") or {}
    if not isinstance(format_data, dict):
        format_data = {}

    duration = _parse_fraction(video_stream.get("duration")) or _parse_fraction(format_data.get("duration"))
    fps = _parse_fraction(video_stream.get("avg_frame_rate")) or _parse_fraction(video_stream.get("r_frame_rate"))
    try:
        frame_count = int(str(video_stream.get("nb_frames")))
    except (TypeError, ValueError):
        frame_count = round(duration * fps) if duration > 0 and fps > 0 else None

    return VideoMetadata(
        path=source,
        duration_s=max(0.0, duration),
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        fps=max(0.0, fps),
        frame_count=frame_count,
        video_codec=str(video_stream.get("codec_name") or "unknown"),
        has_audio=audio_stream is not None,
        audio_codec=str(audio_stream.get("codec_name") or "unknown") if audio_stream else None,
        rotation=_parse_rotation(video_stream),
        size_bytes=source.stat().st_size,
    )


def validate_video(
    path: str | Path,
    *,
    max_duration_s: float = 3600,
    max_size_bytes: int = 20 * 1024**3,
    ffprobe_path: str = "ffprobe",
) -> VideoMetadata:
    """Validate a local upload and return its metadata.

    Validation is based on the actual streams, not only on the extension.  The
    extension check exists to give the user a useful error before probing.
    """

    source = Path(path).expanduser()
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise VideoValidationError(f"Поддерживаются только: {formats}")
    metadata = probe_video(source, ffprobe_path=ffprobe_path)
    if metadata.size_bytes <= 0:
        raise VideoValidationError("Видео пустое")
    if metadata.size_bytes > max_size_bytes:
        raise VideoValidationError("Видео слишком большое для локальной обработки")
    if metadata.duration_s <= 0:
        raise VideoValidationError("Не удалось определить длительность видео")
    if metadata.duration_s > max_duration_s + 0.05:
        raise VideoValidationError(
            f"Длительность {metadata.duration_s / 60:.1f} мин превышает лимит {max_duration_s / 60:.0f} мин"
        )
    if metadata.width < 64 or metadata.height < 64:
        raise VideoValidationError("Разрешение видео слишком маленькое")
    if metadata.fps <= 0:
        raise VideoValidationError("Не удалось определить частоту кадров")
    return metadata


def normalize_video(
    source: str | Path,
    destination: str | Path,
    *,
    output_height: int = 720,
    output_fps: int = 30,
    start_s: float = 0.0,
    end_s: float | None = None,
    max_duration_s: float = 3600,
    preserve_audio: bool = True,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
    progress_callback: Callable[[float], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> Path:
    """Create a CFR H.264/AAC working copy suitable for browsers and analysis."""

    metadata = validate_video(source, max_duration_s=max_duration_s, ffprobe_path=ffprobe_path)
    if output_height < 240 or output_height > 2160:
        raise ValueError("Высота результата должна быть от 240 до 2160 пикселей")
    if output_fps < 1 or output_fps > 120:
        raise ValueError("FPS результата должен быть от 1 до 120")
    if start_s < 0:
        raise ValueError("Начало фрагмента не может быть отрицательным")
    stop_s = metadata.duration_s if end_s is None else min(float(end_s), metadata.duration_s)
    if stop_s <= start_s:
        raise ValueError("Конец фрагмента должен быть позже начала")

    executable = shutil.which(ffmpeg_path)
    if executable is None:
        raise VideoNormalizationError("FFmpeg не найден. Установите FFmpeg и повторите запуск.")

    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.part{output.suffix or '.mp4'}")
    temporary.unlink(missing_ok=True)
    duration = stop_s - start_s
    # No upscaling: this matters for long, low-resolution archive footage.
    target_height = min(output_height, metadata.display_height)
    target_height -= target_height % 2
    video_filter = f"scale=-2:{target_height}:flags=lanczos,fps={output_fps},setsar=1"
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        "-nostats",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(metadata.path),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
    ]
    if preserve_audio and metadata.has_audio:
        command.extend(["-map", "0:a:0?"])
    command.extend(
        [
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if preserve_audio and metadata.has_audio:
        command.extend(["-c:a", "aac", "-b:a", "160k"])
    else:
        command.append("-an")
    command.extend(["-movflags", "+faststart", "-y", str(temporary)])

    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        ffmpeg_output: list[str] = []
        if process.stdout is not None:
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                if cancel_callback is not None and cancel_callback():
                    raise VideoNormalizationError("Нормализация отменена пользователем")
                ready = selector.select(timeout=0.25)
                if not ready:
                    if process.poll() is not None:
                        break
                    continue
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                ffmpeg_output.append(line.rstrip())
                if len(ffmpeg_output) > 120:
                    del ffmpeg_output[:20]
                key, _, raw_value = line.strip().partition("=")
                if key in {"out_time_us", "out_time_ms"} and progress_callback:
                    try:
                        # FFmpeg's out_time_us is microseconds.  Older builds
                        # also label the same unit as out_time_ms.
                        seconds = float(raw_value) / 1_000_000.0
                        progress_callback(min(1.0, max(0.0, seconds / duration)))
                    except (TypeError, ValueError):
                        pass
            selector.close()
        stderr = "\n".join(ffmpeg_output)
        return_code = process.wait()
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise VideoNormalizationError(f"Не удалось запустить FFmpeg: {exc}") from exc
    except BaseException:
        # A progress callback may carry a user cancellation signal. Do not
        # leave FFmpeg encoding in the background after the request returns.
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        temporary.unlink(missing_ok=True)
        raise
    if return_code != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else "неизвестная ошибка"
        raise VideoNormalizationError(f"FFmpeg не смог нормализовать видео: {detail}")
    temporary.replace(output)
    if progress_callback:
        progress_callback(1.0)
    return output


def iter_video_frames(
    path: str | Path,
    *,
    target_fps: float | None = None,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> Iterator[VideoFrame]:
    """Yield decoded frames without keeping the full video in memory."""

    source = str(Path(path).expanduser().resolve())
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise VideoValidationError(f"OpenCV не смог открыть видео: {source}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(native_fps) or native_fps <= 0:
        capture.release()
        raise VideoValidationError("OpenCV не смог определить FPS видео")
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_s) * 1000.0)
    sample_interval_ms = 1000.0 / target_fps if target_fps and target_fps > 0 else 0.0
    next_sample_ms = max(0.0, start_s) * 1000.0
    decoded_index = max(0, round(start_s * native_fps))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = round(capture.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0:
                timestamp_ms = round(decoded_index * 1000.0 / native_fps)
            decoded_index += 1
            if end_s is not None and timestamp_ms > end_s * 1000.0:
                break
            if sample_interval_ms and timestamp_ms + 0.5 < next_sample_ms:
                continue
            if sample_interval_ms:
                while next_sample_ms <= timestamp_ms + 0.5:
                    next_sample_ms += sample_interval_ms
            yield VideoFrame(decoded_index - 1, timestamp_ms, frame)
    finally:
        capture.release()


def scene_cut_score(previous: np.ndarray | None, current: np.ndarray) -> float:
    """Return a 0..1 hard-cut score using luminance change and histograms."""

    if previous is None:
        return 0.0
    if previous.ndim != 3 or current.ndim != 3:
        raise ValueError("Кадр должен иметь форму H×W×3")
    size = (160, 90)
    previous_small = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
    current_small = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
    previous_gray = cv2.cvtColor(previous_small, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
    mean_difference = float(np.mean(cv2.absdiff(previous_gray, current_gray))) / 255.0
    previous_hist = cv2.calcHist([previous_small], [0, 1], None, [16, 16], [0, 256, 0, 256])
    current_hist = cv2.calcHist([current_small], [0, 1], None, [16, 16], [0, 256, 0, 256])
    cv2.normalize(previous_hist, previous_hist)
    cv2.normalize(current_hist, current_hist)
    correlation = float(cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_CORREL))
    histogram_change = min(1.0, max(0.0, (1.0 - correlation) / 1.5))
    return min(1.0, max(0.0, 0.58 * mean_difference + 0.42 * histogram_change))


def detect_scene_cut(
    previous: np.ndarray | None,
    current: np.ndarray,
    *,
    threshold: float = 0.42,
) -> bool:
    return scene_cut_score(previous, current) >= threshold


class SceneCutDetector:
    """Stateful hard-cut detector for a stream of decoded frames."""

    def __init__(self, threshold: float = 0.42, cooldown_frames: int = 3) -> None:
        self.threshold = threshold
        self.cooldown_frames = max(0, cooldown_frames)
        self._previous: np.ndarray | None = None
        self._cooldown = 0

    def reset(self) -> None:
        self._previous = None
        self._cooldown = 0

    def update(self, frame: np.ndarray) -> tuple[bool, float]:
        score = scene_cut_score(self._previous, frame)
        cut = score >= self.threshold and self._cooldown == 0
        self._previous = frame.copy()
        if cut:
            self._cooldown = self.cooldown_frames
        elif self._cooldown:
            self._cooldown -= 1
        return cut, score


class ReplayDetector:
    """Conservative detector for a repeated shot appearing after a scene cut.

    It compares perceptual hashes only at cut boundaries.  A match is a hint,
    never proof, so the pipeline should expose it as ``possible replay``.
    """

    def __init__(self, *, min_gap_ms: int = 4_000, max_hamming_distance: int = 5, history_size: int = 96) -> None:
        self.min_gap_ms = min_gap_ms
        self.max_hamming_distance = max_hamming_distance
        self.history_size = history_size
        self._history: list[tuple[int, int]] = []

    @staticmethod
    def _dhash(frame: np.ndarray) -> int:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        differences = resized[:, 1:] > resized[:, :-1]
        result = 0
        for bit in differences.flat:
            result = (result << 1) | int(bit)
        return result

    def update(self, frame: np.ndarray, timestamp_ms: int, *, is_scene_cut: bool) -> bool:
        if not is_scene_cut:
            return False
        fingerprint = self._dhash(frame)
        possible_replay = any(
            timestamp_ms - prior_timestamp >= self.min_gap_ms
            and (fingerprint ^ prior_fingerprint).bit_count() <= self.max_hamming_distance
            for prior_timestamp, prior_fingerprint in self._history
        )
        self._history.append((timestamp_ms, fingerprint))
        self._history = self._history[-self.history_size :]
        return possible_replay
