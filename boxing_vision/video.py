"""Video I/O utilities used by the offline analysis pipeline.

The module deliberately keeps FFmpeg behind a small, testable boundary.  It
does not open an input with OpenCV until :func:`validate_video` has established
that the file really contains a supported video stream.
"""

from __future__ import annotations

import json
import math
import queue
import shutil
import subprocess
import threading
from bisect import bisect_left
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .contracts import SceneState
from .media_process import hidden_process_kwargs, stop_process

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v"}


@dataclass(frozen=True, slots=True)
class SceneSegment:
    shot_id: int
    start_ms: int
    end_ms: int | None
    state: SceneState = SceneState.UNCERTAIN
    confidence: float = 0.0
    review_reason: str | None = None


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
            **hidden_process_kwargs(),
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
    reader: threading.Thread | None = None
    reader_stop = threading.Event()
    reader_done = threading.Event()
    output_lines: queue.Queue[str] = queue.Queue(maxsize=128)
    reader_errors: list[Exception] = []
    ffmpeg_output: deque[str] = deque(maxlen=120)
    process_completed = False

    def read_output() -> None:
        assert process is not None and process.stdout is not None
        try:
            while not reader_stop.is_set():
                # Blocking pipe reads belong to this thread: Windows select()
                # accepts sockets, not FFmpeg's anonymous stdout pipe. Bound
                # both the queue and pathological individual diagnostic lines.
                line = process.stdout.readline(16_384)
                if not line:
                    break
                while not reader_stop.is_set():
                    try:
                        output_lines.put(line, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:  # noqa: BLE001 - Forward reader failures to the calling thread.
            reader_errors.append(exc)
        finally:
            reader_done.set()

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **hidden_process_kwargs(),
        )
        if process.stdout is None:
            raise VideoNormalizationError("FFmpeg не открыл поток прогресса")
        reader = threading.Thread(
            target=read_output, name="boxing-ffmpeg-progress", daemon=True
        )
        reader.start()
        while True:
            if cancel_callback is not None and cancel_callback():
                raise VideoNormalizationError("Нормализация отменена пользователем")
            try:
                line = output_lines.get(timeout=0.1)
            except queue.Empty:
                if reader_done.is_set():
                    if reader_errors:
                        raise VideoNormalizationError(
                            f"Не удалось прочитать прогресс FFmpeg: {reader_errors[0]}"
                        ) from reader_errors[0]
                    if process.poll() is not None:
                        break
                continue
            ffmpeg_output.append(line.rstrip())
            key, _, raw_value = line.strip().partition("=")
            if key in {"out_time_us", "out_time_ms"} and progress_callback:
                try:
                    # Both historical FFmpeg keys are in microseconds.
                    seconds = float(raw_value) / 1_000_000.0
                except (TypeError, ValueError):
                    continue
                progress_callback(min(1.0, max(0.0, seconds / duration)))
        stderr = "\n".join(ffmpeg_output)
        return_code = process.wait()
        process_completed = True
    except OSError as exc:
        raise VideoNormalizationError(f"Не удалось запустить FFmpeg: {exc}") from exc
    finally:
        # Terminate before joining: exiting the child closes the write end and
        # releases a silent blocked reader. Do not close a TextIO pipe while its
        # reader holds the lock, which can itself deadlock on Windows.
        reader_stop.set()
        try:
            if process is not None:
                stop_process(process)
        finally:
            if reader is not None:
                reader.join(timeout=1)
            if process is not None and process.stdout is not None and (
                reader is None or not reader.is_alive()
            ):
                process.stdout.close()
            if not process_completed:
                # Windows cannot unlink an output still held by FFmpeg.
                temporary.unlink(missing_ok=True)
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
    """Stateful adaptive hard-cut detector for decoded broadcast frames.

    A rolling baseline makes the detector less sensitive to ordinary camera
    movement than a single absolute threshold.  The public ``update`` contract
    is unchanged so existing pipeline callers remain compatible.
    """

    def __init__(
        self,
        threshold: float = 0.42,
        cooldown_frames: int = 3,
        *,
        adaptive_ratio: float = 3.0,
        adaptive_min_score: float = 0.18,
        history_frames: int = 12,
    ) -> None:
        self.threshold = threshold
        self.cooldown_frames = max(0, cooldown_frames)
        self.adaptive_ratio = max(1.0, float(adaptive_ratio))
        self.adaptive_min_score = max(0.0, float(adaptive_min_score))
        self.history_frames = max(3, int(history_frames))
        self._previous: np.ndarray | None = None
        self._cooldown = 0
        self._scores: deque[float] = deque(maxlen=self.history_frames)

    def reset(self) -> None:
        self._previous = None
        self._cooldown = 0
        self._scores.clear()

    def update(self, frame: np.ndarray) -> tuple[bool, float]:
        score = scene_cut_score(self._previous, frame)
        baseline = float(np.median(self._scores)) if self._scores else 0.0
        adaptive_cut = (
            len(self._scores) >= 3
            and score >= self.adaptive_min_score
            and score >= max(1e-6, baseline) * self.adaptive_ratio
        )
        cut = (score >= self.threshold or adaptive_cut) and self._cooldown == 0
        self._previous = frame.copy()
        if cut:
            self._cooldown = self.cooldown_frames
            self._scores.clear()
        elif self._cooldown:
            self._cooldown -= 1
        else:
            self._scores.append(score)
        return cut, score


@dataclass(slots=True)
class _ReplayCandidate:
    source_shot_id: int
    offset_ms: int
    first_match_ms: int
    last_match_ms: int
    matches: int = 1
    comparisons: int = 1
    dynamic_matches: int = 0
    unique_fingerprints: set[int] = field(default_factory=set)
    last_fingerprint: int | None = None


@dataclass(frozen=True, slots=True)
class _ReplaySample:
    sample_id: int
    timestamp_ms: int
    fingerprint: int
    shot_id: int


class ReplayDetector:
    """Confirm replays from a time-consistent sequence inside one shot.

    A matching frame only opens a bounded candidate.  It cannot mark the rest
    of a video as replay: confirmation requires at least two seconds of
    aligned matches and every candidate is discarded at the next cut.
    """

    def __init__(
        self,
        *,
        min_gap_ms: int = 4_000,
        max_hamming_distance: int = 5,
        history_size: int = 36_000,
        min_sequence_ms: int = 2_000,
        min_match_ratio: float = 0.70,
        timestamp_tolerance_ms: int = 180,
        proposal_window_ms: int = 1_200,
        max_candidates: int = 24,
        max_proposals_per_frame: int = 8,
        max_index_hits: int = 512,
        min_unique_fingerprints: int = 4,
        min_dynamic_matches: int = 3,
    ) -> None:
        self.min_gap_ms = max(0, int(min_gap_ms))
        self.max_hamming_distance = max(0, int(max_hamming_distance))
        self.history_size = max(32, int(history_size))
        self.min_sequence_ms = max(250, int(min_sequence_ms))
        self.min_match_ratio = min(1.0, max(0.5, float(min_match_ratio)))
        self.timestamp_tolerance_ms = max(1, int(timestamp_tolerance_ms))
        self.proposal_window_ms = max(0, int(proposal_window_ms))
        self.max_candidates = max(1, int(max_candidates))
        self.max_proposals_per_frame = max(1, int(max_proposals_per_frame))
        self.max_index_hits = max(8, int(max_index_hits))
        self.min_unique_fingerprints = max(2, int(min_unique_fingerprints))
        self.min_dynamic_matches = max(1, int(min_dynamic_matches))
        self._history_ids: deque[int] = deque()
        self._samples: dict[int, _ReplaySample] = {}
        self._shot_history: dict[int, list[tuple[int, int]]] = {}
        self._hash_bands: dict[tuple[int, int], deque[int]] = {}
        self._next_sample_id = 0
        self._shot_id = 0
        self._shot_start_ms: int | None = None
        self._candidates: dict[tuple[int, int], _ReplayCandidate] = {}
        self._confirmed_current_shot = False
        self.confirmed_start_ms: int | None = None
        self.confirmed_source_start_ms: int | None = None

    @property
    def shot_id(self) -> int:
        return self._shot_id

    @property
    def is_replay(self) -> bool:
        return self._confirmed_current_shot

    def reset(self) -> None:
        self._history_ids.clear()
        self._samples.clear()
        self._shot_history.clear()
        self._hash_bands.clear()
        self._next_sample_id = 0
        self._shot_id = 0
        self._shot_start_ms = None
        self._candidates.clear()
        self._confirmed_current_shot = False
        self.confirmed_start_ms = None
        self.confirmed_source_start_ms = None

    @staticmethod
    def _dhash(frame: np.ndarray) -> int:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        differences = resized[:, 1:] > resized[:, :-1]
        result = 0
        for bit in differences.flat:
            result = (result << 1) | int(bit)
        return result

    @staticmethod
    def _has_visual_information(frame: np.ndarray) -> bool:
        """Reject blank/low-detail frames before they enter replay matching."""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
        contrast = float(np.std(gray))
        horizontal = float(np.mean(cv2.absdiff(gray[:, 1:], gray[:, :-1])))
        vertical = float(np.mean(cv2.absdiff(gray[1:, :], gray[:-1, :])))
        return contrast >= 12.0 and max(horizontal, vertical) >= 3.0

    @staticmethod
    def _band_keys(fingerprint: int) -> tuple[tuple[int, int], ...]:
        # With eight 8-bit bands, fingerprints within five changed bits must
        # share at least one complete band (pigeonhole principle).
        mask = (1 << 8) - 1
        return tuple(
            (band_index, (fingerprint >> (band_index * 8)) & mask)
            for band_index in range(8)
        )

    def _append_history(self, fingerprint: int, timestamp_ms: int) -> None:
        sample = _ReplaySample(
            sample_id=self._next_sample_id,
            timestamp_ms=timestamp_ms,
            fingerprint=fingerprint,
            shot_id=self._shot_id,
        )
        self._next_sample_id += 1
        self._history_ids.append(sample.sample_id)
        self._samples[sample.sample_id] = sample
        self._shot_history.setdefault(sample.shot_id, []).append(
            (sample.timestamp_ms, sample.sample_id)
        )
        for key in self._band_keys(fingerprint):
            self._hash_bands.setdefault(key, deque()).append(sample.sample_id)

        while len(self._history_ids) > self.history_size:
            expired_id = self._history_ids.popleft()
            expired = self._samples.pop(expired_id, None)
            if expired is None:
                continue
            for key in self._band_keys(expired.fingerprint):
                bucket = self._hash_bands.get(key)
                if bucket is None:
                    continue
                if bucket and bucket[0] == expired_id:
                    bucket.popleft()
                if not bucket:
                    self._hash_bands.pop(key, None)

    def _similar_prior_samples(
        self,
        fingerprint: int,
        timestamp_ms: int,
    ) -> list[_ReplaySample]:
        sample_ids: set[int] = set()
        per_band_limit = max(1, self.max_index_hits // 8)
        for key in self._band_keys(fingerprint):
            bucket = self._hash_bands.get(key, ())
            for index, sample_id in enumerate(reversed(bucket)):
                if index >= per_band_limit:
                    break
                sample_ids.add(sample_id)
                if len(sample_ids) >= self.max_index_hits:
                    break
            if len(sample_ids) >= self.max_index_hits:
                break
        matches = [
            sample
            for sample_id in sample_ids
            if (sample := self._samples.get(sample_id)) is not None
            and sample.shot_id != self._shot_id
            and timestamp_ms - sample.timestamp_ms >= self.min_gap_ms
            and (fingerprint ^ sample.fingerprint).bit_count()
            <= self.max_hamming_distance
        ]
        matches.sort(
            key=lambda sample: (
                (fingerprint ^ sample.fingerprint).bit_count(),
                -sample.timestamp_ms,
            )
        )
        return matches[: self.max_proposals_per_frame]

    def _nearest_source(
        self,
        *,
        source_shot_id: int,
        target_ms: int,
    ) -> tuple[int, int] | None:
        history = self._shot_history.get(source_shot_id, ())
        if not history:
            return None
        insertion = bisect_left(history, (target_ms, -1))
        best: tuple[int, _ReplaySample] | None = None
        # At normal video rates only a handful of samples can lie inside the
        # tolerance window. Stale, evicted IDs remain harmless tombstones.
        for index in range(max(0, insertion - 8), min(len(history), insertion + 9)):
            _, sample_id = history[index]
            sample = self._samples.get(sample_id)
            if sample is None:
                continue
            difference = abs(sample.timestamp_ms - target_ms)
            if best is None or difference < best[0]:
                best = (difference, sample)
        if best is None or best[0] > self.timestamp_tolerance_ms:
            return None
        return best[1].timestamp_ms, best[1].fingerprint

    def _propose_offsets(self, fingerprint: int, timestamp_ms: int) -> None:
        if self._shot_start_ms is None:
            return
        if timestamp_ms - self._shot_start_ms > self.proposal_window_ms:
            return
        for prior in self._similar_prior_samples(fingerprint, timestamp_ms):
            if len(self._candidates) >= self.max_candidates:
                break
            raw_offset = timestamp_ms - prior.timestamp_ms
            # Bucket offsets to the timestamp tolerance so adjacent frames do
            # not create hundreds of equivalent sequence hypotheses.
            bucket = round(raw_offset / self.timestamp_tolerance_ms) * self.timestamp_tolerance_ms
            key = (prior.shot_id, bucket)
            self._candidates.setdefault(
                key,
                _ReplayCandidate(
                    source_shot_id=prior.shot_id,
                    offset_ms=bucket,
                    first_match_ms=timestamp_ms,
                    last_match_ms=timestamp_ms,
                    unique_fingerprints={fingerprint},
                    last_fingerprint=fingerprint,
                ),
            )

    def _update_candidates(self, fingerprint: int, timestamp_ms: int) -> _ReplayCandidate | None:
        confirmed: _ReplayCandidate | None = None
        stale: list[tuple[int, int]] = []
        for key, candidate in self._candidates.items():
            source = self._nearest_source(
                source_shot_id=candidate.source_shot_id,
                target_ms=timestamp_ms - candidate.offset_ms,
            )
            candidate.comparisons += 1
            if source is not None:
                _, prior_fingerprint = source
                if (fingerprint ^ prior_fingerprint).bit_count() <= self.max_hamming_distance:
                    candidate.matches += 1
                    candidate.last_match_ms = timestamp_ms
                    if len(candidate.unique_fingerprints) < self.min_unique_fingerprints:
                        candidate.unique_fingerprints.add(fingerprint)
                    if (
                        candidate.last_fingerprint is not None
                        and (fingerprint ^ candidate.last_fingerprint).bit_count() >= 2
                    ):
                        candidate.dynamic_matches += 1
                    candidate.last_fingerprint = fingerprint
            # A sequence with no aligned match for 600 ms cannot recover into
            # a continuous replay candidate.
            if timestamp_ms - candidate.last_match_ms > 600:
                stale.append(key)
                continue
            duration = candidate.last_match_ms - candidate.first_match_ms
            ratio = candidate.matches / max(1, candidate.comparisons)
            if (
                duration >= self.min_sequence_ms
                and ratio >= self.min_match_ratio
                and len(candidate.unique_fingerprints)
                >= self.min_unique_fingerprints
                and candidate.dynamic_matches >= self.min_dynamic_matches
            ):
                confirmed = candidate
                break
        for key in stale:
            self._candidates.pop(key, None)
        return confirmed

    def update(self, frame: np.ndarray, timestamp_ms: int, *, is_scene_cut: bool) -> bool:
        fingerprint = self._dhash(frame)
        visually_informative = self._has_visual_information(frame)
        if self._shot_start_ms is None:
            self._shot_start_ms = timestamp_ms
        elif is_scene_cut:
            self._shot_id += 1
            self._shot_start_ms = timestamp_ms
            self._candidates.clear()
            self._confirmed_current_shot = False
            self.confirmed_start_ms = None
            self.confirmed_source_start_ms = None

        newly_confirmed = False
        if not self._confirmed_current_shot and visually_informative:
            self._propose_offsets(fingerprint, timestamp_ms)
            confirmed = self._update_candidates(fingerprint, timestamp_ms)
            if confirmed is not None:
                self._confirmed_current_shot = True
                self.confirmed_start_ms = self._shot_start_ms
                self.confirmed_source_start_ms = (
                    confirmed.first_match_ms - confirmed.offset_ms
                )
                self._candidates.clear()
                newly_confirmed = True

        if visually_informative:
            self._append_history(fingerprint, timestamp_ms)
        return newly_confirmed
