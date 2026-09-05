"""Portable, read-only investor example; no inference, cache or auto-discovery."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .desktop import DesktopSetupError, safe_bundle_file, sha256_file

MANIFEST = "demo-manifest.json"
REQUIRED_FILES = {"workspace-preview.mp4", "events.json", "summary.json"}
_MEDIA = re.compile(r"(?:previews/[A-Za-z0-9_.-]+\.webp|profiles/[A-Za-z0-9_.-]+\.(?:png|jpg|jpeg|webp))\Z")


def _allowed_name(name: str) -> bool:
    return name in REQUIRED_FILES | {"preview_manifest.json"} or bool(_MEDIA.fullmatch(name))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _portable(value: Any) -> Any:
    """Drop workstation references; preserve measured values and review gates."""
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()
                if not key.endswith(("_path", "_url", "_dir"))}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, str) and (
        PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
        or "://" in value or "/Users/" in value or "/private/" in value
    ):
        return None
    return value


def verify_demo(root: Path) -> dict[str, Any]:
    """Strict local allowlist. Never scan arbitrary runs or follow linked assets."""
    root = root.resolve(strict=True)
    manifest = _json(root / MANIFEST)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise DesktopSetupError("Неподдерживаемый комплект демонстрационного видео")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise DesktopSetupError("Демонстрационный комплект не содержит файлов")
    names: set[str] = set()
    for row in records:
        name = row.get("path") if isinstance(row, dict) else None
        if not isinstance(name, str) or not _allowed_name(name) or name in names:
            raise DesktopSetupError("Недопустимый или повторяющийся файл демо")
        names.add(name)
        path = safe_bundle_file(root, name)
        raw = root / name
        if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
               for p in (raw, *raw.parents) if p != root and p.is_relative_to(root)):
            raise DesktopSetupError("Демо не должно содержать ссылки на внешние файлы")
        if not path.is_file() or path.stat().st_size == 0 or sha256_file(path) != row.get("sha256"):
            raise DesktopSetupError(f"Демонстрационный файл повреждён: {name}")
    if not REQUIRED_FILES.issubset(names):
        raise DesktopSetupError("В демо не хватает видео, событий или статистики")
    events, summary = _json(root / "events.json"), _json(root / "summary.json")
    if not isinstance(events, list) or not all(isinstance(e, dict) for e in events) or not isinstance(summary, dict):
        raise DesktopSetupError("Некорректные данные демонстрационного анализа")
    metadata = summary.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("demo_read_only") is not True or metadata.get("bundled_demo") is not True:
        raise DesktopSetupError("Демонстрационный анализ должен быть доступен только для просмотра")
    if "preview_manifest.json" in names:
        previews = _json(root / "preview_manifest.json")
        if not isinstance(previews, dict) or not isinstance(previews.get("sheets"), list) or any(s not in names for s in previews["sheets"]):
            raise DesktopSetupError("Не хватает кадров предпросмотра демо")
    return manifest


def prepare_demo(source_run: Path, output: Path, *, start_ms: int = 49583) -> Path:
    """Create one explicitly chosen snapshot in a new directory; originals stay intact."""
    source = source_run.resolve(strict=True)
    target = output.resolve()
    if target.exists() or target == source or target.is_relative_to(source):
        raise DesktopSetupError("Для демо нужен новый отдельный каталог; существующие файлы не перезаписываются")
    video = source / "workspace-preview.mp4"
    if not video.is_file() or video.stat().st_size == 0:
        raise DesktopSetupError("Сначала создайте актуальное видео workspace-preview.mp4")
    events, summary = _json(source / "events.json"), _json(source / "summary.json")
    if not isinstance(events, list) or not all(isinstance(e, dict) for e in events) or not isinstance(summary, dict):
        raise DesktopSetupError("Повреждён формат готового анализа")
    metadata = summary.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("tracking_preview_stale") is not False:
        raise DesktopSetupError("Сначала обновите трекинг в плеере; устаревшее видео нельзя включить в демо")
    duration = float(metadata.get("duration_s", 0)) * 1000
    if not math.isfinite(duration) or duration <= 0:
        raise DesktopSetupError("Не указана длительность демонстрационного видео")
    if not isinstance(start_ms, int) or start_ms < 0 or start_ms >= duration:
        raise DesktopSetupError("Стартовый кадр демо должен находиться внутри видео")
    initial = min(events, key=lambda e: abs(float(e.get("peak_ms", 0)) - start_ms - 350), default={})
    summary = _portable(summary)
    summary["metadata"].update({
        "bundled_demo": True, "demo_read_only": True, "demo_start_ms": start_ms,
        "demo_selected_event_id": initial.get("event_id"), "render_cache_available": False,
        "workspace_preview": "workspace-preview.mp4", "workspace_preview_mode": "tracking_only",
    })
    # Each source media file is explicitly referenced, not globbed. No source
    # video, logs, identity profiles, caches or old/stale annotated MP4 is copied.
    copies = {"workspace-preview.mp4": video}
    previews = None
    if (source / "preview_manifest.json").is_file():
        previews = _json(source / "preview_manifest.json")
        if not isinstance(previews, dict) or not isinstance(previews.get("sheets"), list):
            raise DesktopSetupError("Некорректный манифест кадров предпросмотра")
        for name in previews["sheets"]:
            if not isinstance(name, str) or not _MEDIA.fullmatch(name) or not name.startswith("previews/"):
                raise DesktopSetupError("Некорректный путь кадра предпросмотра")
            copies[name] = safe_bundle_file(source, name)
    for profile in summary.get("fighters", {}).values():
        if not isinstance(profile, dict):
            continue
        name = profile.get("portrait_filename")
        if name:
            if not isinstance(name, str) or not _MEDIA.fullmatch(name) or not name.startswith("profiles/"):
                raise DesktopSetupError("Некорректное имя портрета демо")
            copies[name] = safe_bundle_file(source, name)
    for name, path in copies.items():
        raw = source / name
        if not path.is_file() or any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                                    for p in (raw, *raw.parents) if p != source and p.is_relative_to(source)):
            raise DesktopSetupError(f"Нужна локальная копия файла демо: {name}")
    target.mkdir(parents=True)
    for name, path in copies.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    _write_json(target / "events.json", _portable(events))
    _write_json(target / "summary.json", summary)
    names = set(copies) | {"events.json", "summary.json"}
    if previews is not None:
        _write_json(target / "preview_manifest.json", previews)
        names.add("preview_manifest.json")
    _write_json(target / MANIFEST, {
        "schema_version": 1, "title": "Готовый разбор спарринга", "audience": "private_investor_preview",
        "files": [{"path": name, "sha256": sha256_file(target / name)} for name in sorted(names)],
    })
    verify_demo(target)
    return target


def stage_demo(source: Path, payload: Path) -> dict[str, str]:
    """Copy only the verified demo allowlist into a new Windows payload."""
    manifest = verify_demo(source)
    target = payload / "demo"
    target.mkdir(parents=True, exist_ok=False)
    for row in [*manifest["files"], {"path": MANIFEST}]:
        destination = target / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / row["path"], destination)
    verify_demo(target)
    return {"role": "demo_manifest", "path": f"demo/{MANIFEST}", "sha256": sha256_file(target / MANIFEST)}
