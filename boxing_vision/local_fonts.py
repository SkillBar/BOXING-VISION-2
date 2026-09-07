"""Serve only explicitly named, already installed display-font files locally."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote


def _druk_assets(directory: Path) -> tuple[list[Path], str]:
    """Return an exact-file allowlist and runtime CSS; never copy font files.

    ``fonts_dir`` exists solely to make filesystem behavior testable. Production
    callers omit it and inspect the two known filenames in ~/Library/Fonts.
    Missing Medium returns no assets or override rather than claiming a font
    has been loaded. This helper makes no determination about redistribution.
    """
    entries = (("DrukCyr-Medium.ttf", "400 500"), ("DrukCyr-Bold.ttf", "600 700"))
    assets: list[Path] = []
    faces: list[str] = []
    for filename, weight in entries:
        candidate = directory / filename
        try:
            if not candidate.is_file():
                if not assets:
                    return [], ""
                continue
            path = candidate.resolve(strict=True)
            modified_ns = path.stat().st_mtime_ns
        except OSError:
            if not assets:
                return [], ""
            continue
        url = f"/gradio_api/file={quote(path.as_posix(), safe='/')}?v={modified_ns}"
        assets.append(path)
        faces.append(
            '@font-face { font-family: "BV Druk Runtime"; '
            f'src: url("{url}") format("truetype"); font-weight: {weight}; '
            'font-style: normal; font-display: block; }'
        )
    override = (
        '#bv-result-workspace { --bv-display: "BV Druk Runtime", "BV Druk", '
        '"Druk Cyr", "Arial Narrow", Impact, sans-serif; }'
    )
    return assets, "\n".join([*faces, override])


def installed_display_font_assets(*, fonts_dir: Path | None = None) -> tuple[list[Path], str]:
    """Load exact local/private font files, without copying or installing them.

    Windows packaging may point BOXING_VISION_FONT_DIR at explicitly supplied
    application fonts. SF's CSS/system fallback remains when files are absent;
    the helper never claims that fallback is the actual SF font.
    """
    configured = os.environ.get("BOXING_VISION_FONT_DIR")
    directory = fonts_dir if fonts_dir is not None else Path(configured).expanduser() if configured else Path.home() / "Library" / "Fonts"
    assets, css = _druk_assets(directory)
    faces = [css] if css else []
    for style, weight in (("Regular", 400), ("Medium", 500), ("Semibold", 600), ("Bold", 700)):
        # Exact allowlist only: no serving the entire fonts directory.
        candidates = [directory / f"{prefix}{style}.{ext}" for prefix in ("SFProText-", "SF-Pro-Text-") for ext in ("otf", "ttf")]
        candidate = next((path for path in candidates if path.is_file()), None)
        if candidate is None:
            continue
        path = candidate.resolve(strict=True)
        url = f"/gradio_api/file={quote(path.as_posix(), safe='/')}?v={path.stat().st_mtime_ns}"
        assets.append(path)
        format_name = "opentype" if path.suffix == ".otf" else "truetype"
        faces.append('@font-face { font-family: "BV SF Pro"; '
                     f'src: url("{url}") format("{format_name}"); font-weight: {weight}; '
                     'font-style: normal; font-display: swap; }')
    return assets, "\n".join(faces)
