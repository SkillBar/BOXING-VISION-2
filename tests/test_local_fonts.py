from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

from boxing_vision.local_fonts import installed_display_font_assets


def test_missing_installed_medium_emits_no_font_override(tmp_path: Path) -> None:
    assert installed_display_font_assets(fonts_dir=tmp_path) == ([], "")
    (tmp_path / "DrukCyr-Bold.ttf").touch()
    assert installed_display_font_assets(fonts_dir=tmp_path) == ([], "")


def test_medium_is_served_as_one_exact_file_not_its_directory(tmp_path: Path) -> None:
    medium = tmp_path / "DrukCyr-Medium.ttf"
    medium.touch()
    assets, css = installed_display_font_assets(fonts_dir=tmp_path)
    assert assets == [medium.resolve()]
    assert all(path.is_file() for path in assets)
    assert tmp_path not in assets
    assert css.count("@font-face") == 1
    assert 'font-family: "BV Druk Runtime"' in css
    assert "font-weight: 400 500" in css
    assert "font-style: normal" in css
    assert "font-display: block" in css
    assert '#bv-result-workspace { --bv-display: "BV Druk Runtime", "BV Druk"' in css
    assert "local(" not in css


def test_optional_bold_has_its_own_weight_range_and_exact_allowlist_entry(tmp_path: Path) -> None:
    medium = tmp_path / "DrukCyr-Medium.ttf"
    bold = tmp_path / "DrukCyr-Bold.ttf"
    medium.touch()
    bold.touch()
    assets, css = installed_display_font_assets(fonts_dir=tmp_path)
    assert assets == [medium.resolve(), bold.resolve()]
    assert css.count("@font-face") == 2
    assert "font-weight: 600 700" in css


def test_font_paths_are_quoted_and_versioned_by_mtime(tmp_path: Path) -> None:
    font_dir = tmp_path / "Local Fonts # test"
    font_dir.mkdir()
    medium = font_dir / "DrukCyr-Medium.ttf"
    medium.touch()
    os.utime(medium, ns=(1_700_000_000_000_000_000, 1_700_000_000_100_000_000))
    assets, css = installed_display_font_assets(fonts_dir=font_dir)
    assert assets == [medium.resolve()]
    assert f'/gradio_api/file={quote(str(medium.resolve()), safe="/")}?v={medium.stat().st_mtime_ns}' in css
    assert "Local%20Fonts%20%23%20test" in css
    assert "Local Fonts # test" not in css
    os.utime(medium, ns=(1_700_000_000_000_000_000, 1_700_000_000_200_000_000))
    assert installed_display_font_assets(fonts_dir=font_dir)[1] != css


def test_unrelated_and_lookalike_files_are_not_exposed(tmp_path: Path) -> None:
    medium = tmp_path / "DrukCyr-Medium.ttf"
    medium.touch()
    for filename in ("DrukCyr-Medium-Trial.ttf", "DrukCyr-Regular.ttf", "DrukCyr-Medium.otf", "Private.ttf"):
        (tmp_path / filename).touch()
    # A directory with the optional filename is not a font file.
    (tmp_path / "DrukCyr-Bold.ttf").mkdir()
    assets, css = installed_display_font_assets(fonts_dir=tmp_path)
    assert assets == [medium.resolve()]
    assert "Private" not in css and "Trial" not in css and "Bold" not in css


def test_default_lookup_is_the_two_named_files_under_home_library_fonts(monkeypatch, tmp_path: Path) -> None:
    font_dir = tmp_path / "Library" / "Fonts"
    font_dir.mkdir(parents=True)
    medium = font_dir / "DrukCyr-Medium.ttf"
    medium.touch()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assets, _ = installed_display_font_assets()
    assert assets == [medium.resolve()]


def test_windows_explicit_sf_files_work_without_mac_or_druk(monkeypatch, tmp_path):
    font = tmp_path / "SF-Pro-Text-Regular.otf"
    font.touch()
    (tmp_path / "private.ttf").touch()
    monkeypatch.setenv("BOXING_VISION_FONT_DIR", str(tmp_path))
    assets, css = installed_display_font_assets()
    assert assets == [font.resolve()]
    assert 'font-family: "BV SF Pro"' in css
    assert 'font-weight: 400' in css
    assert 'format("opentype")' in css
    assert 'BV Druk Runtime' not in css
    assert 'private.ttf' not in css


def test_explicit_font_directory_takes_precedence_over_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("BOXING_VISION_FONT_DIR", "/not/the/font/directory")
    font = tmp_path / "SFProText-Semibold.ttf"
    font.touch()
    assets, css = installed_display_font_assets(fonts_dir=tmp_path)
    assert assets == [font.resolve()]
    assert "font-weight: 600" in css
