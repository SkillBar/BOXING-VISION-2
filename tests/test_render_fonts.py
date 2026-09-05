from types import SimpleNamespace

from boxing_vision import render


def test_export_renderer_uses_explicit_sf_before_platform_fallback(monkeypatch, tmp_path):
    regular = tmp_path / "SF-Pro-Text-Regular.otf"
    semibold = tmp_path / "SF-Pro-Text-Semibold.otf"
    regular.touch()
    semibold.touch()
    monkeypatch.setenv("BOXING_VISION_FONT_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(render.ImageFont, "truetype", lambda path, size: (calls.append((path, size)) or SimpleNamespace()))
    render._font.cache_clear()
    try:
        render._font(16)
        render._font(18, True)
        assert calls == [(str(regular), 16), (str(semibold), 18)]
    finally:
        render._font.cache_clear()


def test_invalid_supplied_font_does_not_crash_renderer(monkeypatch, tmp_path):
    broken = tmp_path / "SF-Pro-Text-Regular.otf"
    broken.touch()
    monkeypatch.setenv("BOXING_VISION_FONT_DIR", str(tmp_path))
    original = render.ImageFont.truetype
    def load(path, size):
        if str(path) == str(broken):
            raise OSError("invalid font")
        return original(path, size=size)
    monkeypatch.setattr(render.ImageFont, "truetype", load)
    render._font.cache_clear()
    try:
        assert render._font(16).getbbox("Боксёр")
    finally:
        render._font.cache_clear()
