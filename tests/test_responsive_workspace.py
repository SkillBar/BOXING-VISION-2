"""Source contracts supplement, but do not replace, native/browser layout QA."""
import re
from pathlib import Path

import pytest

from boxing_vision import ui

CSS_PATH = Path(__file__).resolve().parents[1] / "boxing_vision/static/boxing_vision.css"
RESPONSIVE_MARKER = "/* Narrow native windows and 200% browser zoom"


def declarations(source: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", source)
    assert match, f"Missing explicit responsive selector: {selector}"
    return match.group(1)


def test_mobile_scope_spans_toolbar_with_specificity_over_legacy_rule():
    tail = CSS_PATH.read_text().split(RESPONSIVE_MARKER, 1)[1]
    scope = declarations(tail, "#bv-result-workspace .bv-timeline-toolbar .bv-scope-control")
    assert "grid-column: 1 / -1" in scope
    assert "min-width: 0" in scope
    assert "width: 100%" in scope
    toolbar = declarations(tail, "#bv-result-workspace .bv-timeline-toolbar")
    assert "grid-template-columns: minmax(0, 1fr) auto" in toolbar


def test_small_window_popovers_are_bounded_without_reducing_typography():
    tail = CSS_PATH.read_text().split(RESPONSIVE_MARKER, 1)[1]
    filters = declarations(tail, "#bv-result-workspace .bv-more-filters-popover.bv-filter-row")
    assert "position: fixed" in filters
    assert "min-width: 0" in filters
    assert "max-width: calc(100dvw - 24px)" in filters
    assert "max-height: min(520px, 60dvh)" in filters
    tooltip = declarations(tail, "#bv-result-workspace .bv-timeline-tooltip")
    assert "max-width: min(300px, calc(100dvw - 16px))" in tooltip
    assert "font-size" not in tail


def test_narrow_transport_reserves_columns_for_play_and_exit():
    tail = CSS_PATH.read_text().split(RESPONSIVE_MARKER, 1)[1]
    transport = declarations(tail, "#bv-result-workspace .bv-video-topbar")
    assert "grid-template-columns: minmax(0, 1fr) auto auto" in transport
    assert ".bv-video-theater {" not in tail
    assert ".bv-video-transport {" not in tail


def test_200_percent_small_window_can_reflow_zoom_controls_to_their_own_row():
    narrowest = CSS_PATH.read_text().split("@media (max-width: 479px)", 1)[1]
    toolbar = declarations(narrowest, "#bv-result-workspace .bv-timeline-toolbar")
    zoom = declarations(narrowest, "#bv-result-workspace .bv-timeline-toolbar .bv-zoom-control")
    assert "grid-template-columns: minmax(0, 1fr)" in toolbar
    assert "grid-column: 1 / -1" in zoom
    assert "min-width: 0" in zoom


@pytest.mark.parametrize("value,hidden", [(None, False), ("0", False), ("true", False), ("1", True)])
def test_desktop_chrome_flag_does_not_modify_ordinary_browser(monkeypatch, value, hidden):
    if value is None:
        monkeypatch.delenv("BOXING_VISION_DESKTOP", raising=False)
    else:
        monkeypatch.setenv("BOXING_VISION_DESKTOP", value)
    css = ui._desktop_chrome_css()
    assert bool(css) is hidden
    if hidden:
        assert ".gradio-container footer { display: none !important; }" in css


@pytest.mark.parametrize("desktop", [False, True])
def test_app_injects_footer_hiding_only_for_native_window(monkeypatch, desktop):
    monkeypatch.delenv("BOXING_VISION_DEMO_RUN", raising=False)
    monkeypatch.setenv("BOXING_VISION_DESKTOP", "1" if desktop else "0")
    app = ui.build_app()
    footer_rule = ".gradio-container footer { display: none !important; }"
    assert (footer_rule in app.config["css"]) is desktop
    assert app.analytics_enabled is False
