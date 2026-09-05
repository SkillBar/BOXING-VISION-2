"""Width-specific source contracts; actual browser geometry is checked separately."""

import re
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "boxing_vision/static/boxing_vision.css"
MARKER = "/* The desktop side rails can leave only 606 px"
COMPACT = '#bv-result-workspace:is([data-bv-center-layout="compact"], [data-bv-center-layout="stacked"], [data-bv-center-layout="narrow"])'
STACKED = '#bv-result-workspace:is([data-bv-center-layout="stacked"], [data-bv-center-layout="narrow"])'
NARROW = '#bv-result-workspace[data-bv-center-layout="narrow"]'


def _block(css, selector):
    match = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", css)
    assert match, selector
    return match.group(1)


def test_player_width_queries_are_based_on_center_not_page():
    css = CSS.read_text().split(MARKER, 1)[1]
    # Gradio's CSS scoper discards CSSContainerRule. These must remain flat.
    assert "@container bv-center" not in css
    assert COMPACT in css and STACKED in css and NARROW in css
    js = CSS.with_suffix(".js").read_text()
    assert 'const center = this.root?.querySelector?.("#bv-center-stack")' in js
    assert "center.getBoundingClientRect().width" in js
    assert "this.syncCenterLayout();" in js
    assert "font-size:" not in css and "font:" not in css


def test_round_is_in_grid_flow_and_never_absolute_over_time():
    css = CSS.read_text().split(MARKER, 1)[1]
    round_style = _block(css, "#bv-result-workspace .bv-video-round")
    assert "position: static" in round_style
    assert "transform: none" in round_style
    header = _block(css, COMPACT + " .bv-video-topbar")
    assert "grid-template-columns: auto minmax(0, 1fr) auto auto" in header


def test_606_pixel_timeline_keeps_controls_on_two_reserved_rows():
    css = CSS.read_text().split(MARKER, 1)[1]
    toolbar = _block(css, STACKED + " .bv-timeline-toolbar")
    assert "grid-template-rows: auto auto" in toolbar
    scope = _block(css, STACKED + " .bv-timeline-toolbar .bv-scope-control")
    assert "grid-column: 1 / -1" in scope and "grid-row: 1" in scope
    filters = _block(css, STACKED + " .bv-timeline-toolbar .bv-more-filters")
    zoom = _block(css, STACKED + " .bv-timeline-toolbar .bv-zoom-control")
    assert "grid-column: 1" in filters and "grid-row: 2" in filters
    assert "grid-column: 2" in zoom and "grid-row: 2" in zoom


def test_duplicate_time_is_hidden_even_with_legacy_fullscreen_specificity():
    css = CSS.read_text().split(MARKER, 1)[1]
    readout = _block(css, COMPACT + " .bv-timeline-toolbar .bv-time-readout")
    assert "display: none !important" in readout


def test_extremely_narrow_center_keeps_play_and_exit_plus_scroll_free_zoom():
    css = CSS.read_text().split(MARKER, 1)[1]
    header = _block(css, NARROW + " .bv-video-topbar")
    assert "grid-template-columns: minmax(0, 1fr) auto auto" in header
    zoom = _block(css, NARROW + " .bv-timeline-toolbar .bv-zoom-control")
    assert "grid-row: 3" in zoom and "grid-column: 1 / -1" in zoom


def test_tablet_player_reserves_room_for_timeline_and_transport():
    css = CSS.read_text()
    assert "max-height: max(180px, calc(100dvh - 240px))" in css
    assert "#result-video-card { max-height: none; }" not in css
    assert "@media (min-width: 768px) and (max-width: 1179px)" in css
    assert "grid-template-columns: 128px minmax(0, 1fr) 160px" in css
