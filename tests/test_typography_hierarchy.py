"""Source-contract checks supplement (not replace) live typography QA."""
from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from boxing_vision.ui_presenters import (
    STATIC_DIR,
    build_presentation_payload,
    render_workspace_shell,
)


class InspectorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, dict]] = []
        self.menus: list[dict] = []
        self.actions: list[tuple[dict, bool]] = []
        self.menu_summaries = 0
        self.navigation: set[str] = set()

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        inside_menu = any(name == "details" and "data-bv-review-menu" in attributes
                          for name, attributes in self.stack)
        if tag == "details" and "data-bv-review-menu" in values:
            self.menus.append(values)
        if tag == "summary" and inside_menu:
            self.menu_summaries += 1
        if "data-bv-review" in values:
            self.actions.append((values, inside_menu))
        if "data-bv-event-nav" in values:
            self.navigation.add(values["data-bv-event-nav"])
        if tag not in {"img", "input", "br", "hr", "meta", "link", "source", "wbr"}:
            self.stack.append((tag, values))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                self.stack = self.stack[:index]
                return


@pytest.fixture(scope="module")
def shell() -> str:
    return render_workspace_shell(build_presentation_payload([], {}, 30))


def test_event_feed_contains_only_time_title_and_outcome() -> None:
    source = (STATIC_DIR / "boxing_vision.js").read_text(encoding="utf-8")
    match = re.search(r"eventRow\(event\)\s*\{(.*?)\n    \},", source, re.DOTALL)
    assert match is not None
    body = match.group(1)
    classes = re.findall(r'element\(\s*["`]span["`]\s*,\s*["`]([^"`]+)', body)
    assert [value.split()[0] for value in classes] == [
        "bv-event-time", "bv-event-technique", "bv-event-outcome",
    ]
    assert "bv-event-confidence" not in body
    assert "describeEvent(event)" in body, "full event details remain available to assistive technology"


def test_review_actions_remain_inside_one_collapsed_native_details(shell: str) -> None:
    parser = InspectorParser()
    parser.feed(shell)
    assert len(parser.menus) == 1
    assert "open" not in parser.menus[0]
    assert parser.menu_summaries == 1
    assert {attrs["data-bv-review"] for attrs, _ in parser.actions} == {
        "confirmed", "rejected", "real_not_replay",
    }
    assert len(parser.actions) == 3
    assert all(inside for _, inside in parser.actions), "do not remove review access to simplify the inspector"
    assert all(attrs.get("aria-label") for attrs, _ in parser.actions)
    assert parser.navigation == {"-1", "1"}


def test_inspector_does_not_repeat_target_already_present_in_title(shell: str) -> None:
    assert "data-bv-inspector-title" in shell
    assert "data-bv-inspector-detail" in shell
    assert "data-bv-inspector-target" not in shell
    for field in ("fighter", "round", "review"):
        assert f"data-bv-inspector-{field}" in shell


def rules() -> list[tuple[str, str]]:
    css = (STATIC_DIR / "boxing_vision.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    return [(selector.strip(), body) for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)
            if not selector.strip().startswith("@")]


def explicit_weight(body: str) -> int | None:
    weights = re.findall(r"(?:^|;)\s*font(?:-weight)?\s*:\s*(400|500|600|700)(?:\s|;|!|$)", body)
    return int(weights[-1]) if weights else None


@pytest.mark.parametrize("control", [".bv-mode-control", ".bv-scope-control", ".bv-zoom-control"])
def test_ordinary_segment_buttons_use_regular_and_active_use_medium(control: str) -> None:
    ordinary, active = [], []
    for selector, body in rules():
        weight = explicit_weight(body)
        if weight is None:
            continue
        # Scope/zoom live under workspace-chrome, whose ID-scoped !important
        # declaration beats later legacy generic rules. Mode buttons live in
        # fighter rails and have a separate explicit base declaration.
        if control in selector and "button" in selector and 'aria-pressed="true"' in selector:
            if "#bv-result-workspace" in selector:
                active.append(weight)
        elif control == ".bv-mode-control":
            if control in selector and "button" in selector and "aria-pressed" not in selector:
                ordinary.append(weight)
        elif " ".join(selector.split()) == "#bv-result-workspace .bv-workspace-chrome :where(button, summary, select)":
            ordinary.append(weight)
    assert ordinary and ordinary[-1] == 400
    assert active and active[-1] == 500


def test_event_titles_are_regular_except_the_selected_row() -> None:
    ordinary, selected = [], []
    for selector, body in rules():
        if ".bv-event-technique" not in selector or "#bv-result-workspace" not in selector:
            continue
        weight = explicit_weight(body)
        if weight is not None:
            (selected if "aria-current" in selector else ordinary).append(weight)
    assert ordinary and ordinary[-1] == 400
    assert selected and selected[-1] == 600


def test_outcome_starts_below_title_not_below_time() -> None:
    areas = []
    for selector, body in rules():
        if ".bv-event-outcome" not in selector or "#bv-result-workspace" not in selector:
            continue
        areas.extend(re.findall(r"grid-area\s*:\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", body))
    assert areas
    assert areas[-1][:2] == ("2", "2")


def test_secondary_metrics_are_rectangular_tiles_with_large_display_numbers() -> None:
    """Verify scoped CSS declarations, not a claim about the installed font."""
    def declarations(exact_selector: str) -> dict[str, str]:
        result = {}
        for selector, body in rules():
            if exact_selector not in [part.strip() for part in selector.split(",")]:
                continue
            for declaration in body.split(";"):
                key, separator, value = declaration.partition(":")
                if separator:
                    result[key.strip()] = value.strip()
        return result

    grid = declarations("#bv-result-workspace .bv-rail-secondary")
    tile = declarations("#bv-result-workspace .bv-rail-stat")
    numbers = declarations("#bv-result-workspace .bv-rail-stat strong")
    label = declarations("#bv-result-workspace .bv-rail-stat small")
    ratio = declarations("#bv-result-workspace .bv-headline-metric strong > i")
    assert grid["display"] == "grid"
    assert grid["grid-template-columns"] == "repeat(2, minmax(0, 1fr))"
    assert grid["gap"] == "1px"
    assert tile["display"] == "flex" and tile["flex-direction"] == "column"
    assert int(tile["min-height"].removesuffix("px")) >= 68
    assert tile["background"] != grid["background"]
    assert numbers["font-family"] == "var(--bv-display) !important"
    assert numbers["font-weight"] == "500 !important"
    assert numbers["font-size"] == "28px" and numbers["line-height"] == "30px"
    assert label["font"] == "400 12px/16px var(--bv-ui)"
    assert ratio["font"] == "inherit" and ratio["font-style"] == "normal"
    css = (STATIC_DIR / "boxing_vision.css").read_text(encoding="utf-8")
    assert re.search(r'--bv-display\s*:\s*"BV Druk"', css)
