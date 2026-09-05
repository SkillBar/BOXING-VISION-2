import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

// Run pure presentation logic without a browser, model, or a synthetic video.
const context = vm.createContext({
  window: { cancelAnimationFrame() {}, requestAnimationFrame() { return 1; } },
  document: { querySelectorAll() { return []; }, addEventListener() {}, documentElement: {} },
  MutationObserver: class { observe() {} },
  console,
});
vm.runInContext(readFileSync(new URL("../boxing_vision/static/boxing_vision.js", import.meta.url), "utf8"), context);
const helpers = context.window.BoxingVisionState;
const workspace = context.window.BoxingVisionWorkspace;
const event = (id, ms, extra = {}) => ({
  event_id: id, attacker_id: "fighter_a", defender_id: "fighter_b", target: "body",
  peak_ms: ms, outcome: "likely_landed", confidence: .9, review_status: "confirmed", ...extra,
});
let checks = 0;
const check = (name, run) => { run(); checks += 1; console.log(`✓ ${name}`); };

check("legacy bbox coordinates never become precise body locations", () => {
  const old = event("old", 0, { target_point_norm: { x: .5, y: .3 }, target_point_confidence: .99, target_point_space: "defender_front_canonical_v1" });
  assert.equal(helpers.canonicalContact(old), null);
  assert.equal(helpers.canonicalContact({ ...old, target_point_source: "bbox_projection" }), null);
  assert.equal(helpers.canonicalContact({ ...old, target_point_source: "canonical_contact_v1", target_point_confidence: .79 }), null);
  assert.ok(helpers.canonicalContact({ ...old, target_point_source: "canonical_contact_v1" }));
});
check("invalid or out-of-atlas coordinates are rejected rather than clamped", () => {
  const proven = event("p", 0, { target_point_source: "canonical_contact_v1", target_point_space: "defender_front_canonical_v1", target_point_confidence: .95 });
  for (const point of [{ x: 1.2, y: .2 }, { x: .4, y: NaN }, { x: null, y: .5 }, [.4, .2]]) {
    assert.equal(helpers.canonicalContact({ ...proven, target_point_norm: point }), null);
  }
});
check("crossing a peak emits once; scrubbing does not replay previous hits", () => {
  const events = [event("a", 1000), event("b", 1400), event("c", 1800)];
  assert.equal(helpers.crossedMapEvents(events, 900, 1100).map((e) => e.event_id).join(), "a");
  assert.equal(helpers.crossedMapEvents(events, 1100, 1100).length, 0);
  assert.equal(helpers.crossedMapEvents(events, 1800, 900).length, 0);
  assert.equal(helpers.crossedMapEvents(events, 900, 1900, true).length, 0);
  assert.equal(helpers.crossedMapEvents(events, null, 1900).length, 0);
  assert.equal(helpers.crossedMapEvents(events, 0, 9000).length, 0);
});
check("missed, unclear, rejected and replay events do not pulse", () => {
  const invalid = [event("m", 100, { outcome: "missed" }), event("u", 200, { outcome: "unclear" }), event("r", 300, { is_replay: true }), event("x", 400, { review_status: "rejected" })];
  assert.equal(helpers.crossedMapEvents(invalid, 0, 500).length, 0);
});
check("animation uses video time and freezes when the video clock does", () => {
  const pulse = { startMs: 1000 };
  assert.equal(helpers.pulseProgress(pulse, 1300), .5);
  assert.equal(helpers.pulseProgress(pulse, 1300), .5);
  assert.equal(helpers.pulseProgress(pulse, 1600), 1);
  assert.equal(helpers.pulseProgress(pulse, 1100, true), 1);
});
check("impact strength changes the envelope, never its semantic outcome", () => {
  const soft = helpers.impactEnvelope(event("soft", 0, { impact_proxy_0_100: 10 }), .2);
  const hard = helpers.impactEnvelope(event("hard", 0, { impact_proxy_0_100: 90 }), .2);
  assert.ok(hard.fill > soft.fill);
  assert.ok(hard.fill < .72);
  assert.equal(helpers.impactEnvelope(event("p", 0), 1).energy, 0);
  assert.equal(helpers.impactEnvelope(event("p", 0), .2, true).energy, 0);
});
check("the player builds transport only, never a floating event details card", () => {
  assert.ok(!workspace.ensureVideoOverlay.toString().includes("bv-video-event-card"));
  assert.ok(!workspace.renderVideoOverlay.toString().includes("bv-video-event-card"));
});
check("zero landed remains neutral even when the other zone has hits", () => {
  assert.equal(helpers.zoneOpacity({ thrown: 12, landed: 0 }, 8), 0);
  assert.equal(helpers.zoneOpacity({ thrown: 0, landed: 0 }, 0), 0);
  assert.ok(helpers.zoneOpacity({ thrown: 12, landed: 3 }, 8) > 0);
});
check("received mode consistently takes the attacking opponent's color", () => {
  assert.equal(helpers.mapAttacker("fighter_a", "received"), "fighter_b");
  assert.equal(helpers.mapAttacker("fighter_b", "received"), "fighter_a");
  assert.equal(helpers.mapAttacker("fighter_a", "dealt"), "fighter_a");
});
check("trimmed video rounds use the local timeline origin, not source offset", () => {
  workspace.payload = { duration_ms: 12000, metadata: { fight_start_s: 360, round_length_s: 180, scheduled_rounds: 1 } };
  assert.equal(workspace.roundSegments()[0].start_ms, 0);
  assert.equal(workspace.roundSegments()[0].end_ms, 12000);
});
check("the same knee crop preserves the original atlas aspect", () => {
  const crop = helpers.atlasGeometry();
  assert.ok(Math.abs(crop.aspect - crop.width * 1024 / (crop.height * 1536)) < 1e-10);
  assert.ok(crop.height < .8 && crop.height > .65);
  const custom = helpers.atlasGeometry({ canvas_size: [100, 200], viewport_crop: { x: .1, y: .05, width: .8, height: .5 } });
  assert.equal(custom.aspect, .8);
});
check("timeline handles 0 / 1 / 500 / 2000 events without changing playhead", () => {
  const ctx = new Proxy({}, { get(target, key) { return key in target ? target[key] : () => {}; } });
  workspace.canvas = { width: 1200, height: 150 };
  workspace.context = ctx;
  workspace.dpr = 1;
  workspace.state.currentTimeMs = 12345;
  workspace.state.selectedEventId = null;
  for (const count of [0, 1, 500, 2000]) {
    workspace.payload = { duration_ms: 300000, metadata: { round_length_s: 180, scheduled_rounds: 2 }, events: Array.from({length:count}, (_, i) => event(String(i), i * 140, { start_ms: i * 140 - 50, end_ms: i * 140 + 100, attacker_id: i % 2 ? "fighter_b" : "fighter_a", outcome: ["likely_landed", "blocked", "missed", "unclear"][i % 4] })) };
    for (const zoom of [1, 8]) {
      workspace.state.zoom = zoom;
      workspace.canvas.width = 1200 * zoom;
      const began = performance.now();
      workspace.drawTimeline();
      assert.equal(workspace.state.currentTimeMs, 12345);
      assert.ok(workspace.hitRegions.length <= count);
      assert.equal(workspace.hitRegions.reduce((total, region) => total + region.events.length, 0), count);
      console.log(`  ${count} events, ${zoom}×: ${(performance.now() - began).toFixed(1)} ms (logic only)`);
    }
  }
  workspace.state.zoom = 1;
});
check("ordinary playback reuses a fighter's mounted controls and canvas", () => {
  const mount = { replaceChildren() { throw new Error("must preserve mounted DOM"); } };
  const existing = { mount, head: { isConnected: true }, focusedButton: {} };
  workspace.panelViews.set("fighter_a", existing);
  const original = workspace.updateFighterPanel;
  let received = null;
  workspace.updateFighterPanel = (view) => { received = view; };
  workspace.renderFighterPanel(mount, "fighter_a", []);
  assert.equal(received, existing);
  workspace.updateFighterPanel = original;
});
check("zone-only events illuminate a mask without inventing a point or ring", () => {
  let arcs = 0;
  const ctx = { clearRect() {}, arc() { arcs += 1; } };
  const zones = {};
  const stage = { querySelector(selector) { return zones[selector] ||= { dataset: { baseOpacity: "0" }, style: { setProperty(key, value) { this[key] = value; } } }; } };
  const canvas = { width: 512, height: 768, parentElement: stage, getContext: () => ctx };
  workspace.payload = { events: [event("zone", 100)] };
  workspace.video = null;
  workspace.state.currentTimeMs = 150;
  workspace.state.selectedEventId = null;
  workspace.drawBodyMap(canvas, "fighter_a", "dealt", "latest", workspace.payload.events);
  assert.equal(arcs, 0);
  assert.ok(Number(zones[".bv-body-zone-body"].style["--bv-zone-opacity"]) > 0);
  assert.equal(Number(zones[".bv-body-zone-head"].style["--bv-zone-opacity"]), 0);
  workspace.payload.events.push(event("miss", 140, { outcome: "missed" }));
  workspace.drawBodyMap(canvas, "fighter_a", "dealt", "latest", workspace.payload.events);
  assert.equal(Number(zones[".bv-body-zone-body"].style["--bv-zone-opacity"]), 0);
});
check("theater state survives Gradio replacing the outer group classes", () => {
  const original = {
    root: workspace.root, sync: workspace.syncTheaterButton,
    body: context.document.body, overflow: workspace.previousBodyOverflow,
    fullscreen: context.document.fullscreenElement,
    webkitFullscreen: context.document.webkitFullscreenElement,
  };
  const classes = new Set();
  const root = {
    dataset: {},
    classList: {
      contains: (value) => classes.has(value),
      add: (value) => classes.add(value),
      remove: (value) => classes.delete(value),
    },
  };
  let syncs = 0;
  try {
    workspace.root = root;
    workspace.syncTheaterButton = () => { syncs += 1; };
    context.document.body = { style: { overflow: "auto" } };
    context.document.fullscreenElement = null;
    context.document.webkitFullscreenElement = null;
    workspace.toggleTheater();
    assert.equal(root.dataset.bvTheaterActive, "true");
    assert.equal(context.document.body.style.overflow, "hidden");
    // Svelte can replace className after the click. It must not turn a second
    // click into another entry or leave scrolling permanently locked.
    classes.clear();
    assert.equal(root.classList.contains("bv-theater-mode"), false);
    assert.equal(workspace.isTheaterActive(), true);
    workspace.toggleTheater();
    assert.notEqual(root.dataset.bvTheaterActive, "true");
    assert.equal(workspace.isTheaterActive(), false);
    assert.equal(context.document.body.style.overflow, "auto");
    assert.equal(syncs, 2);
  } finally {
    workspace.root = original.root;
    workspace.syncTheaterButton = original.sync;
    workspace.previousBodyOverflow = original.overflow;
    context.document.body = original.body;
    context.document.fullscreenElement = original.fullscreen;
    context.document.webkitFullscreenElement = original.webkitFullscreen;
  }
});
check("FIT uses the measured scroller at narrow, desktop and zoomed-window sizes", () => {
  const original = {
    canvas: workspace.canvas, scroller: workspace.scroller, context: workspace.context,
    shell: workspace.shell, draw: workspace.drawTimeline, center: workspace.centerPlayhead,
    review: workspace.hasReviewEvents, dpr: workspace.dpr, zoom: workspace.state.zoom,
    time: workspace.state.currentTimeMs, pixelRatio: context.window.devicePixelRatio,
  };
  try {
    workspace.canvas = { style: {}, getContext: () => ({ setTransform() {} }) };
    workspace.scroller = { clientWidth: 0, scrollLeft: 0 };
    workspace.shell = { classList: { toggle() {} }, querySelector: () => null };
    workspace.drawTimeline = () => {};
    workspace.centerPlayhead = () => {};
    workspace.hasReviewEvents = () => true;
    workspace.state.currentTimeMs = 43210;
    context.window.devicePixelRatio = 2;
    // 256/320/512 approximate usable timeline widths at browser zoom 200%.
    for (const width of [0, 192, 256, 320, 512, 768, 1024, 1280, 1440]) {
      workspace.scroller.clientWidth = width;
      for (const zoom of [1, 2, 4, 8]) {
        workspace.state.zoom = zoom;
        workspace.resizeCanvas(false);
        const expected = Math.max(1, width) * zoom;
        assert.equal(workspace.canvas.style.width, `${expected}px`);
        assert.equal(workspace.canvas.width, expected * 2);
        assert.equal(workspace.state.currentTimeMs, 43210);
      }
    }
  } finally {
    workspace.canvas = original.canvas;
    workspace.scroller = original.scroller;
    workspace.context = original.context;
    workspace.shell = original.shell;
    workspace.drawTimeline = original.draw;
    workspace.centerPlayhead = original.center;
    workspace.hasReviewEvents = original.review;
    workspace.dpr = original.dpr;
    workspace.state.zoom = original.zoom;
    workspace.state.currentTimeMs = original.time;
    context.window.devicePixelRatio = original.pixelRatio;
  }
});
check("center layout follows actual width without moving the playhead or focus", () => {
  const original = workspace.root;
  let width = 0;
  const writes = [];
  const dataset = new Proxy({}, { set(target, key, value) { writes.push(value); target[key] = value; return true; } });
  try {
    workspace.root = { dataset, querySelector: () => ({ getBoundingClientRect: () => ({ width }) }) };
    const time = workspace.state.currentTimeMs;
    const selected = workspace.state.selectedEventId;
    for (const [w, expected] of [[606, "stacked"], [606, "stacked"], [860, "compact"], [861, "wide"], [479, "narrow"], [0, "narrow"], [680, "stacked"], [681, "compact"]]) {
      width = w;
      workspace.syncCenterLayout();
      assert.equal(dataset.bvCenterLayout, expected);
      assert.equal(workspace.state.currentTimeMs, time);
      assert.equal(workspace.state.selectedEventId, selected);
    }
    assert.equal(writes.length, 6); // No mutation loop on unchanged/hidden sizes.
  } finally { workspace.root = original; }
});
check("wheel scrolls the page at FIT and zoomed edges without moving video time", () => {
  const original = workspace.scroller;
  const time = workspace.state.currentTimeMs;
  const selected = workspace.state.selectedEventId;
  let prevented = 0;
  const wheel = (deltaY, extra = {}) => workspace.onWheel({ deltaY, deltaX: 0, deltaMode: 0,
    preventDefault() { prevented += 1; }, ...extra });
  try {
    workspace.scroller = { scrollWidth: 600, clientWidth: 600, scrollLeft: 0 };
    wheel(120);
    assert.equal(prevented, 0);
    assert.equal(workspace.scroller.scrollLeft, 0);
    workspace.scroller.scrollWidth = 1200;
    wheel(120);
    assert.equal(prevented, 1);
    assert.equal(workspace.scroller.scrollLeft, 120);
    wheel(3, { deltaMode: 1 });
    assert.equal(workspace.scroller.scrollLeft, 180);
    wheel(1, { deltaMode: 2 });
    assert.equal(workspace.scroller.scrollLeft, 600);
    const atEdge = prevented;
    wheel(120);
    assert.equal(prevented, atEdge);
    wheel(-900);
    assert.equal(workspace.scroller.scrollLeft, 0);
    const atStart = prevented;
    wheel(-120);
    assert.equal(prevented, atStart);
    wheel(2, { deltaX: 60 }); // Native horizontal touchpad scrolling.
    assert.equal(prevented, atStart);
    assert.equal(workspace.state.currentTimeMs, time);
    assert.equal(workspace.state.selectedEventId, selected);
  } finally { workspace.scroller = original; }
});
check("modified wheel still zooms around the pointer", () => {
  const original = { scroller: workspace.scroller, zoom: workspace.setZoom, level: workspace.state.zoom };
  let received;
  let prevented = 0;
  try {
    workspace.scroller = { scrollWidth: 600, clientWidth: 600, scrollLeft: 0 };
    workspace.state.zoom = 1;
    workspace.setZoom = (...args) => { received = args; };
    workspace.onWheel({ ctrlKey: true, deltaY: -1, clientX: 320, preventDefault() { prevented += 1; } });
    assert.deepEqual(received, [2, 320]);
    assert.equal(prevented, 1);
  } finally {
    workspace.scroller = original.scroller;
    workspace.setZoom = original.zoom;
    workspace.state.zoom = original.level;
  }
});
check("bundled demo opens paused once and does not reset the playhead on rebind", () => {
  const saved = { payload: workspace.payload, video: workspace.video, demos: workspace.initializedDemos,
    inspector: workspace.renderInspector, panels: workspace.renderPanels, a11y: workspace.renderA11yListbox,
    time: workspace.state.currentTimeMs, selected: workspace.state.selectedEventId };
  let pauses = 0;
  try {
    workspace.payload = { duration_ms: 188000, events: [event("demo", 49933)],
      metadata: { bundled_demo: true, demo_read_only: true, demo_start_ms: 49583, demo_selected_event_id: "demo" } };
    workspace.initializedDemos = new Set();
    workspace.video = { readyState: 0, duration: NaN, currentTime: 0, pause() { pauses += 1; } };
    workspace.renderInspector = workspace.renderPanels = workspace.renderA11yListbox = () => {};
    assert.equal(workspace.applyInitialDemoState(), false);
    workspace.video.readyState = 1;
    workspace.video.duration = 188;
    assert.equal(workspace.applyInitialDemoState(), true);
    assert.equal(workspace.video.currentTime, 49.583);
    assert.equal(workspace.state.selectedEventId, "demo");
    workspace.video.currentTime = 80;
    assert.equal(workspace.applyInitialDemoState(), false);
    assert.equal(workspace.video.currentTime, 80);
    assert.equal(pauses, 1);
  } finally {
    workspace.payload = saved.payload; workspace.video = saved.video; workspace.initializedDemos = saved.demos;
    workspace.renderInspector = saved.inspector; workspace.renderPanels = saved.panels; workspace.renderA11yListbox = saved.a11y;
    workspace.state.currentTimeMs = saved.time; workspace.state.selectedEventId = saved.selected;
  }
});
console.log(`${checks} workspace regression scenarios passed.`);
