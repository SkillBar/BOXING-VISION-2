(() => {
  "use strict";

  if (window.BoxingVisionWorkspace) {
    window.BoxingVisionWorkspace.hydrate();
    return;
  }

  const COLORS = {
    fighter_a: "#ff514a",
    fighter_b: "#5682ff",
    grid: "#20252b",
    text: "#f5f7fa",
    muted: "#c2c8d0",
    lime: "#98ff48",
    warning: "#f4c15d",
  };
  const OUTCOME_LABELS = {
    likely_landed: "Вероятное попадание",
    blocked: "Блокировано",
    missed: "Промах",
    unclear: "Исход не определён",
  };
  const TECHNIQUE_LABELS = {
    jab: "джеб",
    cross: "кросс",
    straight: "прямой удар",
    hook: "хук",
    uppercut: "апперкот",
    unknown: "тип удара не определён",
  };

  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  };

  const button = (className, text, attributes = {}) => {
    const node = element("button", className, text);
    node.type = "button";
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  };

  const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));
  const number = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  // Pure presentation state is exported for regression tests. The atlas stays
  // in canonical coordinates; one crop transform covers every visual layer.
  const DEFAULT_ATLAS = Object.freeze({
    canvas_size: [1024, 1536],
    viewport_crop: { x: .18, y: 0, width: .64, height: .72 },
  });
  const atlasGeometry = (atlas = {}) => {
    const size = atlas.canvas_size || DEFAULT_ATLAS.canvas_size;
    const raw = atlas.viewport_crop || DEFAULT_ATLAS.viewport_crop;
    const x = clamp(number(raw.x, .18), 0, .9);
    const y = clamp(number(raw.y, 0), 0, .9);
    const width = clamp(number(raw.width, .64), .1, 1 - x);
    const height = clamp(number(raw.height, .72), .1, 1 - y);
    return { x, y, width, height, aspect: width * number(size[0], 1024) / (height * number(size[1], 1536)) };
  };
  const isMapEvent = (event) => ["head", "body"].includes(event?.target)
    && ["likely_landed", "blocked"].includes(event?.outcome)
    && !event.is_replay && !["rejected", "deleted"].includes(event.review_status);
  const mapAttacker = (fighterId, mode) => mode === "received"
    ? (fighterId === "fighter_a" ? "fighter_b" : "fighter_a") : fighterId;
  const zoneOpacity = (target, total) => number(target?.landed) > 0 && total > 0
    ? clamp(.24 + .48 * number(target.landed) / total, .24, .72) : 0;
  const canonicalContact = (event) => {
    const point = event?.target_point_norm;
    // Earlier runs project video bounding boxes into this field. A coordinate
    // name alone is not proof that the location belongs to the canonical body.
    if (event?.target_point_space !== "defender_front_canonical_v1"
      || event?.target_point_source !== "canonical_contact_v1"
      || number(event?.target_point_confidence) < .8
      || !point || !Number.isFinite(point.x) || !Number.isFinite(point.y)
      || point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) return null;
    return {
      x: point.x, y: point.y, precise: true,
      uncertainty: clamp(number(event.target_uncertainty_radius, .06), .025, .28),
    };
  };
  const crossedMapEvents = (events, previousMs, currentMs, seeking = false) => {
    if (seeking || previousMs === null || currentMs <= previousMs || currentMs - previousMs > 1000) return [];
    return events.filter((event) => isMapEvent(event) && number(event.peak_ms) > previousMs && number(event.peak_ms) <= currentMs)
      .sort((a, b) => number(a.peak_ms) - number(b.peak_ms));
  };
  const pulseProgress = (pulse, timestampMs, reducedMotion = false) => {
    if (!pulse || reducedMotion) return 1;
    return clamp((timestampMs - pulse.startMs) / 600, 0, 1);
  };
  const impactEnvelope = (event, progress, reducedMotion = false) => {
    const intensity = clamp(number(event?.impact_proxy_0_100, 45) / 100, 0, 1);
    const phase = clamp(number(progress, 1), 0, 1);
    // A fast, soft attack followed by a longer release. No camera shake or
    // particles: intensity is a relative score, not measured physical force.
    const active = !reducedMotion && phase > 0 && phase < 1;
    const energy = active ? Math.pow(Math.sin(Math.PI * Math.pow(phase, .55)), 2) : 0;
    return { intensity, phase, energy, fill: .24 + energy * (.24 + intensity * .2), outline: .42 + energy * .5 };
  };
  window.BoxingVisionState = Object.freeze({ atlasGeometry, canonicalContact, crossedMapEvents, pulseProgress, zoneOpacity, mapAttacker, impactEnvelope });

  const formatTime = (milliseconds) => {
    const safe = Math.max(0, Math.round(number(milliseconds)));
    const minutes = Math.floor(safe / 60000);
    const seconds = Math.floor((safe % 60000) / 1000);
    const millis = safe % 1000;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
  };

  const techniqueCode = (technique) => TECHNIQUE_LABELS[technique] || String(technique || "тип удара не определён");
  const targetCode = (target) => (target === "head" ? "в голову" : target === "body" ? "в корпус" : "");
  const eventTitle = (event) => {
    const target = targetCode(event.target);
    if (!event.technique || event.technique === "unknown") {
      const hand = event.hand === "left" ? "левой" : event.hand === "right" ? "правой" : "неопределённой рукой";
      return `Удар ${hand} ${target}`.trim();
    }
    const hand = event.hand === "left" ? "Левый" : event.hand === "right" ? "Правый" : "";
    return `${hand} ${techniqueCode(event.technique)} ${target}`.trim();
  };
  const eventQualifier = (event) => {
    const qualifiers = [];
    if (!event.technique || event.technique === "unknown") qualifiers.push("Тип удара не определён");
    if (!event.target || event.target === "unknown") qualifiers.push("Зона не определена");
    return qualifiers.join(" · ");
  };
  const outcomeCode = (outcome) => OUTCOME_LABELS[outcome] || "Исход не определён";
  const eventDisplayTitle = (event) => eventTitle(event);
  const eventMetricLine = (event) => {
    const qualifier = eventQualifier(event);
    const outcome = outcomeCode(event.outcome);
    const confidence = `уверенность ${Math.round(number(event.confidence) * 100)}%`;
    const impact = event.impact_proxy_0_100 === null || event.impact_proxy_0_100 === undefined
      ? ""
      : ` · интенсивность ${Math.round(number(event.impact_proxy_0_100))}`;
    return `${qualifier ? `${qualifier} · ` : ""}${outcome} · ${confidence}${impact}`;
  };
  const REVIEW_LABELS = {
    unreviewed: "На проверку",
    confirmed: "Подтверждено",
    rejected: "Отклонено",
    real_not_replay: "Не повтор",
  };
  const TARGET_LABELS = {
    head: "Голова",
    body: "Корпус",
    unknown: "Зона не определена",
  };

  const initials = (name) => {
    const parts = String(name || "BV").trim().split(/\s+/u).filter(Boolean).slice(0, 2);
    return parts.map((part) => Array.from(part)[0] || "").join("").toUpperCase() || "BV";
  };

  const Workspace = {
    payload: null,
    payloadText: "",
    root: null,
    shell: null,
    video: null,
    canvas: null,
    scroller: null,
    tooltip: null,
    context: null,
    dpr: 1,
    hitRegions: [],
    renderTimer: 0,
    dragPlayhead: false,
    playIntervalEndMs: null,
    boundVideo: null,
    lastPanelTime: -1,
    bodyPulseFrame: 0,
    bodyPulses: {},
    lastPlaybackMs: null,
    reducedMotion: false,
    panelViews: new Map(),
    initializedDemos: new Set(),
    state: {
      currentTimeMs: 0,
      selectedEventId: null,
      filters: {
        fighter: "all",
        round: "all",
        outcome: "all",
        target: "all",
        technique: "all",
        hand: "all",
        confidence: "0",
        review: "all",
      },
      zoom: 1,
      metricScope: "to_time",
      bodyMapMode: { fighter_a: "dealt", fighter_b: "dealt" },
      bodyMapDisplay: { fighter_a: "latest", fighter_b: "latest" },
      mobileFighter: "fighter_a",
    },

    hydrate() {
      const nodes = Array.from(document.querySelectorAll("[data-bv-payload]"));
      const payloadNode = nodes[nodes.length - 1];
      if (!payloadNode) return;
      const text = payloadNode.textContent || "";
      let payload;
      try {
        payload = JSON.parse(text);
      } catch (_error) {
        return;
      }
      const shell = payloadNode.closest(".bv-workspace-chrome");
      if (!shell) return;
      const changed = text !== this.payloadText || shell !== this.shell;
      this.payloadText = text;
      this.payload = payload;
      if (changed) this.panelViews.clear();
      this.shell = shell;
      this.root = document.querySelector("#bv-result-workspace") || shell.parentElement;
      this.canvas = shell.querySelector("[data-bv-timeline]");
      this.scroller = shell.querySelector("[data-bv-canvas-scroller]");
      this.tooltip = shell.querySelector("[data-bv-tooltip]");
      const needsPanelHydrate = ["fighter_a", "fighter_b"].some((fighterId) => {
        const mounts = Array.from(document.querySelectorAll(`.bv-fighter-panel[data-fighter-id="${fighterId}"]`));
        const mount = mounts[mounts.length - 1];
        return !mount || Boolean(mount.querySelector(".bv-panel-skeleton"));
      });
      if (!changed && !needsPanelHydrate) {
        this.bindVideo();
        return;
      }
      if (this.state.selectedEventId && !this.eventById(this.state.selectedEventId)) {
        this.state.selectedEventId = null;
      }
      this.bindShell();
      if (!this.motionPreference && window.matchMedia) {
        this.motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
        this.reducedMotion = this.motionPreference.matches;
        this.motionPreference.addEventListener?.("change", (event) => {
          this.reducedMotion = event.matches;
          this.startBodyPulse();
        });
      }
      this.bindVideo();
      this.populateFilters();
      this.applyMobilePanel();
      this.renderPanels(true);
      this.resizeCanvas(true);
      this.renderInspector();
      this.renderA11yListbox();
      this.renderVideoOverlay();
      if (changed) this.announce("Интерактивный анализ готов");
    },

    bindShell() {
      if (!this.shell || this.shell.dataset.bvBound === "true") return;
      this.shell.dataset.bvBound = "true";

      this.shell.querySelectorAll("[data-bv-scope]").forEach((control) => {
        control.addEventListener("click", () => {
          this.state.metricScope = control.dataset.bvScope || "to_time";
          this.pressGroup("[data-bv-scope]", control);
          this.renderPanels(true);
          this.updateRangeReadout();
        });
      });
      this.shell.querySelectorAll("[data-bv-zoom]").forEach((control) => {
        control.addEventListener("click", () => this.setZoom(number(control.dataset.bvZoom, 1)));
      });
      this.shell.querySelectorAll("[data-bv-mobile-fighter]").forEach((control) => {
        control.addEventListener("click", () => {
          this.state.mobileFighter = control.dataset.bvMobileFighter || "fighter_a";
          this.pressGroup("[data-bv-mobile-fighter]", control);
          this.applyMobilePanel();
        });
      });
      this.shell.querySelectorAll("[data-bv-filter]").forEach((control) => {
        const key = control.dataset.bvFilter;
        if (key && key in this.state.filters) control.value = this.state.filters[key];
        control.addEventListener("change", () => {
          if (key) this.state.filters[key] = control.value;
          this.resizeCanvas(false);
          this.renderA11yListbox();
        });
      });
      this.shell.querySelectorAll("[data-bv-review]").forEach((control) => {
        control.addEventListener("click", () => {
          const menu = control.closest("[data-bv-review-menu]");
          if (menu) { menu.open = false; menu.querySelector("summary")?.focus(); }
          this.sendReview(control.dataset.bvReview);
        });
      });
      const reviewMenu = this.shell.querySelector("[data-bv-review-menu]");
      reviewMenu?.addEventListener("toggle", () => {
        if (!reviewMenu.open) return;
        const anchor = reviewMenu.querySelector("summary").getBoundingClientRect();
        const actions = reviewMenu.querySelector(".bv-review-popover");
        actions.style.left = `${Math.max(8, Math.min(window.innerWidth - 224, anchor.right - 216))}px`;
        actions.style.top = `${Math.max(8, anchor.top - actions.offsetHeight - 8)}px`;
      });
      reviewMenu?.addEventListener("keydown", (event) => {
        if (event.key !== "Escape" || !reviewMenu.open) return;
        event.preventDefault();
        event.stopPropagation();
        reviewMenu.open = false;
        reviewMenu.querySelector("summary")?.focus();
      });
      this.root?.addEventListener("pointerdown", (event) => {
        if (reviewMenu?.open && !reviewMenu.contains(event.target)) reviewMenu.open = false;
      });
      this.shell.querySelectorAll("[data-bv-event-nav]").forEach((control) => {
        control.addEventListener("click", () => this.navigateEvent(number(control.dataset.bvEventNav, 1)));
      });
      this.shell.querySelector("[data-bv-theater]")?.addEventListener("click", () => this.toggleTheater());
      document.addEventListener("fullscreenchange", () => this.syncTheaterButton());
      document.addEventListener("webkitfullscreenchange", () => this.syncTheaterButton());
      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && this.isTheaterActive()) this.toggleTheater();
      });

      if (this.canvas) {
        this.canvas.addEventListener("pointerdown", (event) => this.onPointerDown(event));
        this.canvas.addEventListener("pointermove", (event) => this.onPointerMove(event));
        this.canvas.addEventListener("pointerleave", () => this.hideTooltip());
        this.canvas.addEventListener("dblclick", (event) => this.onDoubleClick(event));
        this.canvas.addEventListener("keydown", (event) => this.onKeyDown(event));
      }
      if (this.scroller) {
        this.scroller.addEventListener("scroll", () => this.hideTooltip(), { passive: true });
        this.scroller.addEventListener("wheel", (event) => this.onWheel(event), { passive: false });
      }
      window.addEventListener("pointerup", () => { this.dragPlayhead = false; });
      if (window.ResizeObserver && this.scroller) {
        this.resizeObserver = new ResizeObserver(() => this.resizeCanvas(false));
        this.resizeObserver.observe(this.scroller);
      }
      this.syncTheaterButton();
      this.ensureVideoOverlay();
    },

    isTheaterActive() {
      return this.root?.dataset?.bvTheaterActive === "true"
        || Boolean(this.root && (document.fullscreenElement || document.webkitFullscreenElement) === this.root);
    },

    toggleTheater(requestNativeFullscreen = true) {
      const target = this.root || document.querySelector("#bv-result-workspace");
      if (!target) return;
      const activeElement = document.fullscreenElement || document.webkitFullscreenElement;
      const request = target.requestFullscreen || target.webkitRequestFullscreen;
      const exit = document.exitFullscreen || document.webkitExitFullscreen;
      if (this.isTheaterActive()) {
        delete target.dataset.bvTheaterActive;
        target.classList.remove("bv-theater-mode");
        document.body.style.overflow = this.previousBodyOverflow || "";
        if (activeElement === target && exit) Promise.resolve(exit.call(document)).catch(() => {});
        this.syncTheaterButton();
        return;
      }
      this.previousBodyOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      // Svelte owns the Gradio group's class attribute. A separate data flag
      // survives its class reconciliation during unrelated control updates.
      target.dataset.bvTheaterActive = "true";
      target.classList.add("bv-theater-mode");
      this.syncTheaterButton();
      if (requestNativeFullscreen && request && activeElement !== target) Promise.resolve(request.call(target)).catch(() => {});
    },

    syncTheaterButton() {
      const control = this.shell?.querySelector("[data-bv-theater]");
      if (!control) return;
      const active = this.isTheaterActive();
      control.textContent = active ? "Выйти из полного экрана" : "На весь экран";
      control.setAttribute("aria-pressed", active ? "true" : "false");
      if (this.video) {
        if (active) {
          if (this.video.dataset.bvHadControls === undefined) {
            this.video.dataset.bvHadControls = this.video.controls ? "true" : "false";
          }
          this.video.controls = false;
        } else if (this.video.dataset.bvHadControls !== undefined) {
          this.video.controls = this.video.dataset.bvHadControls === "true";
          delete this.video.dataset.bvHadControls;
        }
      }
      window.setTimeout(() => this.resizeCanvas(true), 80);
      this.renderVideoOverlay();
    },

    bindVideo() {
      const video = document.querySelector("#annotated-video video, #result-video-card video");
      if (!video) {
        window.setTimeout(() => this.bindVideo(), 220);
        return;
      }
      this.video = video;
      if (this.boundVideo === video) {
        this.applyInitialDemoState();
        return;
      }
      this.boundVideo = video;
      video.addEventListener("loadedmetadata", () => {
        if (!number(this.payload?.duration_ms) && Number.isFinite(video.duration)) {
          this.payload.duration_ms = Math.round(video.duration * 1000);
        }
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          this.root?.style.setProperty("--bv-video-aspect", `${video.videoWidth} / ${video.videoHeight}`);
        }
        this.applyInitialDemoState();
        this.syncFromVideo(true);
        this.resizeCanvas(true);
        this.renderVideoOverlay();
      });
      video.addEventListener("timeupdate", () => this.syncFromVideo(false));
      video.addEventListener("seeking", () => {
        this.lastPlaybackMs = null;
        this.bodyPulses = {};
      });
      video.addEventListener("seeked", () => {
        this.lastPlaybackMs = Math.round(video.currentTime * 1000);
        this.syncFromVideo(true);
        this.startBodyPulse();
      });
      video.addEventListener("play", () => {
        this.lastPlaybackMs = Math.round(video.currentTime * 1000);
        this.drawTimeline();
        this.startBodyPulse();
      });
      video.addEventListener("pause", () => { this.drawTimeline(); this.drawBodyMaps(); });
      video.addEventListener("ratechange", () => this.renderVideoOverlay());
      video.addEventListener("ended", () => { this.playIntervalEndMs = null; });
      this.applyInitialDemoState();
      this.syncFromVideo(true);
      this.ensureVideoOverlay();
      this.syncTheaterButton();
    },

    isReadOnlyDemo() {
      return this.payload?.metadata?.demo_read_only === true;
    },

    applyInitialDemoState() {
      const metadata = this.payload?.metadata;
      const video = this.video;
      if (metadata?.bundled_demo !== true || !this.isReadOnlyDemo()
        || !video || video.readyState < 1 || !Number.isFinite(video.duration) || video.duration <= 0) return false;
      const key = `${metadata.demo_start_ms}:${metadata.demo_selected_event_id || ""}:${this.payload.duration_ms}`;
      if (this.initializedDemos.has(key)) return false;
      const target = clamp(number(metadata.demo_start_ms), 0, Math.max(0, video.duration * 1000 - 1));
      video.pause();
      // Mark before seeking: time/metadata events can synchronously rehydrate
      // Gradio's workspace. A resize, rebind or reviewer selection is not a
      // second first launch and must never rewind the user's playhead.
      this.initializedDemos.add(key);
      this.playIntervalEndMs = null;
      this.lastPlaybackMs = null;
      this.bodyPulses = {};
      this.state.selectedEventId = this.eventById(metadata.demo_selected_event_id)?.event_id || null;
      this.state.currentTimeMs = target;
      video.currentTime = target / 1000;
      if (metadata.desktop_theater === true && !this.isTheaterActive()) {
        // Fill the native content surface without asking WebView2 for a
        // user-gesture-only browser fullscreen permission.
        this.toggleTheater(false);
      }
      this.renderInspector();
      this.renderPanels(true);
      this.renderA11yListbox();
      return true;
    },

    syncFromVideo(forcePanels) {
      if (!this.video) return;
      this.state.currentTimeMs = Math.round(number(this.video.currentTime) * 1000);
      this.updatePlaybackPulses();
      if (this.playIntervalEndMs !== null && this.state.currentTimeMs >= this.playIntervalEndMs) {
        this.video.pause();
        this.playIntervalEndMs = null;
      }
      const readout = this.shell?.querySelector("[data-bv-time]");
      if (readout) readout.textContent = formatTime(this.state.currentTimeMs);
      if (forcePanels || Math.abs(this.state.currentTimeMs - this.lastPanelTime) >= 120) {
        this.renderPanels(false);
        this.lastPanelTime = this.state.currentTimeMs;
      }
      this.renderVideoOverlay();
      this.scheduleTimeline();
    },

    pressGroup(selector, pressed) {
      this.shell?.querySelectorAll(selector).forEach((node) => {
        node.setAttribute("aria-pressed", node === pressed ? "true" : "false");
      });
    },

    populateFilters() {
      if (!this.payload || !this.shell) return;
      const events = this.payload.events || [];
      const rounds = Array.from(new Set(events.map((event) => number(event.round, 1)))).sort((a, b) => a - b);
      const techniques = Array.from(new Set(events.map((event) => String(event.technique || "unknown")))).sort();
      this.fillSelect("round", rounds.map((value) => [String(value), `Раунд ${value}`]));
      this.fillSelect("technique", techniques.map((value) => [value, techniqueCode(value)]));
    },

    fillSelect(key, options) {
      const select = this.shell?.querySelector(`[data-bv-filter="${key}"]`);
      if (!select || select.dataset.bvPopulated === "true") return;
      options.forEach(([value, label]) => {
        const option = element("option", "", label);
        option.value = value;
        select.append(option);
      });
      select.dataset.bvPopulated = "true";
      select.value = this.state.filters[key] || "all";
    },

    currentRound() {
      const metadata = this.payload?.metadata || {};
      // Playback and event times are relative to the normalized, trimmed video.
      const startMs = number(metadata.timeline_origin_ms);
      const roundMs = number(metadata.round_length_s) * 1000;
      const restMs = number(metadata.rest_length_s) * 1000;
      const scheduled = Math.max(1, number(metadata.scheduled_rounds, 1));
      if (roundMs > 0 && this.state.currentTimeMs >= startMs) {
        const cycle = roundMs + restMs;
        return clamp(Math.floor((this.state.currentTimeMs - startMs) / Math.max(1, cycle)) + 1, 1, scheduled);
      }
      const prior = (this.payload?.events || []).filter((event) => number(event.peak_ms) <= this.state.currentTimeMs);
      return prior.length ? number(prior[prior.length - 1].round, 1) : 1;
    },

    countableEvents() {
      return (this.payload?.events || []).filter((event) => !event.is_replay && !["rejected", "deleted"].includes(event.review_status));
    },

    scopedEvents() {
      const events = this.countableEvents();
      if (this.state.metricScope === "to_time") {
        return events.filter((event) => number(event.peak_ms) <= this.state.currentTimeMs);
      }
      if (this.state.metricScope === "round") {
        const round = this.currentRound();
        return events.filter((event) => number(event.round, 1) === round);
      }
      return events;
    },

    aggregate(fighterId, mode, sourceEvents) {
      const identityKey = mode === "received" ? "defender_id" : "attacker_id";
      const selected = sourceEvents.filter((event) => event[identityKey] === fighterId);
      const stats = {
        attempts: selected.length,
        likely_landed: 0,
        blocked: 0,
        missed: 0,
        unclear: 0,
        accuracy: 0,
        average_impact_proxy: 0,
        targets: {
          head: { landed: 0, thrown: 0 },
          body: { landed: 0, thrown: 0 },
          unknown: { landed: 0, thrown: 0 },
        },
      };
      let impact = 0;
      let impactCount = 0;
      selected.forEach((event) => {
        const outcome = OUTCOME_LABELS[event.outcome] ? event.outcome : "unclear";
        const target = ["head", "body"].includes(event.target) ? event.target : "unknown";
        stats[outcome] += 1;
        stats.targets[target].thrown += 1;
        if (outcome === "likely_landed") stats.targets[target].landed += 1;
        if (event.impact_proxy_0_100 !== null && event.impact_proxy_0_100 !== undefined) {
          impact += number(event.impact_proxy_0_100);
          impactCount += 1;
        }
      });
      const classified = stats.likely_landed + stats.blocked + stats.missed;
      stats.accuracy = classified ? stats.likely_landed / classified : 0;
      stats.average_impact_proxy = impactCount ? impact / impactCount : 0;
      return stats;
    },

    renderPanels(force) {
      if (!this.payload) return;
      const events = this.scopedEvents();
      ["fighter_a", "fighter_b"].forEach((fighterId) => {
        const mounts = Array.from(document.querySelectorAll(`.bv-fighter-panel[data-fighter-id="${fighterId}"]`));
        const mount = mounts[mounts.length - 1];
        if (!mount) return;
        const mapMode = this.state.bodyMapMode?.[fighterId] || "dealt";
        const mapDisplay = this.state.bodyMapDisplay?.[fighterId] || "latest";
        const renderKey = `${this.state.metricScope}:${Math.floor(this.state.currentTimeMs / 120)}:${mapMode}:${mapDisplay}:${this.state.selectedEventId || ""}`;
        if (!force && mount.dataset.bvTimeBucket === renderKey) return;
        mount.dataset.bvTimeBucket = renderKey;
        this.renderFighterPanel(mount, fighterId, events);
      });
    },

    renderFighterPanel(mount, fighterId, events) {
      const previous = this.panelViews.get(fighterId);
      if (previous?.mount === mount && previous.head.isConnected) {
        this.updateFighterPanel(previous, fighterId, events);
        return;
      }
      const profile = this.payload.fighters?.[fighterId] || { name: fighterId };
      const stats = this.aggregate(fighterId, "dealt", events);
      const mapMode = this.state.bodyMapMode?.[fighterId] || "dealt";
      const mapDisplay = this.state.bodyMapDisplay?.[fighterId] || "latest";
      const mapStats = this.aggregate(fighterId, mapMode, events);
      const hasEvents = (this.payload.events || []).length > 0;
      const totals = (!hasEvents && this.state.metricScope === "fight") ? profile.summary_totals : stats;
      mount.replaceChildren();

      const head = element("div", "bv-fighter-head");
      const avatar = element("div", "bv-avatar");
      if (profile.portrait_url) {
        const image = element("img");
        image.src = profile.portrait_url;
        image.alt = "";
        if (profile.demo_portrait) {
          image.alt = profile.portrait_label;
          avatar.title = `${profile.portrait_label}. ${profile.portrait_attribution}`;
        }
        avatar.append(image);
      } else {
        avatar.textContent = fighterId === "fighter_a" ? "A" : "B";
      }
      const identity = element("div", "bv-fighter-identity");
      identity.append(element("span", "bv-corner-label", fighterId === "fighter_a" ? "Красный угол" : "Синий угол"));
      identity.append(element("strong", "bv-fighter-name", profile.name));
      if (profile.record) identity.append(element("span", "bv-fighter-record", `Рекорд · ${profile.record}`));
      head.append(avatar, identity, element("span", "bv-corner-token", fighterId === "fighter_a" ? "A" : "B"));

      const headlines = element("div", "bv-headline-metrics bv-metrics-grid");
      const ratio = element("div", "bv-headline-metric");
      const ratioNumber = element("strong");
      ratioNumber.append(document.createTextNode(String(totals?.likely_landed || 0)), element("i", "", `/${totals?.attempts || 0}`));
      ratio.append(ratioNumber, element("small", "", "Попадания / попытки"));
      const accuracy = element("div", "bv-headline-metric");
      accuracy.append(element("strong", "", `${Math.round(number(totals?.accuracy) * 100)}%`), element("small", "", "Точность"));
      headlines.append(ratio, accuracy);

      const secondary = element("div", "bv-rail-secondary");
      [
        ["Интенсивность", Math.round(number(totals?.average_impact_proxy))],
        ["Блокировано", totals?.blocked],
        ["Промахи", totals?.missed],
        ["Не определено", totals?.unclear],
      ].forEach(([label, value]) => {
        const metric = element("div", "bv-rail-stat");
        metric.append(element("strong", "", value || 0), element("small", "", label));
        secondary.append(metric);
      });

      const mapSection = element("section", "bv-map-section");
      const mapHeader = element("div", "bv-section-row");
      const mapTitle = element("div", "bv-map-title");
      mapTitle.append(
        element("span", "bv-section-kicker", "Распределение ударов"),
        element("small", "bv-map-mode-note", mapMode === "dealt" ? "Атаки бойца" : "Атаки соперника"),
      );
      mapHeader.append(mapTitle);
      const mapControls = element("div", "bv-map-controls");
      const modeControl = element("div", "bv-mode-control");
      [["dealt", "Нанёс"], ["received", "Пропустил"]].forEach(([mode, label]) => {
        const control = button("", label, {
          "aria-pressed": mapMode === mode,
          "data-bv-body-mode": mode,
          "aria-label": `${label}, боец ${fighterId === "fighter_a" ? "A" : "B"}`,
        });
        control.addEventListener("click", () => {
          this.state.bodyMapMode = { ...(this.state.bodyMapMode || {}), [fighterId]: mode };
          this.renderPanels(true);
        });
        modeControl.append(control);
      });
      const displayControl = element("div", "bv-mode-control bv-map-display-control");
      [["latest", "Последний удар"], ["heatmap", "Тепловая карта"]].forEach(([mode, label]) => {
        const control = button("", label, {
          "aria-pressed": mapDisplay === mode,
          "data-bv-map-display": mode,
          "aria-label": `${label}, боец ${fighterId === "fighter_a" ? "A" : "B"}`,
        });
        control.addEventListener("click", () => {
          this.state.bodyMapDisplay = { ...(this.state.bodyMapDisplay || {}), [fighterId]: mode };
          this.renderPanels(true);
        });
        displayControl.append(control);
      });
      mapControls.append(modeControl, displayControl);
      mapHeader.append(mapControls);
      const bodyMap = element("div", "bv-body-map");
      const headStat = this.zoneStat("Голова", mapStats.targets.head);
      const bodyStat = this.zoneStat("Корпус", mapStats.targets.body);
      const figure = element("div", "bv-body-figure");
      const mapAccentFighter = mapMode === "received"
        ? (fighterId === "fighter_a" ? "fighter_b" : "fighter_a")
        : fighterId;
      figure.style.setProperty("--bv-map-accent", COLORS[mapAccentFighter]);
      const base = element("img", "bv-body-base");
      base.src = this.payload.assets?.body_map_base || this.payload.assets?.body_map || "";
      base.alt = "Силуэт боксёра с зонами головы и корпуса";
      const landedTotal = number(mapStats.targets.head.landed) + number(mapStats.targets.body.landed);
      const headZone = element("span", "bv-body-zone bv-body-zone-head");
      headZone.setAttribute("aria-hidden", "true");
      headZone.style.setProperty("--bv-zone-mask", `url("${this.payload.assets?.body_map_head_mask || base.src}")`);
      headZone.style.setProperty("--bv-zone-opacity", String(zoneOpacity(mapStats.targets.head, landedTotal)));
      const bodyZone = element("span", "bv-body-zone bv-body-zone-body");
      bodyZone.setAttribute("aria-hidden", "true");
      bodyZone.style.setProperty("--bv-zone-mask", `url("${this.payload.assets?.body_map_body_mask || base.src}")`);
      bodyZone.style.setProperty("--bv-zone-opacity", String(zoneOpacity(mapStats.targets.body, landedTotal)));
      const hitCanvas = element("canvas", "bv-body-hit-canvas");
      hitCanvas.width = 512;
      hitCanvas.height = 768;
      hitCanvas.setAttribute("aria-hidden", "true");
      const stage = element("div", "bv-body-atlas-stage");
      stage.append(base, headZone, bodyZone, hitCanvas);
      figure.append(stage);
      const mapCaption = element("div", "bv-map-caption");
      mapCaption.hidden = true;
      figure.append(mapCaption);
      this.applyAtlasCrop(figure, stage);
      bodyMap.append(headStat, figure, bodyStat);
      const unknown = element("div", "bv-unknown-target");
      unknown.append(element("span", "", "Зона не распознана"), element("b", "", `${mapStats.targets.unknown.landed}/${mapStats.targets.unknown.thrown}`));
      unknown.hidden = number(mapStats.targets.unknown.thrown) === 0;
      mapSection.append(mapHeader, bodyMap, unknown);
      const mapEvents = events.slice();
      const selectedForMap = this.eventById(this.state.selectedEventId);
      if (selectedForMap && !mapEvents.some((event) => event.event_id === selectedForMap.event_id)) {
        mapEvents.push(selectedForMap);
      }
      mapEvents.sort((a, b) => number(a.peak_ms) - number(b.peak_ms));
      this.drawBodyMap(hitCanvas, fighterId, mapMode, mapDisplay, mapEvents);

      const feed = element("section", "bv-event-feed");
      const feedHeader = element("div", "bv-section-row");
      feedHeader.append(element("span", "bv-section-kicker", "Последние удары"), element("span", "bv-corner-label", `До ${formatTime(this.state.currentTimeMs)}`));
      feed.append(feedHeader);
      const list = element("div", "bv-event-feed-list");
      const recent = (this.payload.events || [])
        .filter((event) => event.attacker_id === fighterId && number(event.peak_ms) <= this.state.currentTimeMs && !event.is_replay && !["rejected", "deleted"].includes(event.review_status))
        .slice(-7)
        .reverse();
      if (!recent.length) {
        list.append(element("div", "bv-empty-feed", "До текущей позиции подтверждённых кандидатов нет. Перемотайте видео или выберите событие на таймлайне."));
      } else {
        recent.forEach((event) => list.append(this.eventRow(event)));
      }
      feed.append(list);
      mount.append(head, headlines, secondary, mapSection, feed);
      const view = { mount, head, avatar, identity, ratioNumber, accuracy, secondary,
        figure, stage, base, headZone, bodyZone, hitCanvas, headStat, bodyStat,
        modeControl, displayControl, mapTitle, unknown, feedHeader, list };
      this.panelViews.set(fighterId, view);
      this.updateFighterPanel(view, fighterId, events);
    },

    applyAtlasCrop(figure, stage) {
      const crop = atlasGeometry(this.payload?.assets?.body_map_atlas);
      figure.style.setProperty("--bv-figure-aspect", String(crop.aspect));
      stage.style.width = `${100 / crop.width}%`;
      stage.style.height = `${100 / crop.height}%`;
      stage.style.left = `${-100 * crop.x / crop.width}%`;
      stage.style.top = `${-100 * crop.y / crop.height}%`;
    },

    updateFighterPanel(view, fighterId, events) {
      const scrollTop = view.list.scrollTop;
      const setText = (node, value) => { if (node && node.textContent !== String(value)) node.textContent = String(value); };
      const mode = this.state.bodyMapMode[fighterId] || "dealt";
      const display = this.state.bodyMapDisplay[fighterId] || "latest";
      const stats = this.aggregate(fighterId, "dealt", events);
      const mapStats = this.aggregate(fighterId, mode, events);
      const totals = !(this.payload.events || []).length && this.state.metricScope === "fight"
        ? (this.payload.fighters?.[fighterId]?.summary_totals || stats) : stats;
      setText(view.ratioNumber.firstChild, String(totals?.likely_landed || 0));
      setText(view.ratioNumber.querySelector("i"), `/${totals?.attempts || 0}`);
      setText(view.accuracy.querySelector("strong"), `${Math.round(number(totals?.accuracy) * 100)}%`);
      const values = [Math.round(number(totals?.average_impact_proxy)), totals?.blocked, totals?.missed, totals?.unclear];
      view.secondary.querySelectorAll("strong").forEach((node, index) => setText(node, values[index] || 0));
      setText(view.headStat.querySelector("b"), `${mapStats.targets.head.landed}/${mapStats.targets.head.thrown}`);
      setText(view.bodyStat.querySelector("b"), `${mapStats.targets.body.landed}/${mapStats.targets.body.thrown}`);
      setText(view.mapTitle.querySelector("small"), mode === "dealt" ? "Атаки бойца" : "Атаки соперника");
      view.modeControl.querySelectorAll("button").forEach((node) => node.setAttribute("aria-pressed", String(node.dataset.bvBodyMode === mode)));
      view.displayControl.querySelectorAll("button").forEach((node) => node.setAttribute("aria-pressed", String(node.dataset.bvMapDisplay === display)));
      view.figure.style.setProperty("--bv-map-accent", COLORS[mapAttacker(fighterId, mode)]);
      view.headZone.dataset.baseOpacity = String(zoneOpacity(mapStats.targets.head, mapStats.targets.head.landed + mapStats.targets.body.landed));
      view.bodyZone.dataset.baseOpacity = String(zoneOpacity(mapStats.targets.body, mapStats.targets.head.landed + mapStats.targets.body.landed));
      setText(view.unknown.querySelector("b"), `${mapStats.targets.unknown.landed}/${mapStats.targets.unknown.thrown}`);
      view.unknown.hidden = number(mapStats.targets.unknown.thrown) === 0;
      setText(view.feedHeader.querySelector(".bv-corner-label"), `До ${formatTime(this.state.currentTimeMs)}`);
      // Selection is a preview, not a contribution to cumulative statistics.
      this.drawBodyMap(view.hitCanvas, fighterId, mode, display, events);
      const recent = (this.payload.events || []).filter((event) => event.attacker_id === fighterId
        && number(event.peak_ms) <= this.state.currentTimeMs && !event.is_replay
        && !["rejected", "deleted"].includes(event.review_status)).slice(-7).reverse();
      const wanted = new Set(recent.map((event) => event.event_id));
      Array.from(view.list.children).forEach((node) => {
        if (!recent.length && node.classList.contains("bv-empty-feed")) return;
        if (!wanted.has(node.dataset.eventId)) node.remove();
      });
      recent.forEach((event, index) => {
        let row = Array.from(view.list.children).find((node) => node.dataset.eventId === event.event_id);
        if (!row) row = this.eventRow(event);
        row.setAttribute("aria-current", String(this.state.selectedEventId === event.event_id));
        if (view.list.children[index] !== row) view.list.insertBefore(row, view.list.children[index] || null);
      });
      if (!recent.length && !view.list.querySelector(".bv-empty-feed")) {
        view.list.append(element("div", "bv-empty-feed", "На этом отрезке ударов пока нет."));
      }
      view.list.scrollTop = scrollTop;
    },

    zoneStat(label, target) {
      const node = element("div", "bv-zone-stat");
      node.append(element("em", "", label), element("b", "", `${target.landed}/${target.thrown}`), element("span", "", "Попадания / попытки"));
      return node;
    },

    bodyMapEvents(fighterId, mode, events) {
      const identityKey = mode === "received" ? "defender_id" : "attacker_id";
      return events.filter((event) => (
        event[identityKey] === fighterId
        && ["head", "body"].includes(event.target)
        && ["likely_landed", "blocked"].includes(event.outcome)
        && !event.is_replay
        && !["rejected", "deleted"].includes(event.review_status)
      ));
    },

    contactPoint(event) {
      return canonicalContact(event);
    },

    zoneFxLayer(canvas, target, color) {
      // Derive the luminous contour from the registered raster mask itself.
      // Never approximate body anatomy with a different polygon or clip-path.
      if (typeof Image === "undefined") return null;
      const url = this.payload.assets?.[target === "head" ? "body_map_head_mask" : "body_map_body_mask"];
      if (!url) return null;
      canvas.__bvMasks ||= new Map();
      let entry = canvas.__bvMasks.get(url);
      if (!entry) {
        const image = new Image();
        entry = { image, colors: new Map() };
        canvas.__bvMasks.set(url, entry);
        image.onload = () => { if (canvas.isConnected) this.drawBodyMaps(); };
        image.src = url;
      }
      if (!entry.image.complete || !entry.image.naturalWidth) return null;
      if (entry.colors.has(color)) return entry.colors.get(color);
      const make = () => {
        const layer = document.createElement("canvas");
        layer.width = canvas.width; layer.height = canvas.height;
        return layer;
      };
      const fill = make();
      const paint = fill.getContext("2d");
      paint.drawImage(entry.image, 0, 0, fill.width, fill.height);
      paint.globalCompositeOperation = "source-in";
      paint.fillStyle = color;
      paint.fillRect(0, 0, fill.width, fill.height);
      const outline = make();
      const edge = outline.getContext("2d");
      [[-2, 0], [2, 0], [0, -2], [0, 2], [-1.5, -1.5], [1.5, 1.5]].forEach(([x, y]) => edge.drawImage(fill, x, y));
      edge.globalCompositeOperation = "destination-out";
      edge.drawImage(entry.image, 0, 0, fill.width, fill.height);
      const layer = { fill, outline, sweep: make() };
      entry.colors.set(color, layer);
      return layer;
    },

    drawZoneFx(ctx, canvas, event, color, envelope) {
      const layers = this.zoneFxLayer(canvas, event.target, color);
      if (!layers) return;
      ctx.save();
      ctx.globalAlpha = envelope.outline;
      ctx.shadowColor = color;
      ctx.shadowBlur = 5 + envelope.energy * 14;
      ctx.drawImage(layers.outline, 0, 0);
      ctx.shadowBlur = 0;
      if (envelope.energy > .01) {
        // A traveling sheen, clipped to the entire known zone. This indicates
        // regional evidence without inventing an exact contact location.
        const sweep = layers.sweep.getContext("2d");
        const w = canvas.width, h = canvas.height;
        sweep.clearRect(0, 0, w, h);
        sweep.globalCompositeOperation = "source-over";
        sweep.drawImage(layers.fill, 0, 0);
        sweep.globalCompositeOperation = "source-in";
        const range = event.target === "head" ? [.025, .15] : [.14, .39];
        const center = h * (range[0] + (range[1] - range[0]) * envelope.phase);
        const band = h * .035;
        const shine = sweep.createLinearGradient(0, center - band, 0, center + band);
        shine.addColorStop(0, "rgba(245,247,250,0)");
        shine.addColorStop(.5, "rgba(245,247,250,.35)");
        shine.addColorStop(1, "rgba(245,247,250,0)");
        sweep.fillStyle = shine;
        sweep.fillRect(0, 0, w, h);
        ctx.globalAlpha = envelope.energy * .7;
        ctx.drawImage(layers.sweep, 0, 0);
      }
      ctx.restore();
    },

    drawBodyMap(canvas, fighterId, mode, display, sourceEvents) {
      if (!canvas) return;
      canvas.__bvFighterId = fighterId;
      canvas.__bvMode = mode;
      canvas.__bvDisplay = display;
      canvas.__bvEvents = sourceEvents;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const width = canvas.width || 512;
      const height = canvas.height || 768;
      ctx.clearRect(0, 0, width, height);
      const candidates = this.bodyMapEvents(fighterId, mode, sourceEvents);
      const selected = this.eventById(this.state.selectedEventId);
      const identityKey = mode === "received" ? "defender_id" : "attacker_id";
      const belongs = (event) => event?.[identityKey] === fighterId && !event.is_replay
        && !["rejected", "deleted"].includes(event.review_status);
      const selectedBelongs = selected && (!this.video || this.video.paused)
        && belongs(selected);
      const accentFighter = mapAttacker(fighterId, mode);
      const pulse = this.bodyPulses[accentFighter];
      const playbackEvent = pulse && this.eventById(pulse.eventId);
      const now = this.video ? this.video.currentTime * 1000 : this.state.currentTimeMs;
      const recent = sourceEvents.filter((event) => belongs(event) && number(event.peak_ms) <= now);
      const pulseActive = pulse && now >= pulse.startMs && pulseProgress(pulse, now) < 1 && belongs(playbackEvent);
      const latest = selectedBelongs ? selected : pulseActive ? playbackEvent : recent[recent.length - 1];
      // A subsequent miss/unclear event clears the old contact highlight.
      const current = isMapEvent(latest) ? latest : null;
      const progress = pulse && current?.event_id === pulse.eventId ? pulseProgress(pulse, now, this.reducedMotion) : 1;
      const accent = COLORS[accentFighter] || COLORS[fighterId];
      const stage = canvas.parentElement;
      const envelope = impactEnvelope(current, progress, this.reducedMotion);
      const effectColor = current?.outcome === "blocked" ? COLORS.warning : accent;
      const caption = stage?.parentElement?.querySelector(".bv-map-caption");
      if (caption) {
        caption.hidden = display !== "latest" || !current;
        caption.dataset.outcome = current?.outcome || "";
        caption.textContent = current ? `${TARGET_LABELS[current.target]} · ${current.outcome === "blocked" ? "Блок" : "Вероятное попадание"}` : "";
        caption.title = current && this.contactPoint(current) ? "Локализованная точка контакта" : "Подсвечена целевая зона. Точная точка контакта не определена.";
      }
      ["head", "body"].forEach((target) => {
        const zone = stage?.querySelector(`.bv-body-zone-${target}`);
        if (!zone) return;
        const isCurrent = display === "latest" && current?.target === target;
        const opacity = display === "heatmap" ? number(zone.dataset.baseOpacity)
          : isCurrent ? envelope.fill : 0;
        zone.style.setProperty("--bv-zone-opacity", String(opacity));
        zone.style.backgroundColor = isCurrent ? effectColor : accent;
      });
      const drawHit = (event, strength, phase = 1) => {
        const point = this.contactPoint(event);
        if (!point) return;
        const x = point.x * width;
        const y = point.y * height;
        const impact = clamp(number(event.impact_proxy_0_100, 45) / 100, .18, 1);
        const uncertainty = point.uncertainty * Math.min(width, height);
        const radius = clamp(uncertainty * .58 + impact * 17, 20, 62);
        const eased = 1 - Math.pow(1 - phase, 3);
        const pulseRadius = radius * (1 + eased * .9);
        const glowColor = event.outcome === "blocked" ? COLORS.warning : accent;
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, pulseRadius);
        gradient.addColorStop(0, event.outcome === "likely_landed" ? "rgba(152,255,72,.84)" : "rgba(244,193,93,.72)");
        gradient.addColorStop(.32, event.outcome === "likely_landed" ? "rgba(152,255,72,.24)" : "rgba(244,193,93,.18)");
        gradient.addColorStop(1, "rgba(0,0,0,0)");
        ctx.save();
        ctx.globalAlpha = strength * (phase < 1 ? .8 : .25);
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, pulseRadius, 0, Math.PI * 2);
        ctx.fill();
        if (display === "latest") {
          ctx.globalAlpha = strength * (phase < 1 ? 1 - eased * .55 : .5);
          ctx.strokeStyle = glowColor;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.arc(x, y, Math.max(12, radius * .44 + eased * 14), 0, Math.PI * 2);
          ctx.stroke();
          if (phase < 1 && !this.reducedMotion) {
            ctx.globalAlpha = strength * (1 - eased) * .6;
            ctx.beginPath();
            ctx.arc(x, y, radius * (.65 + eased), 0, Math.PI * 2);
            ctx.stroke();
          }
        }
        if (point.precise && event.outcome === "likely_landed") {
          ctx.globalAlpha = strength;
          ctx.fillStyle = COLORS.lime;
          ctx.beginPath();
          ctx.arc(x, y, 6 + impact * 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "#f5f7fa";
          ctx.lineWidth = 3;
          ctx.stroke();
        }
        ctx.restore();
      };
      if (display === "heatmap") {
        candidates.slice(-120).forEach((event) => drawHit(event, .22 + number(event.confidence, .5) * .26));
      } else if (current) {
        this.drawZoneFx(ctx, canvas, current, effectColor, envelope);
        drawHit(current, 1, progress);
      }
    },

    updatePlaybackPulses() {
      if (!this.video) return false;
      const now = Math.round(this.video.currentTime * 1000);
      const crossed = crossedMapEvents(this.payload?.events || [], this.lastPlaybackMs, now, this.video.seeking);
      this.lastPlaybackMs = now;
      crossed.forEach((event) => {
        this.bodyPulses[event.attacker_id] = { eventId: event.event_id, startMs: number(event.peak_ms) };
      });
      return crossed.length > 0;
    },

    drawBodyMaps() {
      this.panelViews.forEach((view) => {
        const canvas = view.hitCanvas;
        this.drawBodyMap(canvas, canvas.__bvFighterId, canvas.__bvMode, canvas.__bvDisplay, canvas.__bvEvents || []);
      });
    },

    startBodyPulse(event = null) {
      if (event && isMapEvent(event)) {
        this.bodyPulses[event.attacker_id] = { eventId: event.event_id, startMs: number(event.peak_ms) };
      }
      window.cancelAnimationFrame(this.bodyPulseFrame);
      this.drawBodyMaps();
      if (!this.video || this.video.paused || this.reducedMotion) return;
      const tick = () => {
        if (!this.video || this.video.paused || this.video.seeking || this.reducedMotion) return;
        this.updatePlaybackPulses();
        this.drawBodyMaps();
        this.bodyPulseFrame = window.requestAnimationFrame(tick);
      };
      this.bodyPulseFrame = window.requestAnimationFrame(tick);
    },

    eventRow(event) {
      const row = button("bv-event-row", "", {
        "aria-label": this.describeEvent(event),
        "aria-current": this.state.selectedEventId === event.event_id,
        "data-outcome": event.outcome || "unclear",
        "data-event-id": event.event_id,
      });
      row.append(
        element("span", "bv-event-time", formatTime(event.peak_ms)),
        element("span", "bv-event-technique", eventDisplayTitle(event)),
        element(
          "span",
          `bv-event-outcome ${event.outcome === "likely_landed" ? "is-landed" : ""}`,
          outcomeCode(event.outcome),
        ),
      );
      row.addEventListener("click", () => this.selectEvent(event, true));
      return row;
    },

    applyMobilePanel() {
      const panelA = document.querySelector("#bv-panel-a-card");
      const panelB = document.querySelector("#bv-panel-b-card");
      panelA?.classList.toggle("bv-mobile-hidden", this.state.mobileFighter !== "fighter_a");
      panelB?.classList.toggle("bv-mobile-hidden", this.state.mobileFighter !== "fighter_b");
      if (this.root) this.root.dataset.bvMobileFighter = this.state.mobileFighter;
    },

    ensureVideoOverlay() {
      const videoCard = document.querySelector("#result-video-card");
      if (!videoCard) return null;
      let overlay = videoCard.querySelector(".bv-video-ui");
      if (overlay) return overlay;
      overlay = element("div", "bv-video-ui");
      const bar = element("div", "bv-video-topbar");
      const status = element("span", "bv-video-status", "Анализ боя");
      const round = element("span", "bv-video-round", "Раунд 1 / 1");
      const time = element("span", "bv-video-time", "00:00.000 / 00:00.000");
      const rate = element("span", "bv-video-rate", "1.0×");
      const transport = element("button", "bv-video-transport", "Играть");
      transport.type = "button";
      transport.addEventListener("click", () => {
        if (this.video?.paused) this.video.play().catch(() => {});
        else this.video?.pause();
      });
      const theater = element("button", "bv-video-theater", "Экран");
      theater.type = "button";
      theater.setAttribute("aria-label", "Переключить полный экран видео и аналитики");
      theater.addEventListener("click", () => this.toggleTheater());
      bar.append(status, round, time, rate, transport, theater);
      // Event details belong to the persistent inspector and the MP4 export,
      // never on top of the live fight area in the analysis workspace.
      overlay.append(bar);
      videoCard.append(overlay);
      return overlay;
    },

    activeVideoEvent() {
      const selected = this.eventById(this.state.selectedEventId);
      const now = this.state.currentTimeMs;
      if (
        selected
        && now >= number(selected.start_ms) - 500
        && now <= number(selected.end_ms) + 1200
      ) return selected;
      return [...(this.payload?.events || [])].reverse().find((event) => (
        !event.is_replay
        && !["rejected", "deleted"].includes(event.review_status)
        && now >= number(event.start_ms) - 180
        && now <= number(event.end_ms) + 900
      )) || null;
    },

    renderVideoOverlay() {
      const overlay = this.ensureVideoOverlay();
      if (!overlay || !this.payload) return;
      const metadata = this.payload.metadata || {};
      const scheduled = Math.max(1, number(metadata.scheduled_rounds, 1));
      const videoRate = this.video ? number(this.video.playbackRate, 1) : 1;
      const roundNode = overlay.querySelector(".bv-video-round");
      const timeNode = overlay.querySelector(".bv-video-time");
      const rateNode = overlay.querySelector(".bv-video-rate");
      if (roundNode) roundNode.textContent = `Раунд ${this.currentRound()} / ${scheduled}`;
      if (timeNode) timeNode.textContent = `${formatTime(this.state.currentTimeMs)} / ${formatTime(this.payload.duration_ms)}`;
      if (rateNode) rateNode.textContent = `${videoRate.toFixed(1)}×`;
      const transport = overlay.querySelector(".bv-video-transport");
      if (transport) transport.textContent = this.video?.paused ? "Играть" : "Пауза";
      const theater = overlay.querySelector(".bv-video-theater");
      if (theater) theater.textContent = this.isTheaterActive() ? "Свернуть" : "Экран";
    },

    renderA11yListbox() {
      const listbox = this.shell?.querySelector("[data-bv-a11y-listbox]");
      if (!listbox) return;
      const events = this.filteredTimelineEvents().slice().sort((a, b) => number(a.peak_ms) - number(b.peak_ms));
      listbox.replaceChildren();
      events.forEach((event, index) => {
        const option = element("div", "bv-a11y-event-option", this.describeEvent(event));
        option.id = `bv-event-option-${index}`;
        option.setAttribute("role", "option");
        option.setAttribute("aria-selected", this.state.selectedEventId === event.event_id ? "true" : "false");
        option.dataset.eventId = event.event_id;
        listbox.append(option);
        if (this.state.selectedEventId === event.event_id) {
          this.canvas?.setAttribute("aria-activedescendant", option.id);
        }
      });
      if (!this.state.selectedEventId) this.canvas?.removeAttribute("aria-activedescendant");
    },

    navigateEvent(direction) {
      const events = this.filteredTimelineEvents().slice().sort((a, b) => number(a.peak_ms) - number(b.peak_ms));
      if (!events.length) return;
      const currentIndex = events.findIndex((event) => event.event_id === this.state.selectedEventId);
      const baseIndex = currentIndex < 0
        ? (direction < 0 ? events.length : -1)
        : currentIndex;
      const next = events[clamp(baseIndex + (direction < 0 ? -1 : 1), 0, events.length - 1)];
      if (next) this.selectEvent(next, true);
    },

    filteredTimelineEvents() {
      const filters = this.state.filters;
      return (this.payload?.events || []).filter((event) => {
        if (filters.fighter !== "all" && event.attacker_id !== filters.fighter) return false;
        if (filters.round !== "all" && String(event.round) !== filters.round) return false;
        if (filters.outcome !== "all" && event.outcome !== filters.outcome) return false;
        if (filters.target !== "all" && event.target !== filters.target) return false;
        if (filters.technique !== "all" && event.technique !== filters.technique) return false;
        if (filters.hand !== "all" && event.hand !== filters.hand) return false;
        if (number(event.confidence) < number(filters.confidence)) return false;
        if (filters.review === "replay" && !event.is_replay) return false;
        if (!["all", "replay"].includes(filters.review) && event.review_status !== filters.review) return false;
        return true;
      });
    },

    hasReviewEvents() {
      return this.filteredTimelineEvents().some((event) => (
        event.is_replay || !["confirmed", "rejected", "deleted"].includes(event.review_status)
      ));
    },

    syncCenterLayout() {
      // Gradio 5.50's CSS scoper drops CSSContainerRule. Use the existing
      // ResizeObserver and flat selectors, including when browser zoom changes.
      const center = this.root?.querySelector?.("#bv-center-stack");
      if (!center || !this.root?.dataset) return;
      const width = center.getBoundingClientRect().width;
      if (!(width > 0)) return;
      const layout = width <= 479 ? "narrow" : width <= 680 ? "stacked" : width <= 860 ? "compact" : "wide";
      if (this.root.dataset.bvCenterLayout !== layout) this.root.dataset.bvCenterLayout = layout;
    },

    resizeCanvas(resetScroll) {
      this.syncCenterLayout();
      if (!this.canvas || !this.scroller) return;
      // FIT means the actual viewport, including narrow windows/browser zoom.
      // A hidden mount can briefly measure zero; ResizeObserver fixes it later.
      const baseWidth = Math.max(1, number(this.scroller.clientWidth, 1));
      const cssWidth = Math.round(baseWidth * this.state.zoom);
      const hasReview = this.hasReviewEvents();
      const cssHeight = hasReview ? 150 : 116;
      this.shell?.classList.toggle("bv-no-review", !hasReview);
      this.dpr = clamp(window.devicePixelRatio || 1, 1, 2);
      this.canvas.style.width = `${cssWidth}px`;
      this.canvas.style.height = `${cssHeight}px`;
      this.canvas.width = Math.round(cssWidth * this.dpr);
      this.canvas.height = Math.round(cssHeight * this.dpr);
      this.context = this.canvas.getContext("2d");
      this.context?.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      if (resetScroll) this.centerPlayhead();
      this.drawTimeline();
      const inspector = this.shell?.querySelector(".bv-inspector");
      if (inspector) {
        const theater = this.isTheaterActive();
        inspector.style.maxHeight = theater && window.innerWidth >= 1180
          ? `${Math.max(100, window.innerHeight - inspector.getBoundingClientRect().top)}px` : "";
      }
    },

    setZoom(zoom, anchorClientX = null) {
      const next = [1, 2, 4, 8].reduce((best, value) => Math.abs(value - zoom) < Math.abs(best - zoom) ? value : best, 1);
      if (!this.scroller || !this.canvas || next === this.state.zoom) return;
      const oldWidth = this.canvas.getBoundingClientRect().width || this.scroller.clientWidth;
      const anchor = anchorClientX === null ? this.scroller.clientWidth / 2 : anchorClientX - this.scroller.getBoundingClientRect().left;
      const timeRatio = (this.scroller.scrollLeft + anchor) / Math.max(1, oldWidth);
      this.state.zoom = next;
      this.shell?.querySelectorAll("[data-bv-zoom]").forEach((control) => {
        control.setAttribute("aria-pressed", number(control.dataset.bvZoom) === next ? "true" : "false");
      });
      this.resizeCanvas(false);
      const newWidth = this.canvas.getBoundingClientRect().width;
      this.scroller.scrollLeft = timeRatio * newWidth - anchor;
      this.updateRangeReadout();
    },

    centerPlayhead() {
      if (!this.scroller || !this.canvas || this.state.zoom === 1) return;
      const ratio = this.state.currentTimeMs / Math.max(1, number(this.payload?.duration_ms));
      this.scroller.scrollLeft = ratio * this.canvas.getBoundingClientRect().width - this.scroller.clientWidth / 2;
    },

    scheduleTimeline() {
      if (this.renderTimer) return;
      this.renderTimer = window.requestAnimationFrame(() => {
        this.renderTimer = 0;
        this.drawTimeline();
      });
    },

    roundSegments() {
      const duration = Math.max(1, number(this.payload?.duration_ms));
      const metadata = this.payload?.metadata || {};
      const roundLength = number(metadata.round_length_s) * 1000;
      const rest = number(metadata.rest_length_s) * 1000;
      const start = number(metadata.timeline_origin_ms);
      const scheduled = Math.max(1, number(metadata.scheduled_rounds, 1));
      if (roundLength > 0) {
        return Array.from({ length: scheduled }, (_, index) => ({
          round: index + 1,
          start_ms: clamp(start + index * (roundLength + rest), 0, duration),
          end_ms: clamp(start + index * (roundLength + rest) + roundLength, 0, duration),
        })).filter((segment) => segment.end_ms > segment.start_ms);
      }
      const grouped = new Map();
      (this.payload?.events || []).forEach((event) => {
        const round = number(event.round, 1);
        const item = grouped.get(round) || { round, start_ms: duration, end_ms: 0 };
        item.start_ms = Math.min(item.start_ms, number(event.start_ms));
        item.end_ms = Math.max(item.end_ms, number(event.end_ms));
        grouped.set(round, item);
      });
      const segments = Array.from(grouped.values()).sort((a, b) => a.round - b.round);
      return segments.length ? segments : [{ round: 1, start_ms: 0, end_ms: duration }];
    },

    drawTimeline() {
      if (!this.canvas || !this.context || !this.payload) return;
      const ctx = this.context;
      const width = this.canvas.width / this.dpr;
      const height = this.canvas.height / this.dpr;
      const duration = Math.max(1, number(this.payload.duration_ms));
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#07090b";
      ctx.fillRect(0, 0, width, height);
      const hasReview = this.hasReviewEvents();
      this.timelineHasReview = hasReview;
      [34, 72, 110].filter((y) => hasReview || y < 110).forEach((y) => {
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, y + .5);
        ctx.lineTo(width, y + .5);
        ctx.stroke();
      });

      this.roundSegments().forEach((segment, index) => {
        const x1 = segment.start_ms / duration * width;
        const x2 = segment.end_ms / duration * width;
        ctx.fillStyle = index % 2 ? "#11151a" : "#0d1115";
        ctx.fillRect(x1, 0, Math.max(1, x2 - x1), 33);
        ctx.strokeStyle = COLORS.grid;
        ctx.strokeRect(x1 + .5, .5, Math.max(0, x2 - x1 - 1), 32);
      });

      const visibleTicks = clamp(Math.round(width / 150), 4, 10);
      ctx.font = '400 12px "BV SF Pro", "SF Pro Text", -apple-system, sans-serif';
      ctx.textAlign = "center";
      for (let index = 0; index <= visibleTicks; index += 1) {
        const x = index / visibleTicks * width;
        const time = index / visibleTicks * duration;
        ctx.fillStyle = "#c2c8d0";
        ctx.textAlign = index === 0 ? "left" : index === visibleTicks ? "right" : "center";
        ctx.fillText(formatTime(time).slice(0, 5), x, 31);
      }
      ctx.textAlign = "left";

      const events = this.filteredTimelineEvents();
      const pixelsPerMs = width / duration;
      const clustered = this.clusterEvents(events, pixelsPerMs);
      this.hitRegions = [];
      clustered.forEach((cluster) => this.drawCluster(ctx, cluster, width, duration));

      const playheadX = clamp(this.state.currentTimeMs / duration * width, 0, width);
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(playheadX + .5, 0);
      ctx.lineTo(playheadX + .5, height);
      ctx.stroke();
      ctx.fillStyle = COLORS.text;
      ctx.beginPath();
      ctx.moveTo(playheadX - 4, 0);
      ctx.lineTo(playheadX + 4, 0);
      ctx.lineTo(playheadX, 6);
      ctx.closePath();
      ctx.fill();
    },

    clusterEvents(events, pixelsPerMs) {
      const shouldCluster = this.state.zoom <= 2 && pixelsPerMs < .02;
      if (!shouldCluster) return events.map((event) => ({ events: [event], lane: event.attacker_id }));
      const groups = [];
      ["fighter_a", "fighter_b"].forEach((fighterId) => {
        const laneEvents = events.filter((event) => event.attacker_id === fighterId).sort((a, b) => a.peak_ms - b.peak_ms);
        laneEvents.forEach((event) => {
          const previous = groups[groups.length - 1];
          if (previous && previous.lane === fighterId && (event.peak_ms - previous.events[previous.events.length - 1].peak_ms) * pixelsPerMs < 9) {
            previous.events.push(event);
          } else {
            groups.push({ lane: fighterId, events: [event] });
          }
        });
      });
      return groups.sort((a, b) => a.events[0].peak_ms - b.events[0].peak_ms);
    },

    drawCluster(ctx, cluster, width, duration) {
      const events = cluster.events;
      const event = events[Math.floor(events.length / 2)];
      const laneTop = cluster.lane === "fighter_a" ? 41 : 79;
      const color = COLORS[cluster.lane] || COLORS.muted;
      const start = Math.min(...events.map((item) => number(item.start_ms)));
      const end = Math.max(...events.map((item) => number(item.end_ms)));
      let x1 = start / duration * width;
      let x2 = end / duration * width;
      if (events.length > 1) {
        const center = number(event.peak_ms) / duration * width;
        x1 = center - 9;
        x2 = center + 9;
      }
      const eventWidth = Math.max(4, x2 - x1);
      const y = laneTop;
      const h = 24;
      ctx.save();
      ctx.lineWidth = this.state.selectedEventId && events.some((item) => item.event_id === this.state.selectedEventId) ? 2 : 1;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      const outcome = event.outcome;
      if (events.length > 1 || outcome === "likely_landed") {
        ctx.globalAlpha = events.length > 1 ? .88 : .62;
        ctx.fillRect(x1, y, eventWidth, h);
        ctx.globalAlpha = 1;
        ctx.strokeRect(x1 + .5, y + .5, Math.max(0, eventWidth - 1), h - 1);
      } else if (outcome === "blocked") {
        ctx.globalAlpha = .2;
        ctx.fillRect(x1, y, eventWidth, h);
        ctx.globalAlpha = 1;
        ctx.strokeRect(x1 + .5, y + .5, Math.max(0, eventWidth - 1), h - 1);
        ctx.beginPath();
        ctx.rect(x1, y, eventWidth, h);
        ctx.clip();
        for (let offset = -h; offset < eventWidth + h; offset += 5) {
          ctx.beginPath();
          ctx.moveTo(x1 + offset, y + h);
          ctx.lineTo(x1 + offset + h, y);
          ctx.stroke();
        }
      } else {
        if (outcome === "unclear") ctx.setLineDash([3, 3]);
        ctx.strokeRect(x1 + .5, y + .5, Math.max(0, eventWidth - 1), h - 1);
      }
      ctx.setLineDash([]);
      const peakX = number(event.peak_ms) / duration * width;
      ctx.strokeStyle = outcome === "likely_landed" ? "#050608" : color;
      ctx.beginPath();
      ctx.moveTo(peakX + .5, y + 3);
      ctx.lineTo(peakX + .5, y + h - 3);
      ctx.stroke();
      if (events.length > 1) {
        ctx.fillStyle = "#050608";
        ctx.font = '600 12px "BV SF Pro", "SF Pro Text", -apple-system, sans-serif';
        ctx.textAlign = "center";
        ctx.fillText(String(events.length), (x1 + x2) / 2, y + 16);
        ctx.textAlign = "left";
      }
      ctx.restore();

      const hitPadding = Math.max(3, (24 - eventWidth) / 2);
      this.hitRegions.push({ x1: x1 - hitPadding, x2: x1 + eventWidth + hitPadding, y1: y - 3, y2: y + h + 3, events });
      const needsReview = events.some((item) => item.is_replay || !["confirmed", "rejected", "deleted"].includes(item.review_status));
      if (needsReview && this.timelineHasReview) {
        const reviewX = number(event.peak_ms) / duration * width;
        ctx.fillStyle = event.is_replay ? COLORS.warning : color;
        ctx.beginPath();
        ctx.moveTo(reviewX, 119);
        ctx.lineTo(reviewX + 5, 128);
        ctx.lineTo(reviewX, 137);
        ctx.lineTo(reviewX - 5, 128);
        ctx.closePath();
        ctx.fill();
      }
    },

    canvasPoint(event) {
      const rect = this.canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    },

    hitAt(point) {
      return this.hitRegions.filter((region) => point.x >= region.x1 && point.x <= region.x2 && point.y >= region.y1 && point.y <= region.y2)
        .sort((a, b) => Math.abs(point.x - (a.x1 + a.x2) / 2) - Math.abs(point.x - (b.x1 + b.x2) / 2))[0] || null;
    },

    onPointerDown(event) {
      if (!this.canvas) return;
      const point = this.canvasPoint(event);
      const hit = this.hitAt(point);
      if (hit) {
        const nearest = hit.events.reduce((best, item) => Math.abs(item.peak_ms - this.xToTime(point.x)) < Math.abs(best.peak_ms - this.xToTime(point.x)) ? item : best, hit.events[0]);
        this.selectEvent(nearest, true);
        return;
      }
      this.dragPlayhead = true;
      this.seek(this.xToTime(point.x), false);
      this.canvas.setPointerCapture?.(event.pointerId);
    },

    onPointerMove(event) {
      if (!this.canvas) return;
      const point = this.canvasPoint(event);
      if (this.dragPlayhead && event.buttons) {
        this.seek(this.xToTime(point.x), false);
        return;
      }
      const hit = this.hitAt(point);
      if (hit) this.showTooltip(hit.events[0], event.clientX, event.clientY, hit.events.length);
      else this.hideTooltip();
    },

    onDoubleClick(event) {
      const hit = this.hitAt(this.canvasPoint(event));
      if (!hit || !this.video) return;
      const targetTime = this.xToTime(this.canvasPoint(event).x);
      const selected = hit.events.reduce((best, item) => (
        Math.abs(number(item.peak_ms) - targetTime) < Math.abs(number(best.peak_ms) - targetTime)
          ? item
          : best
      ), hit.events[0]);
      this.selectEvent(selected, false);
      this.video.currentTime = number(selected.start_ms) / 1000;
      this.playIntervalEndMs = number(selected.end_ms);
      this.video.play().catch(() => {});
    },

    onWheel(event) {
      if (!this.scroller) return;
      if (event.metaKey || event.ctrlKey) {
        event.preventDefault();
        const direction = event.deltaY > 0 ? -1 : 1;
        const levels = [1, 2, 4, 8];
        const index = levels.indexOf(this.state.zoom);
        this.setZoom(levels[clamp(index + direction, 0, levels.length - 1)], event.clientX);
      } else if (Math.abs(event.deltaY) > Math.abs(event.deltaX)) {
        // FIT has no horizontal overflow. Let the page scroll to the inspector
        // and fighter panels instead of trapping the wheel over the timeline.
        // At either zoomed edge, hand vertical scrolling back to the page too.
        const maximum = Math.max(0, this.scroller.scrollWidth - this.scroller.clientWidth);
        const current = clamp(this.scroller.scrollLeft, 0, maximum);
        const unit = event.deltaMode === 1 ? 20 : event.deltaMode === 2 ? this.scroller.clientWidth : 1;
        const next = clamp(current + event.deltaY * unit, 0, maximum);
        if (Math.abs(next - current) > .5) {
          event.preventDefault();
          this.scroller.scrollLeft = next;
        }
      }
    },

    xToTime(x) {
      const width = this.canvas?.getBoundingClientRect().width || 1;
      return clamp(x / width * number(this.payload?.duration_ms), 0, number(this.payload?.duration_ms));
    },

    seek(milliseconds, pause = true) {
      const target = clamp(number(milliseconds), 0, number(this.payload?.duration_ms));
      this.state.currentTimeMs = target;
      if (this.video) {
        if (pause) this.video.pause();
        this.video.currentTime = target / 1000;
      }
      this.syncFromVideo(true);
    },

    selectEvent(event, seekVideo) {
      this.state.selectedEventId = event.event_id;
      if (seekVideo) this.seek(Math.max(0, number(event.peak_ms) - 350), true);
      this.renderInspector();
      this.renderPanels(true);
      this.renderA11yListbox();
      this.renderVideoOverlay();
      this.startBodyPulse(event);
      this.drawTimeline();
      this.announce(this.describeEvent(event));
    },

    eventById(eventId) {
      return (this.payload?.events || []).find((event) => event.event_id === eventId) || null;
    },

    describeEvent(event) {
      const impact = event.impact_proxy_0_100 === null || event.impact_proxy_0_100 === undefined
        ? ""
        : `, интенсивность ${Math.round(number(event.impact_proxy_0_100))}`;
      return `${formatTime(event.peak_ms)}, боец ${event.attacker_id === "fighter_a" ? "A" : "B"}, ${eventTitle(event)}, ${outcomeCode(event.outcome)}, уверенность ${Math.round(number(event.confidence) * 100)} процентов${impact}`;
    },

    renderInspector() {
      const inspector = this.shell?.querySelector("[data-bv-inspector]");
      if (!inspector) return;
      const event = this.eventById(this.state.selectedEventId);
      inspector.dataset.empty = event ? "false" : "true";
      const title = inspector.querySelector("[data-bv-inspector-title]");
      const detail = inspector.querySelector("[data-bv-inspector-detail]");
      const time = inspector.querySelector("[data-bv-inspector-time]");
      const fighter = inspector.querySelector("[data-bv-inspector-fighter]");
      const round = inspector.querySelector("[data-bv-inspector-round]");
      const review = inspector.querySelector("[data-bv-inspector-review]");
      const reviewMenu = inspector.querySelector("[data-bv-review-menu]");
      if (reviewMenu) {
        reviewMenu.open = false;
        reviewMenu.hidden = this.isReadOnlyDemo();
      }
      const thumbnail = inspector.querySelector("[data-bv-inspector-thumbnail]");
      const controls = inspector.querySelectorAll("button");
      if (!event) {
        if (time) time.textContent = "Выберите событие";
        if (title) title.textContent = "Нажмите на маркер удара в таймлайне";
        if (detail) detail.textContent = "Здесь появятся исход, уверенность и интенсивность.";
        [fighter, round, review].forEach((node) => { if (node) node.textContent = "—"; });
        if (thumbnail) thumbnail.hidden = true;
        controls.forEach((control) => { control.disabled = true; });
        return;
      }
      controls.forEach((control) => { control.disabled = this.isReadOnlyDemo() && control.hasAttribute("data-bv-review"); });
      if (time) time.textContent = formatTime(event.peak_ms);
      if (title) title.textContent = eventDisplayTitle(event);
      if (detail) {
        const impact = event.impact_proxy_0_100 === null || event.impact_proxy_0_100 === undefined
          ? "" : ` · интенсивность ${Math.round(number(event.impact_proxy_0_100))}`;
        detail.replaceChildren(
          element("span", "bv-inspector-outcome", outcomeCode(event.outcome)),
          element("span", "bv-inspector-measures", ` · уверенность ${Math.round(number(event.confidence) * 100)}%${impact}`),
        );
        detail.setAttribute("aria-label", eventMetricLine(event));
        detail.dataset.outcome = event.outcome;
      }
      if (fighter) fighter.textContent = event.attacker_id === "fighter_a" ? "Боксёр A" : "Боксёр B";
      if (round) round.textContent = `Раунд ${number(event.round, 1)}`;
      if (review) review.textContent = `${REVIEW_LABELS[event.review_status] || "На проверку"}${event.is_replay ? " · возможный повтор" : ""}`;
      const note = inspector.querySelector("[data-bv-inspector-note]");
      if (note) note.textContent = this.isReadOnlyDemo() ? "Демо-разбор · только просмотр" : eventQualifier(event) || "Сверьте событие с видео перед подтверждением.";
      if (thumbnail) {
        thumbnail.replaceChildren();
        const preview = this.previewFor(number(event.peak_ms));
        if (event.preview_url) {
          const image = element("img");
          image.src = event.preview_url;
          image.alt = "";
          thumbnail.append(image);
          thumbnail.hidden = false;
        } else if (preview) {
          thumbnail.style.backgroundImage = `url("${String(preview.url).replace(/["\\\n\r]/g, "")}")`;
          thumbnail.style.aspectRatio = `${preview.tile_width} / ${preview.tile_height}`;
          thumbnail.style.backgroundSize = `${100 * preview.sheet_width / preview.tile_width}% ${100 * preview.sheet_height / preview.tile_height}%`;
          const x = preview.sheet_width > preview.tile_width ? 100 * preview.x / (preview.sheet_width - preview.tile_width) : 0;
          const y = preview.sheet_height > preview.tile_height ? 100 * preview.y / (preview.sheet_height - preview.tile_height) : 0;
          thumbnail.style.backgroundPosition = `${x}% ${y}%`;
          thumbnail.hidden = false;
        } else {
          thumbnail.removeAttribute("style");
          thumbnail.hidden = true;
        }
      }
      const sorted = this.filteredTimelineEvents().slice().sort((a, b) => number(a.peak_ms) - number(b.peak_ms));
      const selectedIndex = sorted.findIndex((item) => item.event_id === event.event_id);
      inspector.querySelector('[data-bv-event-nav="-1"]')?.toggleAttribute("disabled", selectedIndex <= 0);
      inspector.querySelector('[data-bv-event-nav="1"]')?.toggleAttribute("disabled", selectedIndex < 0 || selectedIndex >= sorted.length - 1);
    },

    showTooltip(event, clientX, clientY, clusterSize) {
      if (!this.tooltip) return;
      this.tooltip.replaceChildren();
      const preview = this.previewFor(number(event.peak_ms));
      if (preview) {
        const frame = element("div", "bv-tooltip-preview");
        frame.style.width = `${preview.tile_width}px`;
        frame.style.height = `${preview.tile_height}px`;
        frame.style.backgroundImage = `url("${String(preview.url).replace(/["\\\n\r]/g, "")}")`;
        frame.style.backgroundSize = `${preview.sheet_width}px ${preview.sheet_height}px`;
        frame.style.backgroundPosition = `${-preview.x}px ${-preview.y}px`;
        this.tooltip.append(frame);
      }
      this.tooltip.append(
        element("small", "bv-tooltip-time", `${formatTime(event.peak_ms)} · Раунд ${number(event.round, 1)}${clusterSize > 1 ? ` · ${clusterSize} событий` : ""}`),
        element("strong", "bv-tooltip-title", eventTitle(event)),
        element("div", "bv-tooltip-outcome", outcomeCode(event.outcome)),
        element("div", "bv-tooltip-metrics", `Уверенность ${Math.round(number(event.confidence) * 100)}%${event.impact_proxy_0_100 === null || event.impact_proxy_0_100 === undefined ? "" : ` · интенсивность ${Math.round(number(event.impact_proxy_0_100))}`}`),
        element("small", "bv-tooltip-review", `${event.is_replay ? "Возможный повтор · " : ""}${REVIEW_LABELS[event.review_status] || "На проверку"}`),
      );
      this.tooltip.dataset.outcome = event.outcome;
      this.tooltip.hidden = false;
      const left = clamp(clientX + 14, 8, window.innerWidth - this.tooltip.offsetWidth - 8);
      const top = clamp(clientY + 14, 8, window.innerHeight - this.tooltip.offsetHeight - 8);
      this.tooltip.style.left = `${left}px`;
      this.tooltip.style.top = `${top}px`;
    },

    previewFor(timeMs) {
      const manifest = this.payload?.preview_manifest;
      if (!manifest || !Array.isArray(manifest.frames) || !Array.isArray(manifest.sheets) || !manifest.frames.length) return null;
      let low = 0;
      let high = manifest.frames.length - 1;
      while (low < high) {
        const middle = Math.floor((low + high + 1) / 2);
        if (number(manifest.frames[middle].time_ms) <= timeMs) low = middle;
        else high = middle - 1;
      }
      const frame = manifest.frames[low];
      const url = manifest.sheets[number(frame.sheet, -1)];
      if (!url) return null;
      return {
        ...frame,
        url,
        tile_width: number(manifest.tile_width, 160),
        tile_height: number(manifest.tile_height, 90),
        sheet_width: number(manifest.sheet_width, 160),
        sheet_height: number(manifest.sheet_height, 90),
      };
    },

    hideTooltip() {
      if (this.tooltip) this.tooltip.hidden = true;
    },

    sendReview(action) {
      if (this.isReadOnlyDemo()) {
        this.announce("Демо-разбор доступен только для просмотра. Нажмите «Новый анализ», чтобы загрузить свой бой.");
        return;
      }
      const event = this.eventById(this.state.selectedEventId);
      if (!event || !["confirmed", "rejected", "real_not_replay"].includes(action)) return;
      const textarea = document.querySelector("#bv-review-command textarea, #bv-review-command input");
      const submit = document.querySelector("#bv-review-submit button");
      if (!textarea || !submit) {
        this.announce("Review bridge недоступен");
        return;
      }
      const descriptor = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(textarea), "value");
      const value = JSON.stringify({ event_id: event.event_id, action });
      if (descriptor?.set) descriptor.set.call(textarea, value);
      else textarea.value = value;
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new Event("change", { bubbles: true }));
      submit.click();
      this.announce(`Review отправлен: ${event.event_id}`);
    },

    onKeyDown(event) {
      if (!this.video) return;
      const frameMs = 1000 / Math.max(1, number(this.payload?.metadata?.output_fps, 30));
      if (event.code === "Space") {
        event.preventDefault();
        if (this.video.paused) this.video.play().catch(() => {});
        else this.video.pause();
      } else if (event.key === "k" || event.key === "K") {
        event.preventDefault();
        this.video.pause();
      } else if (event.key === "j" || event.key === "J") {
        event.preventDefault();
        this.seek(this.state.currentTimeMs - 5000, true);
      } else if (event.key === "l" || event.key === "L") {
        event.preventDefault();
        this.video.playbackRate = this.video.paused ? 1 : (this.video.playbackRate >= 2 ? 1 : this.video.playbackRate + .5);
        this.video.play().catch(() => {});
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        const direction = event.key === "ArrowLeft" ? -1 : 1;
        this.seek(this.state.currentTimeMs + direction * (event.shiftKey ? 1000 : frameMs), true);
      } else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
        event.preventDefault();
        this.navigateEvent(event.key === "ArrowUp" ? -1 : 1);
      } else if (event.key === "Home" || event.key === "End") {
        event.preventDefault();
        const events = this.filteredTimelineEvents().slice().sort((a, b) => number(a.peak_ms) - number(b.peak_ms));
        const selected = event.key === "Home" ? events[0] : events[events.length - 1];
        if (selected) this.selectEvent(selected, true);
      }
    },

    updateRangeReadout() {
      const label = this.shell?.querySelector("[data-bv-range]");
      if (!label) return;
      const scope = this.state.metricScope === "to_time" ? "до момента" : this.state.metricScope === "round" ? `раунд ${this.currentRound()}` : "весь бой";
      label.textContent = `${this.state.zoom === 1 ? "По ширине" : `${this.state.zoom}×`} · ${scope}`;
    },

    announce(message) {
      const live = this.shell?.querySelector("[data-bv-live]");
      if (live) live.textContent = String(message || "");
    },
  };

  window.BoxingVisionWorkspace = Workspace;
  const observer = new MutationObserver(() => {
    window.clearTimeout(Workspace.hydrateTimer);
    Workspace.hydrateTimer = window.setTimeout(() => Workspace.hydrate(), 60);
  });
  observer.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  document.addEventListener("DOMContentLoaded", () => Workspace.hydrate(), { once: true });
  Workspace.hydrate();
})();
