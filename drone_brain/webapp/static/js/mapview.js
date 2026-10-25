// MapLibre wrapper: base layers (offline vector / satellite / plain grid) plus all drone overlays.
import { THEMES, buildStyle } from "./style.js";
import { circlePolygon } from "./geo.js";

const GRID_STEPS = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];
const EMPTY = { type: "FeatureCollection", features: [] };
let protocolReady = false;

function dom(cls, html) { const e = document.createElement("div"); e.className = cls; if (html) e.innerHTML = html; return e; }
const feat = (geometry, props) => ({ type: "Feature", properties: props || {}, geometry });
const ll = (p) => [p.lon, p.lat];

export class MapView {
  constructor(container, info, theme, handlers) {
    this.container = container; this.info = info || { layers: [] }; this.theme = theme; this.h = handlers || {};
    this.mode = "grid"; this.data = {}; this.markers = { corners: [], alerts: new Map(), rb: [] };
    this.failed = false; this.lastMarkerDown = 0; this.editable = false;
    if (typeof maplibregl === "undefined") { this.failed = true; return; }
    if (!protocolReady && typeof pmtiles !== "undefined") {
      maplibregl.addProtocol("pmtiles", new pmtiles.Protocol().tile);
      protocolReady = true;
    }
    this.mode = this.hasLayer("vector") ? "vector" : "grid";
    const c = this.info.center ? { lat: this.info.center[0], lon: this.info.center[1] } : { lat: 0, lon: 0 };
    try {
      this.map = new maplibregl.Map({
        container, style: buildStyle(theme, this.mode, this.info), center: ll(c), zoom: this.info.center ? 17 : 17,
        maxZoom: 20, minZoom: 1, attributionControl: false, dragRotate: false, pitchWithRotate: false,
        fadeDuration: 0, localIdeographFontFamily: false,
      });
    } catch (e) { this.failed = true; this.report("error", "map constructor failed: " + e); return; }
    this.map.touchZoomRotate.disableRotation();
    // WebGL guard: a lost context (GPU reset, tab backgrounded on a phone) or any WebGL error must degrade to the
    // no-map layout instead of leaving a dead/crashing canvas.
    try {
      const canvas = this.map.getCanvas();
      canvas.addEventListener("webglcontextlost", (e) => {
        e.preventDefault(); this.report("warn", "WebGL context lost");
        this._lostTimer = setTimeout(() => this.degrade("WebGL context lost"), 3000);
      });
      canvas.addEventListener("webglcontextrestored", () => {
        clearTimeout(this._lostTimer); this.report("info", "WebGL context restored");
        try { this.map.resize(); this.map.triggerRepaint(); } catch (err) { this.degrade("WebGL restore failed"); }
      });
    } catch (e) { this.report("warn", "no canvas for WebGL guard: " + e); }
    this.map.addControl(new maplibregl.AttributionControl({ compact: true, customAttribution: this.info.attribution || "" }), "bottom-left");
    this.map.addControl(new maplibregl.ScaleControl({ unit: "metric", maxWidth: 90 }), "bottom-right");
    this.gridLabels = dom("grid-labels"); container.appendChild(this.gridLabels);
    this.map.on("style.load", () => this.initOverlays());
    this.map.on("move", () => this.updateGrid());
    this.map.on("click", (e) => {
      if (Date.now() - this.lastMarkerDown < 450) return;
      if (this.h.onTap) this.h.onTap({ lat: e.lngLat.lat, lon: e.lngLat.lng });
    });
    this.map.on("error", (e) => {
      const msg = e && e.error ? String(e.error.message || e.error) : "";
      if (/webgl|gl context|shader|framebuffer/i.test(msg)) { this.degrade("map error: " + msg.slice(0, 120)); return; }
      if (msg && !/tile|404|Failed to fetch/i.test(msg)) { console.warn("map", e.error); this.report("warn", "map: " + msg.slice(0, 200)); }
    });
  }

  report(level, msg) { try { if (window.clientLog) window.clientLog(level, msg, "mapview", ""); } catch (e) {} }
  /** Give up on the map: remove it and tell app.js to show the no-map layout. Idempotent. */
  degrade(reason) {
    if (this.failed) return;
    this.failed = true; clearTimeout(this._lostTimer);
    this.report("error", "map disabled: " + reason);
    try { this.markers.corners.forEach((m) => m.remove()); this.markers.alerts.forEach((m) => m.remove()); this.markers.rb.forEach((m) => m.remove()); } catch (e) {}
    try { this.map.remove(); } catch (e) {}
    if (this.h.onDegrade) this.h.onDegrade(reason);
  }

  hasLayer(kind) {
    return (this.info.layers || []).some((l) => (kind === "satellite" ? l.id === "satellite" : (l.type === "pmtiles" && l.kind === "vector")));
  }
  availableModes() {
    const m = ["grid"];
    if (this.hasLayer("vector")) m.unshift("vector");
    if (this.hasLayer("satellite")) m.push("satellite");
    return m;
  }

  // ---- styles / theme ----
  setMode(mode) {
    if (this.failed || mode === this.mode) return;
    this.mode = mode; this.map.setStyle(buildStyle(this.theme, mode, this.info), { diff: false });
  }
  setTheme(theme) {
    this.theme = theme;
    if (this.failed) return;
    this.map.setStyle(buildStyle(theme, this.mode, this.info), { diff: false });
    for (const m of this.markers.corners) m.getElement().dataset.theme = theme;
  }
  colors() { return THEMES[this.theme] || THEMES.dark; }

  initOverlays() {
    const c = this.colors(), m = this.map;
    const src = (id) => m.addSource(id, { type: "geojson", data: this.data[id] || EMPTY });
    ["grid", "fence", "poly", "legs", "route", "wps", "trail", "rb"].forEach(src);
    m.addLayer({ id: "grid", type: "line", source: "grid", paint: { "line-color": c.grid, "line-width": 1, "line-opacity": 0.8 } });
    m.addLayer({ id: "fence-line", type: "line", source: "fence", paint: { "line-color": c.fenceLine, "line-width": 2, "line-dasharray": [3, 3] } });
    m.addLayer({ id: "poly-fill", type: "fill", source: "poly", paint: { "fill-color": c.poly, "fill-opacity": 0.2 } });
    m.addLayer({ id: "poly-line", type: "line", source: "poly", paint: { "line-color": c.poly, "line-width": 3 } });
    m.addLayer({ id: "trail", type: "line", source: "trail", paint: { "line-color": c.trail, "line-width": 3, "line-opacity": 0.9 } });
    m.addLayer({ id: "legs", type: "line", source: "legs", paint: { "line-color": c.route, "line-width": 2, "line-dasharray": [2, 2], "line-opacity": 0.8 } });
    m.addLayer({ id: "route-casing", type: "line", source: "route", paint: { "line-color": c.casing, "line-width": 6, "line-opacity": 0.55 }, layout: { "line-join": "round" } });
    m.addLayer({ id: "route", type: "line", source: "route", paint: { "line-color": c.route, "line-width": 3 }, layout: { "line-join": "round" } });
    m.addLayer({ id: "wps", type: "circle", source: "wps", paint: { "circle-radius": 4, "circle-color": c.route, "circle-stroke-color": c.casing, "circle-stroke-width": 1.5 } });
    // what the Pixhawk holds (read back): thin dashed contrasting line on top of the planned route
    m.addLayer({ id: "rb-line", type: "line", source: "rb", paint: { "line-color": c.rb, "line-width": 2, "line-dasharray": [1.6, 1.4] }, layout: { "line-join": "round" } });
    this.updateGrid();
  }

  _set(id, fc) {
    this.data[id] = fc;
    const s = this.map && this.map.getSource && this.map.getSource(id);
    if (s) s.setData(fc);
  }

  // ---- camera ----
  setView(p, zoom) { if (!this.failed) this.map.jumpTo({ center: ll(p), zoom: zoom || this.map.getZoom() }); }
  flyTo(p, zoom) { if (!this.failed) this.map.easeTo({ center: ll(p), zoom: zoom || this.map.getZoom(), duration: 600 }); }
  fit(points, padding) {
    if (this.failed || !points.length) return;
    const b = new maplibregl.LngLatBounds();
    points.forEach((p) => b.extend(ll(p)));
    this.map.fitBounds(b, { padding: padding || 50, maxZoom: 19, duration: 500 });
  }
  resize() { if (!this.failed) this.map.resize(); }
  center() { if (this.failed) return { lat: 0, lon: 0 }; const c = this.map.getCenter(); return { lat: c.lat, lon: c.lng }; }

  // ---- corners / polygon ----
  setEditable(v) { this.editable = v; this.markers.corners.forEach((m) => m.setDraggable(v)); }
  setCorners(corners) {
    if (this.failed) return;
    const ms = this.markers.corners;
    while (ms.length > corners.length) ms.pop().remove();
    corners.forEach((c, i) => {
      if (!ms[i]) {
        const el = dom("corner-marker"); el.dataset.theme = this.theme;
        const mk = new maplibregl.Marker({ element: el, draggable: this.editable }).setLngLat(ll(c)).addTo(this.map);
        const stamp = () => { this.lastMarkerDown = Date.now(); };
        el.addEventListener("pointerdown", stamp); el.addEventListener("touchstart", stamp, { passive: true });
        mk.on("dragstart", stamp);
        mk.on("drag", () => this._dragging(i));
        mk.on("dragend", () => { stamp(); this._dragging(i, true); });
        ms[i] = mk;
      } else ms[i].setLngLat(ll(c));
      ms[i].getElement().textContent = String(i + 1);
    });
    this._drawPoly(corners);
  }
  _dragging(i, done) {
    const pts = this.markers.corners.map((m) => { const p = m.getLngLat(); return { lat: p.lat, lon: p.lng }; });
    this._drawPoly(pts);
    if (this.h.onCornerMove) this.h.onCornerMove(i, pts[i], !!done, pts);
  }
  _drawPoly(corners) {
    const ring = corners.map(ll);
    const f = [];
    if (corners.length >= 3) f.push(feat({ type: "Polygon", coordinates: [[...ring, ring[0]]] }));
    else if (corners.length === 2) f.push(feat({ type: "LineString", coordinates: ring }));
    this._set("poly", { type: "FeatureCollection", features: f });
  }

  // ---- fixed points ----
  _single(key, cls, html, p, opts) {
    if (this.failed) return;
    if (!p) { if (this.markers[key]) { this.markers[key].remove(); this.markers[key] = null; } return; }
    if (!this.markers[key]) this.markers[key] = new maplibregl.Marker(Object.assign({ element: dom(cls, html), rotationAlignment: "map" }, opts || {})).setLngLat(ll(p)).addTo(this.map);
    else this.markers[key].setLngLat(ll(p));
  }
  setHome(p) { this._single("home", "home-marker", '<span>H</span>', p); }
  setDrone(p, heading) {
    this._single("drone", "drone-marker", '<svg viewBox="-16 -16 32 32"><path d="M0,-13 L9,10 L0,5 L-9,10 Z" /></svg>', p);
    if (p && this.markers.drone) this.markers.drone.setRotation(heading || 0);
  }
  // label sits just above the first waypoint (MapLibre owns the element's transform, so offset via the Marker API)
  setStart(p) { this._single("start", "start-marker", "<span>START</span>", p, { anchor: "bottom", offset: [0, -7] }); }
  setFence(home, radius) {
    this._set("fence", { type: "FeatureCollection", features: home && radius ? [feat({ type: "LineString", coordinates: circlePolygon(home, radius) })] : [] });
  }
  setTrail(pts) {
    this._set("trail", { type: "FeatureCollection", features: pts.length > 1 ? [feat({ type: "LineString", coordinates: pts.map(ll) })] : [] });
  }

  // ---- route ----
  setRoute(home, wps) {
    if (!wps || !wps.length) { this._set("route", EMPTY); this._set("legs", EMPTY); this._set("wps", EMPTY); return; }
    const pts = wps.map(ll);
    this._set("route", { type: "FeatureCollection", features: [feat({ type: "LineString", coordinates: pts })] });
    this._set("legs", { type: "FeatureCollection", features: [
      feat({ type: "LineString", coordinates: [ll(home), pts[0]] }),
      feat({ type: "LineString", coordinates: [pts[pts.length - 1], ll(home)] })] });
    this._set("wps", { type: "FeatureCollection", features: pts.map((c, i) => feat({ type: "Point", coordinates: c }, { i })) });
    this.setStart(wps[0]);
  }
  clearRoute() { this.setRoute(null, null); this.setStart(null); }

  setPolygonOnly(corners) { this._drawPoly(corners); }

  /** Read-back route (items: [{seq, lat, lon, positional}]) drawn over the planned one, numbered markers (red = differs
   from the plan, badSeqs); null clears. */
  setReadback(items, badSeqs) {
    if (this.failed) return;
    const pts = (items || []).filter((i) => i.seq >= 1 && i.positional && (i.lat || i.lon));
    this._set("rb", { type: "FeatureCollection", features: pts.length > 1 ? [feat({ type: "LineString", coordinates: pts.map(ll) })] : [] });
    const ms = this.markers.rb;
    while (ms.length > pts.length) ms.pop().remove();
    pts.forEach((p, k) => {
      if (!ms[k]) ms[k] = new maplibregl.Marker({ element: dom("rb-marker"), pitchAlignment: "map" }).setLngLat(ll(p)).addTo(this.map);
      else ms[k].setLngLat(ll(p));
      ms[k].getElement().textContent = String(p.seq);
      ms[k].getElement().classList.toggle("bad", !!(badSeqs && badSeqs.has(p.seq)));
    });
  }

  // ---- alerts ----
  addAlert(a, onClick) {
    if (this.failed || a.lat == null || a.lon == null) return;
    const key = [a.level, a.track_id, a.ts].join("|");
    if (this.markers.alerts.has(key)) return;
    const icon = a.level === "heat" ? "🔥" : (a.level === "target_match" ? "★" : "P");
    const el = dom("alert-pin " + a.level, "<span>" + icon + "</span>");
    el.addEventListener("click", (ev) => { ev.stopPropagation(); this.lastMarkerDown = Date.now(); if (onClick) onClick(a); });
    el.addEventListener("pointerdown", () => { this.lastMarkerDown = Date.now(); });
    this.markers.alerts.set(key, new maplibregl.Marker({ element: el }).setLngLat(ll(a)).addTo(this.map));
    if (this.markers.alerts.size > 300) {      // bound the DOM on very long flights: drop the oldest pin
      const oldest = this.markers.alerts.keys().next().value;
      this.markers.alerts.get(oldest).remove(); this.markers.alerts.delete(oldest);
    }
  }
  clearAlerts() { this.markers.alerts.forEach((m) => m.remove()); this.markers.alerts.clear(); }

  // ---- plain grid (graticule) ----
  updateGrid() {
    if (this.failed || !this.map.getSource("grid")) return;
    if (this.mode !== "grid") { this._set("grid", EMPTY); this.gridLabels.textContent = ""; return; }
    const b = this.map.getBounds();
    const w = b.getWest(), e = b.getEast(), s = b.getSouth(), n = b.getNorth();
    const target = Math.min(n - s, (e - w) * 0.6) / 4;
    const step = GRID_STEPS.find((x) => x >= target) || GRID_STEPS[GRID_STEPS.length - 1];
    const feats = [], labels = [];
    const dec = Math.max(0, Math.ceil(-Math.log10(step)));
    for (let lat = Math.ceil(s / step) * step; lat <= n; lat += step) {
      feats.push(feat({ type: "LineString", coordinates: [[w, lat], [e, lat]] }));
      labels.push({ x: 4, y: this.map.project([w, lat]).y, t: lat.toFixed(dec) + "°", cls: "lat" });
    }
    for (let lon = Math.ceil(w / step) * step; lon <= e; lon += step) {
      feats.push(feat({ type: "LineString", coordinates: [[lon, s], [lon, n]] }));
      labels.push({ x: this.map.project([lon, n]).x, y: 4, t: lon.toFixed(dec) + "°", cls: "lon" });
    }
    this._set("grid", { type: "FeatureCollection", features: feats });
    this.gridLabels.textContent = "";
    for (const l of labels.slice(0, 40)) {
      const d = dom("gl " + l.cls); d.textContent = l.t; d.style.left = l.x + "px"; d.style.top = l.y + "px";
      this.gridLabels.appendChild(d);
    }
  }
}
