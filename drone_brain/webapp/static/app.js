/* KESTREL (drone-brain) phone UI (ES module). Fully offline: map library + tiles are served by the Jetson.
   All dynamic text goes through textContent. */
import { MapView } from "./js/mapview.js";
import * as G from "./js/geo.js";

const $ = (id) => document.getElementById(id);
// Mission states with a vehicle in the air or about to be ("manual" = pilot took over; the app only watches).
const ACTIVE_STATES = ["flying", "uploading", "arming", "rtl", "manual"];

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}
function fmt(v, d, unit) { return (v === null || v === undefined || Number.isNaN(v)) ? "-" : Number(v).toFixed(d) + (unit || ""); }
function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { /* private mode */ } return null; }

const DEFAULT_LIMITS = { min_alt_m: 5, max_alt_m: 60, default_alt_m: 20, default_speed_ms: 3, max_speed_ms: 10, overlap: 0.25,
  hfov_deg: 52.8, vfov_deg: 31.2, sensor_width_mm: 5.952, focal_length_mm: 6, image_width_px: 1920, person_height_m: 1.7,
  min_person_px: 80, geofence_radius_m: 300, max_area_m2: 200000, min_area_m2: 100, frame_interval_s: 1 };

const S = {
  step: 1, corners: [], plan: null, planning: false, planTimer: null, status: {}, tel: {}, alerts: [], seen: new Set(),
  limits: DEFAULT_LIMITS, info: { layers: [] }, trail: [], centered: false, lastState: null, planSig: null,
  altitude: 20, overlap: 0.25,
};
let mv = null;

/* ================= theme ================= */
// "dark" (graphite, default) or "light" (outdoor). The initial value is set by the inline script in index.html
// (saved choice, else prefers-color-scheme); the old "contrast" value maps to "light".
function curTheme() { return document.documentElement.dataset.theme === "light" ? "light" : "dark"; }
function applyTheme(t, save) {
  document.documentElement.dataset.theme = t;
  const m = document.querySelector('meta[name="theme-color"]'); if (m) m.content = t === "dark" ? "#121417" : "#f2f4f7";
  if (save) store("theme", t);
  if (mv) { mv.setTheme(t); }
}
$("theme-btn").addEventListener("click", () => applyTheme(curTheme() === "dark" ? "light" : "dark", true));
try {
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", (e) => {
    const saved = store("theme"); if (saved !== "dark" && saved !== "light") applyTheme(e.matches ? "light" : "dark", false);
  });
} catch (e) { /* old browsers */ }

/* ================= websocket ================= */
let ws = null, retry = 0, lastRx = 0;
function wsUrl() { return (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws"; }
let bannerTimer = null, connLost = false, connDismissed = false;
function setConn(ok) {
  $("conn-dot").className = "dot " + (ok ? "ok" : "bad"); $("conn-text").textContent = ok ? "online" : "offline";
  $("conn-dot").parentElement.title = ok ? "Connected" : "Offline";
  if (ok) { clearTimeout(bannerTimer); bannerTimer = null; connLost = false; connDismissed = false; updateConnBanner(); }
  else if (!bannerTimer && !connLost) bannerTimer = setTimeout(() => { bannerTimer = null; connLost = true; updateConnBanner(); }, 1500);   // not on every blip / page load
}
// The lost-connection line can be hidden only while nothing flies; in an active state it always shows.
function updateConnBanner() {
  const active = ACTIVE_STATES.includes(S.status.mission_state);
  $("conn-banner").hidden = !connLost || (connDismissed && !active);
  $("conn-x").hidden = active;
}
$("conn-x").addEventListener("click", (e) => { e.stopPropagation(); connDismissed = true; updateConnBanner(); });
$("conn-banner").addEventListener("click", () => $("conn-banner").classList.toggle("open"));
function connect() {
  try { ws = new WebSocket(wsUrl()); } catch (e) { scheduleReconnect(); return; }
  ws.onopen = () => { retry = 0; lastRx = Date.now(); setConn(true); };
  ws.onmessage = (ev) => { lastRx = Date.now(); let m; try { m = JSON.parse(ev.data); } catch (e) { return; } handle(m); };
  ws.onclose = () => { setConn(false); scheduleReconnect(); };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
}
function scheduleReconnect() { retry = Math.min(retry + 1, 6); setTimeout(connect, Math.min(500 * retry, 3000)); }
setInterval(() => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    try { ws.send(JSON.stringify({ type: "ping" })); } catch (e) {}
    if (Date.now() - lastRx > 15000) { try { ws.close(); } catch (e) {} }
  }
}, 5000);
const wsOpen = () => ws && ws.readyState === WebSocket.OPEN;

const ROUTES = { chat: "/api/chat", confirm: "/api/confirm", abort: "/api/abort", target: "/api/target", plan_area: "/api/plan_area", go: "/api/go" };
async function send(cmd) {
  if (wsOpen()) { ws.send(JSON.stringify(cmd)); return; }
  const body = cmd.type === "abort" ? "{}" : JSON.stringify(cmd);
  const r = await fetch(ROUTES[cmd.type], { method: "POST", headers: { "Content-Type": "application/json" }, body });
  if (!r.ok) throw new Error("HTTP " + r.status);
  handle(await r.json());
}

/* ================= message handling ================= */
function handle(m) {
  const d = m.data || {};
  switch (m.type) {
    case "status": applyStatus(d); break;
    case "telemetry": applyTelemetry(d); break;
    case "chat":
      if (d.role === "user") onUserEcho(d);
      else { if (d.role !== "system") setWaiting(false); addChat(d.role || "assistant", d.text || "", d.ts); }
      break;
    case "plan": {
      // a plan asked for in Chat stays in Chat as a card ("Open in Preview"); a plan from the Survey wizard opens Preview
      const fromChat = chat.waiting && !S.planning;
      setWaiting(false); endPlanning();
      showPlan(d, !fromChat);
      addChat("plan", "", null, { kind: "plan", plan: planSummary(d) });
      break;
    }
    case "alert": addAlert(d, true); break;
    case "flight_summary": onFlightSummary(d, true); break;
    case "error":
      if (chat.waiting) { setWaiting(false); addChat("error", d.message || "Error", null, { retry: chat.lastText }); }
      else addChat("error", d.message || "Error");
      if ($("panel-chat").hidden) toast(d.message || "Error");
      if (S.planning) { endPlanning(); goStep(S.corners.length >= 3 ? 2 : 1); }
      break;
  }
}

function applyStatus(s) {
  S.status = s;
  checkBoot(s.boot_id);
  if (s.telemetry) applyTelemetry(s.telemetry);
  const st = s.mission_state;
  document.body.classList.toggle("active", ACTIVE_STATES.includes(st));
  document.body.dataset.state = st || "";
  $("sb-state").textContent = (s.vehicle_mode === "readonly" ? "RO " : "") + shortState(st);
  $("sb-state").parentElement.title = stateLabel(st);
  updateConnBanner();
  $("demo-banner").hidden = !s.demo;
  document.body.classList.toggle("demo", !!s.demo);
  if (s.legal_ack !== undefined) S.legalAck = !!s.legal_ack;
  renderVehicleLog(s.statustext);
  renderSysNotes(s);
  $("sb-state").className = st === "flying" ? "val-ok" : (["rtl", "manual", "arming", "staged"].includes(st) ? "val-warn" : "");
  const t = s.target;
  $("cur-target").textContent = t ? ("Current target: " + t.description) : "No target set";
  $("cur-target").className = "target-line" + (t ? "" : " muted");
  if (Array.isArray(s.alerts)) s.alerts.forEach((a) => addAlert(a, false));
  if (s.plan) showPlan(s.plan, false);
  if (st !== S.lastState) {
    const flying = ACTIVE_STATES.includes(st);
    // upload / read-only mode never fly: stay on the Preview (the read-back table lives there)
    if (flying && S.step !== 4 && !["upload", "readonly"].includes(vmode())) { switchTab("survey"); goStep(4); }
    // uploaded for review: land on the Review step (Preview)
    if (st === "staged" && !["upload", "readonly"].includes(vmode())) { switchTab("survey"); if (S.step !== 3) goStep(3); }
    S.lastState = st;
  }
  renderBench(s); onReadback(s); renderReview(s);
  if (s.last_flight_summary) onFlightSummary(s.last_flight_summary, false);
  $("new-survey").hidden = ACTIVE_STATES.includes(st);
  updateMapContext(); updateFlightInfo(); updateStepper();
}

const FIX = { 0: "NO FIX", 1: "NO FIX", 2: "2D", 3: "3D", 4: "DGPS", 5: "RTK-F", 6: "RTK" };
function applyTelemetry(t) {
  S.tel = t;
  $("sb-mode").textContent = t.connected === false ? "NO LINK" : (t.mode || "-");
  $("sb-mode").className = t.connected === false ? "val-bad" : "";
  $("sb-mode-item").classList.toggle("nolink", t.connected === false);   // a lost vehicle link shows in every state
  $("sb-armed").textContent = "ARMED";
  $("sb-armed-item").hidden = !t.armed;
  const fix = t.gps_fix;
  $("sb-gps").textContent = (fix === null || fix === undefined) ? "-" : (fix < 2 ? "NO FIX" : (FIX[fix] || fix) + " " + (t.satellites ?? "?"));
  $("sb-gps").parentElement.title = "GPS fix and satellites" + (fix == null ? "" : ": " + (FIX[fix] || fix) + ", " + (t.satellites ?? "?") + " sat");
  $("sb-gps").className = (fix === null || fix === undefined) ? "" : (fix >= 3 ? "val-ok" : "val-bad");
  $("sb-batt").textContent = fmt(t.battery_pct, 0, "%");
  $("sb-batt").className = t.battery_pct == null ? "" : (t.battery_pct < 25 ? "val-bad" : (t.battery_pct < 40 ? "val-warn" : ""));
  $("sb-alt").textContent = fmt(t.alt_rel_m, 0, "m");
  if (t.lat != null && t.lon != null && mv) {
    const p = { lat: t.lat, lon: t.lon };
    const last = S.trail[S.trail.length - 1];
    if (t.armed && (!last || G.distM(last, p) > 1.5)) { S.trail.push(p); if (S.trail.length > 600) S.trail.shift(); mv.setTrail(S.trail); }
    mv.setDrone(p, t.heading_deg);
    if (!S.centered) { S.centered = true; mv.setView(homePoint() || p, 17); }   // drone home first, else the drone
    $("dronepos-btn").disabled = false;
    if (S.step === 4 && t.armed && followDrone) mv.flyTo(p);
  }
  updateFlightInfo(); renderVehicleLive();
  if (S.step === 1) refreshArea();
}
let followDrone = true;

function homePoint() {
  const h = S.status && S.status.home;
  if (h && h.lat != null) return { lat: h.lat, lon: h.lon };
  if (S.plan && S.plan.plan && S.plan.plan.home) return S.plan.plan.home;
  if (S.tel && S.tel.lat != null) return { lat: S.tel.lat, lon: S.tel.lon };
  return null;
}
function updateMapContext() {
  if (!mv) return;
  const h = homePoint();
  mv.setHome(h); mv.setFence(h, S.limits.geofence_radius_m);
  if (h && !S.centered) { S.centered = true; mv.setView(h, 17); }
}

/* ================= wizard / stepper ================= */
function goStep(n) {
  S.step = n;
  for (let i = 1; i <= 4; i++) $("step-" + i).hidden = i !== n;
  updateStepper();
  mv && mv.setEditable(n === 1);
  if (n === 1) { S.plan && (mv && mv.clearRoute()); mv && mv.setCorners(S.corners); refreshArea(); }
  if (n === 2) { mv && mv.clearRoute(); mv && mv.setCorners(S.corners); refreshQuestions(); }
  if (n === 3) { renderPlanView(); }
  if (n === 4) { if (S.plan) drawPlan(S.plan); updateFlightInfo(); }
  drawReadback();
  $("sheet").scrollTop = 0;
  requestAnimationFrame(() => mv && mv.resize());
  if (n === 3 && S.plan) setTimeout(() => drawPlan(S.plan, true), 120);
}
function areaValid() { return S.corners.length >= 3 && !G.selfIntersects(S.corners); }
// One-line progress: "Step 2 of 4 · Questions" with back / next; next is enabled under the same rules the old step buttons had.
const STEP_NAMES = ["", "Area", "Questions", "Preview", "Flight"];
function stepAllowed(n) {
  const flying = ACTIVE_STATES.includes(S.status.mission_state);
  if (n < 1 || n > 4) return false;
  return !((n === 2 && !areaValid()) || (n === 3 && !S.plan && !S.planning && !hasReadback()) || (n === 4 && !flying && !S.plan && S.alerts.length === 0 && S.step !== 4));
}
function updateStepper() {
  const name = (S.step === 3 && S.status.mission_state === "staged") ? "Review" : STEP_NAMES[S.step];
  $("step-label").textContent = "Step " + S.step + " of 4 · " + name;
  document.querySelectorAll("#stepper .step-bar i").forEach((b) => {
    const n = Number(b.dataset.step);
    b.className = n === S.step ? "on" : (n < S.step ? "done" : "");
  });
  $("step-prev").disabled = S.step <= 1;
  $("step-next").disabled = !stepAllowed(S.step + 1);
}
$("step-prev").addEventListener("click", () => { if (S.step > 1) goStep(S.step - 1); });
$("step-next").addEventListener("click", () => { if (stepAllowed(S.step + 1)) goStep(S.step + 1); });

/* ================= step 1: area ================= */
function addCorner(p) {
  if (S.step !== 1) return;
  if (S.corners.length >= 8) { toast("At most 8 corners. Use Undo or drag a corner."); return; }
  S.corners.push(p); mv.setCorners(S.corners); refreshArea();
}
const STATE_LABEL = { staged: "Uploaded \u2014 awaiting GO", arming: "ARMING" };
const stateLabel = (st) => STATE_LABEL[st] || String(st || "-").toUpperCase();
const shortState = (st) => (st === "staged" ? "UPLOADED" : String(st || "-").toUpperCase());   // status bar: one short word
function chip(label, value, cls) {
  const c = el("span", "chip" + (cls ? " " + cls : ""));
  c.appendChild(el("small", null, label)); c.appendChild(document.createTextNode(value)); return c;
}
function notice(cls, text) { return el("div", "notice " + cls, text); }

function refreshArea() {
  const n = S.corners.length, L = S.limits;
  const chips = $("area-stats"), warn = $("area-warn");
  chips.textContent = ""; warn.textContent = "";
  const hint = $("area-hint");
  if (n < 4) hint.textContent = "Tap the map to drop corner " + (n + 1) + " of 4.";
  else hint.textContent = "Drag a numbered corner to adjust, or continue.";
  const home = homePoint();
  let valid = n >= 3;
  if (n >= 3) {
    const a = G.areaM2(S.corners);
    chips.appendChild(chip("Area", G.fmtArea(a) + (a >= 10000 ? "" : "  (" + (a / 10000).toFixed(2) + " ha)"), a > L.max_area_m2 ? "low" : ""));
    if (G.selfIntersects(S.corners)) { valid = false; warn.appendChild(notice("err", "The outline crosses itself. Drag corners so 1-2-3-4 go around the area in order.")); }
    if (a > L.max_area_m2) { valid = false; warn.appendChild(notice("err", "Area is above the " + G.fmtArea(L.max_area_m2) + " limit.")); }
    else if (a < L.min_area_m2) { valid = false; warn.appendChild(notice("err", "Area is below the " + L.min_area_m2 + " m² minimum.")); }
    $("map-chip").hidden = false; $("map-chip").className = ""; $("map-chip").textContent = G.fmtArea(a);
  } else { $("map-chip").hidden = true; }
  if (n >= 1 && home) {
    const far = Math.max(...S.corners.map((c) => G.distM(home, c)));
    const out = far > L.geofence_radius_m;
    chips.appendChild(chip("From home", Math.round(far) + " m", out ? "low" : ""));
    if (out) { valid = false; warn.appendChild(notice("err", "A corner is " + Math.round(far) + " m from home, outside the " + L.geofence_radius_m + " m geofence (orange circle).")); }
  } else if (n >= 1 && !home) {
    warn.appendChild(notice("warn", "No GPS position yet: distances from home cannot be checked."));
  }
  $("to-questions").disabled = !(valid && n >= 3);
  $("undo-btn").disabled = n === 0; $("clear-btn").disabled = n === 0;
  updateStepper();
}
$("undo-btn").addEventListener("click", () => { S.corners.pop(); mv.setCorners(S.corners); refreshArea(); });
$("clear-btn").addEventListener("click", () => { S.corners = []; mv.setCorners(S.corners); refreshArea(); });
$("dronepos-btn").addEventListener("click", () => {
  const t = S.tel;
  if (t.lat == null) { toast("No GPS position from the drone yet."); return; }
  addCorner({ lat: t.lat, lon: t.lon });
});
$("paste-toggle").addEventListener("click", () => {
  const w = $("paste-wrap"); w.hidden = !w.hidden; $("paste-toggle").setAttribute("aria-expanded", String(!w.hidden));
});
$("paste-apply").addEventListener("click", async () => {
  try {
    const r = await fetch("/api/parse_coords", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: $("paste-input").value }) });
    const j = await r.json();
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : (Array.isArray(j.detail) && j.detail[0] && j.detail[0].msg) || "HTTP " + r.status);
    S.corners = j.corners; mv.setCorners(S.corners); mv.fit(S.corners); refreshArea();
    $("paste-wrap").hidden = true;
  } catch (e) { toast("Coordinates: " + e.message); }
});
$("to-questions").addEventListener("click", () => goStep(2));

/* ================= step 2: questions ================= */
function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
function setupQuestions() {
  const L = S.limits, r = $("alt-range");
  r.min = L.min_alt_m; r.max = L.max_alt_m;
  S.altitude = clamp(Number(store("alt")) || L.default_alt_m, L.min_alt_m, L.max_alt_m);
  r.value = S.altitude;
  S.overlap = L.overlap; $("ov-range").value = Math.round(L.overlap * 100);
  const tt = store("target"); if (tt) $("target-text").value = tt;
  r.addEventListener("input", () => { S.altitude = Number(r.value); refreshQuestions(); });
  $("ov-range").addEventListener("input", () => { S.overlap = Number($("ov-range").value) / 100; refreshQuestions(); });
  $("alt-suggest").addEventListener("click", () => { S.altitude = suggestedAlt(); r.value = S.altitude; refreshQuestions(); });
}
function suggestedAlt() {
  const L = S.limits;
  return Math.floor(clamp(G.maxAltForPx(L.min_person_px, L), L.min_alt_m, L.max_alt_m));
}
function refreshQuestions() {
  const L = S.limits, h = S.altitude;
  $("alt-val").textContent = Math.round(h) + " m";
  $("ov-val").textContent = Math.round(S.overlap * 100) + "%";
  const px = G.personPx(h, L), w = G.footprintW(h, L.hfov_deg), len = G.footprintL(h, L.vfov_deg);
  const cls = px >= L.min_person_px ? "good" : (px >= 0.6 * L.min_person_px ? "mid" : "low");
  const box = $("alt-feedback"); box.textContent = "";
  box.appendChild(chip("Footprint", w.toFixed(0) + " m wide"));
  box.appendChild(chip("Person", Math.round(px) + " px", cls));
  box.appendChild(chip("GSD", (G.gsd(h, L) * 100).toFixed(1) + " cm/px"));
  const sug = suggestedAlt(), b = $("alt-suggest");
  b.hidden = !(h > sug);
  b.textContent = "Person drops below " + L.min_person_px + " px. Use suggested " + sug + " m";
  const vmax = len * (1 - S.overlap) / Math.max(L.frame_interval_s, 0.01);
  const speed = Math.min(L.default_speed_ms, vmax, L.max_speed_ms);
  $("speed-line").textContent = "Speed: " + speed.toFixed(1) + " m/s (set automatically" + (speed < L.default_speed_ms ? ", limited by the camera frame rate" : "") + ")";
  mv && mv.setEditable(false);
}
$("q-back").addEventListener("click", () => goStep(1));
$("to-preview").addEventListener("click", async () => {
  if (!areaValid()) { toast("Fix the area first."); return; }
  const text = $("target-text").value.trim();
  store("alt", String(S.altitude)); store("target", text);
  S.planning = true; S.plan = null; S.planSig = null;
  clearTimeout(S.planTimer); S.planTimer = setTimeout(() => { if (S.planning) { endPlanning(); toast("Planning timed out."); goStep(2); } }, 130000);
  goStep(3);
  const req = { type: "plan_area", corners: S.corners, altitude_m: S.altitude, overlap: S.overlap };
  if (text) req.target_text = text;
  try { await send(req); }
  catch (e) { endPlanning(); toast("Could not send: " + e.message); goStep(2); }
});
function endPlanning() { S.planning = false; clearTimeout(S.planTimer); }

/* ================= step 3: preview ================= */
function showPlan(p, navigate) {
  if (S.plan && S.plan.plan_id === p.plan_id && !navigate) return;
  S.plan = p;
  if (navigate) { switchTab("survey"); goStep(3); } else if (S.step === 3) renderPlanView();
  updateStepper();
}
function renderPlanView() {
  const p = S.plan;
  renderBench(S.status); renderReadback();
  $("planning").hidden = !S.planning;
  $("plan-empty").hidden = !!p || S.planning;
  $("plan-body").hidden = !p || S.planning;
  if (!p) return;
  const plan = p.plan || {}, safety = p.safety || { ok: false, errors: ["No safety result"], warnings: [] };
  $("plan-target").textContent = "Looking for: " + ((p.target && p.target.description) || "-");
  const errs = $("plan-errors"), warns = $("plan-warnings");
  errs.textContent = ""; warns.textContent = "";
  (safety.errors || []).forEach((x) => errs.appendChild(notice("err", "BLOCKED: " + x)));
  if (safety.ok) errs.appendChild(notice("okc", "Safety checks passed"));
  if (p.home_source === "bench" && p.bench_notice) warns.appendChild(notice("warn", p.bench_notice));
  [...new Set([].concat(safety.warnings || [], plan.warnings || []))].filter((x) => !String(x).startsWith("BENCH HOME")).forEach((x) => warns.appendChild(notice("warn", x)));
  const L = S.limits, wps = plan.waypoints || [];
  const area = plan.area_m2 ?? ((plan.mission && plan.mission.length_m) ? plan.mission.length_m * plan.mission.width_m : null);
  const lanes = plan.lane_count ?? Math.floor(wps.length / 2);
  const pxCls = plan.person_px >= L.min_person_px ? "good" : (plan.person_px >= 0.6 * L.min_person_px ? "mid" : "low");
  const stats = [   // first six on the sheet, the rest under "Plan details"
    ["Flight time", G.fmtDur(plan.estimated_flight_s)], ["Area", area == null ? "-" : G.fmtArea(area)], ["Altitude", fmt(plan.altitude_m, 0, " m")],
    ["Lanes", String(lanes)], ["Path", fmt(plan.path_length_m, 0, " m")], ["Person size", fmt(plan.person_px, 0, " px"), pxCls],
    ["Speed", fmt(plan.speed_ms, 1, " m/s")], ["GSD", plan.gsd_m == null ? "-" : fmt(plan.gsd_m * 100, 1, " cm/px")],
    ["Waypoints", String(wps.length)],
  ];
  const dl = $("plan-stats"), dl2 = $("plan-stats-more"); dl.textContent = ""; dl2.textContent = "";
  stats.forEach(([k, v, c], i) => { const d = el("div"); d.appendChild(el("dt", null, k)); d.appendChild(el("dd", c || "", v)); (i < 6 ? dl : dl2).appendChild(d); });
  $("plan-battery").textContent = plan.battery_note || "";
  renderPreflight(p);
  const tb = $("wp-table").tBodies[0]; tb.textContent = "";
  wps.forEach((w) => { const tr = el("tr"); [w.seq, fmt(w.north_m, 1), fmt(w.east_m, 1), fmt(w.alt_m, 0)].forEach((v) => tr.appendChild(el("td", null, String(v)))); tb.appendChild(tr); });
  resetConfirm();
  $("confirm-btn").disabled = !safety.ok;
  $("confirm-row").hidden = S.status.mission_state === "staged";
  $("confirm-btn").textContent = safety.ok ? confirmLabel() : "Cannot confirm: safety check failed";
  $("clear-mission-btn").hidden = vmode() !== "upload";
  drawPlan(p, true);
}
function drawPlan(p, fit) {
  if (!mv || !p) return;
  const plan = p.plan || {}, home = plan.home;
  const m = plan.mission || {};
  const outline = (m.corners && m.corners.length) ? m.corners : (m.length_m ? G.rectCorners(m, home) : []);
  mv.setCorners([]); mv.setPolygonOnly(outline);
  mv.setRoute(home, plan.waypoints || []);
  if (fit) mv.fit([home, ...outline, ...(plan.waypoints || [])], { top: 76, bottom: 44, left: 40, right: 96 });
}
$("p-back").addEventListener("click", () => goStep(S.corners.length >= 3 ? 1 : 1));
$("wp-toggle").addEventListener("click", () => {
  const w = $("wp-wrap"); w.hidden = !w.hidden;
  $("wp-toggle").textContent = w.hidden ? "Show waypoints" : "Hide waypoints";
  $("wp-toggle").setAttribute("aria-expanded", String(!w.hidden));
});
let confirmTimer = null;
function resetConfirm() {
  clearTimeout(confirmTimer); confirmTimer = null;
  const b = $("confirm-btn"); b.classList.remove("armed-confirm");
  if (S.plan && S.plan.safety && S.plan.safety.ok) b.textContent = confirmLabel();
}
function vmode() { return (S.status && S.status.vehicle_mode) || "sim"; }
function confirmLabel() {
  const m = vmode();
  return m === "upload" ? "Upload to Pixhawk (no flight)" : (m === "readonly" ? "Save plan (read-only, no upload)" : "Upload plan to drone");
}
$("confirm-btn").addEventListener("click", async () => {
  const p = S.plan; if (!p || !(p.safety && p.safety.ok)) return;
  const b = $("confirm-btn");
  if (!confirmTimer) {
    b.classList.add("armed-confirm");
    b.textContent = vmode() === "upload" ? "TAP AGAIN TO UPLOAD (NO FLIGHT)" : (vmode() === "readonly" ? "TAP AGAIN TO SAVE" : "TAP AGAIN TO UPLOAD PLAN");
    confirmTimer = setTimeout(resetConfirm, 5000); return;
  }
  resetConfirm();
  if (!(await ensureLegal())) return;
  try { await send({ type: "confirm", plan_id: p.plan_id }); }   // sim/full only uploads: stay on Review until GO
  catch (err) { addChat("error", "Confirm failed: " + err.message); toast("Confirm failed: " + err.message); }
});

let clearTimer = null;
$("clear-mission-btn").addEventListener("click", async () => {
  const b = $("clear-mission-btn");
  if (!clearTimer) {
    b.classList.add("armed-confirm"); b.textContent = "TAP AGAIN TO CLEAR THE PIXHAWK MISSION";
    clearTimer = setTimeout(() => { clearTimer = null; b.classList.remove("armed-confirm"); b.textContent = "Clear mission on Pixhawk"; }, 5000);
    return;
  }
  clearTimeout(clearTimer); clearTimer = null; b.classList.remove("armed-confirm"); b.textContent = "Clear mission on Pixhawk";
  try {
    const r = await fetch("/api/vehicle/clear_mission", { method: "POST" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const m = await r.json(); toast((m.data && m.data.text) || "done", true);
  } catch (e) { toast("Clear failed: " + e.message); }
});

/* Phase 7: Vehicle tab - live state, STATUSTEXT feed, read-only vehicle check */
const SEV = (s) => (s <= 3 ? "sev-bad" : (s === 4 ? "sev-warn" : "sev-info"));
function renderVehicleLive() {
  const dl = $("veh-live"); if (!dl || $("panel-vehicle").hidden) return;
  const t = S.tel || {}, st = S.status || {};
  $("veh-mode").textContent = "(" + vmode() + " mode)";
  $("veh-read-btn").hidden = !["upload", "readonly"].includes(vmode());
  const rows = [
    ["Link", t.connected ? "OK, hb " + fmt(t.heartbeat_age_s, 1, " s") : "LOST", t.connected ? "good" : "low"],
    ["Mode", t.mode || "-"], ["Armed", t.armed ? "YES" : "no", t.armed ? "mid" : ""],
    ["Landed", t.landed_state || t.system_status || "-"], ["GPS", (FIX[t.gps_fix] || "-") + " " + (t.satellites ?? "?") + " sat"],
    ["HDOP", fmt(t.hdop, 1)], ["Position", t.lat == null ? "-" : t.lat.toFixed(6) + ", " + t.lon.toFixed(6)],
    ["Alt rel", fmt(t.alt_rel_m, 1, " m")], ["Heading", fmt(t.heading_deg, 0, "\u00b0")],
    ["Speed", fmt(t.groundspeed_ms, 1, " m/s")], ["Battery", fmt(t.voltage_v, 2, " V") + " / " + fmt(t.battery_pct, 0, "%")],
    ["Mission", t.mission_seq == null ? "-" : (t.mission_seq + " / " + (t.mission_count ?? "?"))],
    ["Home", st.home ? st.home.lat.toFixed(6) + ", " + st.home.lon.toFixed(6) + " (" + fmt(st.home.amsl_m, 0, " m") + ")" : "unknown"],
  ];
  dl.textContent = "";
  rows.forEach(([k, v, c]) => { const d = el("div"); d.appendChild(el("dt", null, k)); d.appendChild(el("dd", c || "", v)); dl.appendChild(d); });
}
function renderVehicleLog(lines) {
  const box = $("veh-log"); if (!box || !Array.isArray(lines) || !lines.length) return;
  box.textContent = "";
  lines.forEach((m) => {   // each distinct message once; x N = repeat counter, time = last seen
    const d = el("div", SEV(m.severity), new Date(m.ts * 1000).toLocaleTimeString() + "  " + m.text + (m.count > 1 ? "  \u00d7" + m.count : "")); box.appendChild(d);
  });
  box.scrollTop = box.scrollHeight;
}
function renderVehicleReport(r) {
  const box = $("veh-report"); box.textContent = "";
  box.appendChild(notice(r.verdict === "READY" ? "okc" : "err", r.verdict + (r.duration_s != null ? " (" + r.duration_s + " s)" : "")));
  const list = (title, items, cls) => {
    if (!items || !items.length) return;
    box.appendChild(el("h3", null, title)); const ul = el("ul");
    items.forEach((x) => ul.appendChild(el("li", cls || null, String(x)))); box.appendChild(ul);
  };
  list("Reasons", r.reasons, "val-bad"); list("Warnings", r.warnings, "val-warn");
  if (r.prearm) list("Pre-arm messages (ack " + r.prearm.ack + ")", (r.prearm.messages || []).map((m) => m.text));
  const g = r.gps || {}, e = r.ekf || {}, sn = r.sensors || {}, fw = r.firmware || {}, v = r.vehicle || {}, rc = r.rc || {};
  list("Vehicle", [(v.type || "-") + ", frame " + ((r.frame || {}).class || "-") + " " + ((r.frame || {}).type || ""),
    "Firmware " + (fw.version || "-") + " " + (fw.type || ""), "Mode " + (v.mode || "-") + ", armed " + v.armed + ", " + (v.landed_state || v.system_status || "")]);
  list("GPS", [g.fix + ", " + (g.satellites ?? 0) + " sats, HDOP " + (g.hdop ?? "-"), "Home " + (g.home ? g.home.lat.toFixed(6) + ", " + g.home.lon.toFixed(6) : "not set")]);
  list("EKF " + (e.level || "-"), Object.entries(e.variances || {}).map(([k, x]) => k + " " + x + " " + ((e.levels || {})[k] || "")));
  list("Sensors", sn.available ? ["unhealthy: " + ((sn.unhealthy || []).join(", ") || "none"), "pre-arm bit: " + sn.prearm_ok] : ["no SYS_STATUS"]);
  list("Battery / RC", [fmt((r.battery || {}).voltage_v, 2, " V") + " " + fmt((r.battery || {}).pct, 0, "%"),
    "RC " + (rc.received ? rc.chancount + " ch, rssi " + (rc.rssi ?? "-") : "not received")]);
  if (r.preflight) list("Parameters", (r.preflight.checks || []).map((c) => (c.ok ? "\u2713 " : "\u2717 ") + c.param + " = " + (c.value ?? "-") + ": " + c.message));
}
$("veh-check-btn").addEventListener("click", async () => {
  const b = $("veh-check-btn"); b.disabled = true; b.textContent = "Checking (about 10 s)...";
  try {
    const r = await fetch("/api/vehicle/check");
    if (!r.ok) throw new Error("HTTP " + r.status);
    renderVehicleReport(await r.json());
  } catch (e) { toast("Vehicle check failed: " + e.message); }
  b.disabled = false; b.textContent = "Run vehicle check (read-only)";
});

/* Phase 7: airspace ceiling, Pixhawk pre-flight cross-check, downloads */
function renderPreflight(p) {
  const box = $("plan-preflight"), files = $("plan-files");
  if (!box) return;
  box.textContent = ""; files.textContent = "";
  const lim = p.limits, pre = p.preflight;
  if (lim) box.appendChild(notice("info", "Ceiling " + fmt(lim.effective_max_alt_m, 0, " m") + " (" + lim.ceiling_source + ")"
    + " \u00b7 floor " + fmt(lim.effective_min_alt_m, 0, " m") + " (" + lim.floor_source + ")"));
  if (pre) {
    const line = "Pixhawk pre-flight " + (pre.ok ? "OK" : "FAILED") + " \u00b7 RTL " + fmt(pre.rtl_alt_m, 0, " m") + " \u00b7 fence " + (pre.fence || "-");
    box.appendChild(notice(pre.ok ? "okc" : "err", line));
    if (!pre.ok) $("plan-warnings").appendChild(notice("err", line + " (see Plan details)"));   // never only inside the collapsed details
    const t = el("table"), tb = el("tbody"); t.appendChild(tb);
    (pre.checks || []).forEach((c) => {
      const tr = el("tr");
      tr.appendChild(el("td", c.ok ? "ok" : (c.severity === "warning" ? "warn" : "bad"), c.ok ? "\u2713" : (c.severity === "warning" ? "!" : "\u2717")));
      tr.appendChild(el("td", null, c.param)); tr.appendChild(el("td", null, c.value == null ? "-" : String(c.value)));
      tr.appendChild(el("td", null, c.message)); tb.appendChild(tr);
    });
    box.appendChild(t);
  }
  if (p.files) {
    Object.entries(p.files).forEach(([name, url]) => {
      if (!name.startsWith("mission")) return;
      const a = el("a", null, name === "mission.waypoints" ? "mission.waypoints (planned)" : name); a.href = url; a.setAttribute("download", ""); files.appendChild(a);
    });
    files.appendChild(el("div", "rb-hint", "Mission Planner \u2192 Plan \u2192 Load WP File"));
  }
}

/* ================= bench home + read-back view ================= */
const BENCH_TEXT = "BENCH HOME \u2014 not from GPS (upload test only)";
function benchChip(text) { return el("div", "bench-chip", "\u26a0 " + (text || BENCH_TEXT)); }
/* slim one-line banners: tap to expand, x to dismiss (per text; a new text shows again) */
const slimDismissed = new Set(), slimOpen = new Set();
function slimLine(cls, text, details) {
  const d = el("div", "slim " + cls + (slimOpen.has(text) ? " open" : ""));
  const t = el("span", "slim-txt"); t.appendChild(el("span", "slim-sum", text));
  if (details && details.length) details.forEach((x) => t.appendChild(el("span", "slim-detail", x)));
  d.appendChild(t);
  if (details && details.length) d.appendChild(el("span", "slim-more", "Details"));
  d.addEventListener("click", () => { d.classList.toggle("open"); if (d.classList.contains("open")) slimOpen.add(text); else slimOpen.delete(text); });
  const x = el("button", "slim-x", "×"); x.type = "button"; x.setAttribute("aria-label", "Dismiss");
  x.addEventListener("click", (e) => {
    e.stopPropagation(); slimDismissed.add(d.dataset.key || text);
    const box = d.parentElement; d.remove(); if (box && !box.children.length) box.hidden = true;
  });
  d.appendChild(x); return d;
}
function renderSlims(box, items) {   // items: [cls, summary text, optional full-text details]
  const key = (t, det) => t + "|" + (det || []).join("|");   // dismissed until the underlying text changes
  box.textContent = "";
  items.filter(([, t, det]) => t && !slimDismissed.has(key(t, det))).forEach(([c, t, det]) => {
    const d = slimLine(c, t, det); d.dataset.key = key(t, det); box.appendChild(d);
  });
  box.hidden = !box.children.length;
}
function renderBench(s) {
  const b = (s || {}).bench_home, bar = $("benchbar"), wrap = $("bench-chip-wrap");
  wrap.textContent = "";
  renderSlims(bar, !b ? [] : (b.active ? [["warn", "⚠ " + (b.text || BENCH_TEXT)]] : [["info", b.note || "Bench home is not in use."]]));
  const p = S.plan, show = !!(p && p.home_source === "bench");
  wrap.hidden = !show; if (show) wrap.appendChild(benchChip());
}
const hasReadback = () => Array.isArray(S.status.readback) && S.status.readback.length > 0;
function onReadback(s) {
  const info = s.readback_info, ts = info ? info.ts : 0;
  if (S.rbSeen === undefined) { S.rbSeen = ts; }
  else if (ts && ts !== S.rbSeen) {      // a new upload / re-read: put a card in the chat too
    S.rbSeen = ts;
    addChat("plan", "", null, { kind: "readback", info: { plan_id: info.plan_id, source: info.source, files: info.files, hint: info.hint,
      text: (info.diff && info.diff.summary && info.diff.summary.text) || "", ok: !!(info.diff && info.diff.summary && info.diff.summary.all_ok),
      compared: !!(info.diff && info.diff.compared), bench: (s.bench_home && s.bench_home.active) } });
  }
  if (S.step === 3) renderReadback();
  drawReadback();
  updateStepper();
}
function drawReadback() {
  const on = (S.step === 3 || S.step === 4) && hasReadback();
  const rows = (((S.status.readback_info || {}).diff || {}).rows) || [];
  mv && mv.setReadback(on ? S.status.readback : null, new Set(rows.filter((r) => r.ok === false).map((r) => r.seq)));
  const lg = $("map-legend"); lg.textContent = "";
  lg.hidden = !on;
  if (on) {
    const a = el("span"); a.appendChild(el("i")); a.appendChild(document.createTextNode("Planned"));
    const b = el("span"); b.appendChild(el("i", "pix")); b.appendChild(document.createTextNode("On Pixhawk"));
    lg.appendChild(a); lg.appendChild(b);
  }
}
function rbDescr(v) {
  if (!v) return "-";
  const nf = (x, n) => (typeof x === "number" && isFinite(x)) ? x.toFixed(n) : "?";
  if (v.positional) return nf(v.lat, 7) + ", " + nf(v.lon, 7) + " \u00b7 " + nf(v.alt, 1) + " m";
  const bits = [];
  if (v.alt) bits.push("alt " + nf(v.alt, 1) + " m");
  [["p1", v.p1], ["p2", v.p2], ["p3", v.p3], ["p4", v.p4]].forEach(([k, x]) => { if (x) bits.push(k + "=" + x); });
  return bits.length ? bits.join(" \u00b7 ") : "no parameters";
}
function renderReadback() {
  const m = vmode(), canRead = m === "upload" || m === "readonly";
  const items = S.status.readback || [], info = S.status.readback_info;
  const card = $("rb-card");
  card.hidden = !(canRead || items.length);
  $("rb-read").hidden = !canRead;
  const sum = $("rb-summary"), links = $("rb-links"), tb = $("rb-table").tBodies[0];
  sum.textContent = ""; links.textContent = ""; tb.textContent = "";
  $("rb-hint").hidden = true; $("rb-details").hidden = !items.length;
  if (!info) { if (canRead) sum.appendChild(notice("info", "Nothing read from the Pixhawk yet. Upload a plan, or read what it holds now.")); return; }
  const d = info.diff || {}, sm = d.summary || {};
  const when = new Date(info.ts * 1000).toLocaleTimeString();
  if (d.compared) sum.appendChild(notice(sm.all_ok ? "okc" : "err", (sm.all_ok ? "\u2705 " : "\u274c ") + sm.text + "  (" + (info.source === "upload" ? "uploaded " : "read ") + when + ")"));
  else sum.appendChild(notice("info", sm.text + "  (" + when + ")"));
  if (d.compared && S.plan && info.plan_id && S.plan.plan_id !== info.plan_id) sum.appendChild(notice("warn", "Compared with plan " + info.plan_id + ", not the one in this preview."));
  const names = { "mission.waypoints": "mission.waypoints (planned)", "readback.waypoints": "readback.waypoints (on the Pixhawk)" };
  Object.entries(info.files || {}).forEach(([n, url]) => { const a = el("a", null, names[n] || n); a.href = url; a.setAttribute("download", ""); links.appendChild(a); });
  $("rb-hint").hidden = !links.children.length;
  const rows = d.compared ? d.rows : items.map((v) => ({ seq: v.seq, name: v.name, planned: null, readback: v, ok: null, note: "" }));
  rows.forEach((r) => {
    const cls = r.ok === false ? "bad" : "";
    const tr = el("tr", "top " + cls);
    tr.appendChild(el("td", null, String(r.seq)));
    const nm = el("td", "nm", r.name); if (r.note) nm.appendChild(el("small", null, r.note)); tr.appendChild(nm);
    const df = el("td", "df");
    if (r.d_h_cm != null) df.appendChild(el("div", null, Number(r.d_h_cm).toFixed(1) + " cm"));
    if (r.d_alt_m != null && r.seq !== 0) df.appendChild(el("div", null, "\u0394alt " + Number(r.d_alt_m).toFixed(2) + " m"));
    if (r.d_h_cm == null && (r.d_alt_m == null || r.seq === 0)) df.textContent = "-";
    tr.appendChild(df);
    tr.appendChild(el("td", "st", r.ok === null ? "" : (r.ok ? "\u2705" : "\u274c")));
    tb.appendChild(tr);
    const tr2 = el("tr", "sub " + cls), co = el("td", "co"); co.colSpan = 4;
    if (d.compared) { const l1 = el("div"); l1.appendChild(el("i", null, "plan")); l1.appendChild(document.createTextNode(rbDescr(r.planned))); co.appendChild(l1); }
    const l2 = el("div", "pix"); l2.appendChild(el("i", null, "pixhawk")); l2.appendChild(document.createTextNode(rbDescr(r.readback))); co.appendChild(l2);
    tr2.appendChild(co); tb.appendChild(tr2);
  });
}
async function readMission(btn) {
  const old = btn.textContent; btn.disabled = true; btn.textContent = "Reading from the Pixhawk...";
  try {
    const r = await fetch("/api/vehicle/read_mission", { method: "POST" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const m = await r.json(); toast((m.data && m.data.text) || "done", true);
  } catch (e) { toast("Read failed: " + e.message); }
  btn.disabled = false; btn.textContent = old;
}
$("rb-read").addEventListener("click", () => readMission($("rb-read")));
$("veh-read-btn").addEventListener("click", async () => {
  await readMission($("veh-read-btn")); switchTab("survey"); if (S.step < 3) goStep(3); else renderPlanView();
});
async function ensureLegal() {
  if (S.legalAck) return true;
  const info = await getJSON("/api/legal", null);
  if (!info || info.acked || !(info.items || []).length) { S.legalAck = true; return true; }
  return new Promise((resolve) => {
    const m = $("legal"), list = $("legal-items"), ok = $("legal-ok");
    list.textContent = "";
    const boxes = info.items.map((txt) => {
      const lab = el("label", "check-item"), cb = el("input"); cb.type = "checkbox";
      lab.appendChild(cb); lab.appendChild(el("span", null, txt)); list.appendChild(lab); return cb;
    });
    const upd = () => { ok.disabled = !boxes.every((b) => b.checked); };
    boxes.forEach((b) => b.addEventListener("change", upd)); upd();
    const done = (v) => { m.hidden = true; ok.onclick = null; $("legal-cancel").onclick = null; resolve(v); };
    ok.onclick = async () => {
      try {
        const r = await fetch("/api/legal/ack", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ accepted: boxes.map((b) => b.checked) }) });
        if (!r.ok) throw new Error("HTTP " + r.status);
        S.legalAck = true; done(true);
      } catch (e) { toast("Checklist not saved: " + e.message); done(false); }
    };
    $("legal-cancel").onclick = () => done(false);
    m.hidden = false;
  });
}

/* ================= step 4: flight ================= */
function updateFlightInfo() {
  const box = $("flight-info"); if (!box) return;
  const t = S.tel || {}, st = S.status.mission_state || "-";
  box.textContent = "";
  box.appendChild(chip("State", stateLabel(st), st === "flying" ? "good" : (["rtl", "manual", "arming", "staged"].includes(st) ? "mid" : "")));
  if (t.mission_seq != null && t.mission_count) box.appendChild(chip("Waypoint", t.mission_seq + "/" + t.mission_count));
  box.appendChild(chip("Speed", fmt(t.groundspeed_ms, 1, " m/s")));   // altitude and battery are in the status bar
}
$("new-survey").addEventListener("click", () => {
  // the drone resends its recent alerts in every status: remember the newest one (drone clock) so they stay cleared
  const newest = S.alerts.reduce((m, a) => Math.max(m, a.ts || 0), Number(sstore("alerts.cleared")) || 0);
  sstore("alerts.cleared", String(newest));
  S.plan = null; S.corners = []; S.trail = []; mv && (mv.setTrail([]), mv.clearRoute(), mv.clearAlerts()); S.seen.clear(); S.alerts = [];
  $("flight-alerts").textContent = ""; $("flight-alerts-empty").hidden = false; $("alerts").textContent = ""; $("alerts-empty").hidden = false;
  goStep(1);
});
$("target-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const inp = $("target-input"); const text = inp.value.trim(); if (!text) return;
  inp.value = "";
  try { await send({ type: "target", text }); toast_ok("Target update sent"); }
  catch (err) { toast("Target update failed: " + err.message); }
});

/* ================= alerts ================= */
const ALERT_DOM_MAX = 200;   // newest alerts shown; every alert is still saved on the drone (alerts.jsonl)
const LEVEL = { person: "PERSON", target_match: "TARGET MATCH", heat: "🔥 HEAT (unconfirmed)" };
function alertCard(a, compact) {
  const card = el("div", "alert " + a.level);
  if (a.image_url) {
    const img = el("img"); img.src = a.image_url; img.alt = "Alert image"; img.loading = "lazy";
    img.addEventListener("click", () => openLightbox(a)); card.appendChild(img);
  }
  const body = el("div", "body");
  const head = el("div", "head");
  head.appendChild(el("span", "lvl " + a.level, LEVEL[a.level] || a.level));
  const score = a.c_final ?? a.c_match ?? a.c_avg ?? a.c_yolo;
  head.appendChild(el("span", "score", fmt(score === undefined ? null : score * 100, 0, "%")));
  head.appendChild(el("span", "track", "#" + a.track_id));
  body.appendChild(head);
  if (a.reason) body.appendChild(el("div", "reason", a.reason));
  const gps = (a.lat != null && a.lon != null) ? (a.lat.toFixed(6) + ", " + a.lon.toFixed(6) + (a.alt_rel_m != null ? " @" + a.alt_rel_m.toFixed(0) + "m" : "")) : "no GPS";
  body.appendChild(el("div", "meta", new Date(a.ts * 1000).toLocaleTimeString() + " | " + gps));
  if (a.lat != null && mv) {
    const b = el("button", "secondary go", "Show on map");
    b.addEventListener("click", () => { switchTab("survey"); goStep(4); followDrone = false; mv.flyTo({ lat: a.lat, lon: a.lon }, 19); });
    body.appendChild(b);
  }
  card.appendChild(body); return card;
}
function addAlert(a, live) {
  const key = [a.level, a.track_id, a.ts].join("|");
  if (S.seen.has(key) || (a.ts || 0) <= (Number(sstore("alerts.cleared")) || 0)) return;
  S.seen.add(key); S.alerts.push(a);
  $("alerts").insertBefore(alertCard(a), $("alerts").firstChild);
  const fa = $("flight-alerts"); fa.insertBefore(alertCard(a, true), fa.firstChild);
  while (fa.children.length > 8) fa.removeChild(fa.lastChild);
  const al = $("alerts");                                        // a long flight must not bloat the DOM / map
  while (al.children.length > ALERT_DOM_MAX) al.removeChild(al.lastChild);
  if (S.alerts.length > ALERT_DOM_MAX) { S.alerts.splice(0, S.alerts.length - ALERT_DOM_MAX); }
  $("alerts-empty").hidden = true; $("flight-alerts-empty").hidden = true;
  mv && mv.addAlert(a, (x) => { switchTab("alerts"); openLightbox(x); });
  // heat-only alerts are low priority (mostly sun-warmed ground / cars in daylight): listed and pinned, but no badge/buzz/pop-up
  if (live && a.level !== "heat" && !$("tab-alerts").classList.contains("active")) bumpBadge();
  if (live && a.level === "target_match" && navigator.vibrate) { try { navigator.vibrate([200, 100, 200]); } catch (e) {} }
  updateStepper();
}
let unseen = 0;
function bumpBadge() { unseen++; $("alert-count").textContent = String(unseen); $("alert-count").hidden = false; }
function openLightbox(a) {
  if (!a.image_url) return;
  $("lightbox-img").src = a.image_url;
  $("lightbox-cap").textContent = (LEVEL[a.level] || a.level) + " - " + (a.reason || "");
  $("lightbox").hidden = false;
}
$("lightbox").addEventListener("click", () => { $("lightbox").hidden = true; });

/* ================= chat ================= */
// Thread kept in memory and in sessionStorage, so it survives tab switches and reloads within the session.
// Our own lines are shown at once (pending) and matched to the server's echo, which goes to every phone.
const CHAT_KEY = "chat.v1", CHAT_MAX = 150, CHAT_WAIT_MS = 130000;
const chat = { items: [], waiting: false, waitTimer: null, pendingEcho: [], lastText: "" };
function sstore(k, v) { try { if (v === undefined) return sessionStorage.getItem(k); sessionStorage.setItem(k, v); } catch (e) { /* private mode */ } return null; }
// One id per phone tab, kept with the thread: the assistant's draft (area, target, ...) follows it across reconnects.
function newChatSid() {
  const v = "c-" + Array.from({ length: 16 }, () => "abcdefghijklmnopqrstuvwxyz0123456789"[Math.floor(Math.random() * 36)]).join("");
  sstore("chat.sid", v); return v;
}
let CHAT_SID = sstore("chat.sid") || newChatSid();
// The app restarted (reboot, ./start.sh): its drafts are gone, so the old thread on this phone is cleared too.
function checkBoot(id) {
  if (!id || id === sstore("chat.boot")) return;
  const first = sstore("chat.boot") === null && !chat.items.length;
  sstore("chat.boot", id);
  sstore("alerts.cleared", "0");          // a restarted app has no old alerts to hide
  if (first) return;
  chat.items = []; saveChat(); CHAT_SID = newChatSid(); renderChat();
}
function saveChat() { sstore(CHAT_KEY, JSON.stringify(chat.items.slice(-CHAT_MAX))); }
function hhmm(ts) { try { return new Date((ts || Date.now() / 1000) * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); } catch (e) { return ""; } }
const chatInput = $("chat-input");

function planSummary(p) {
  const plan = p.plan || {}, safety = p.safety || {};
  const area = plan.area_m2 ?? ((plan.mission && plan.mission.length_m) ? plan.mission.length_m * plan.mission.width_m : null);
  return { plan_id: p.plan_id, target: (p.target && p.target.description) || "", ok: !!safety.ok, area,
    lanes: plan.lane_count ?? Math.floor((plan.waypoints || []).length / 2), dur: plan.estimated_flight_s, alt: plan.altitude_m,
    bench: p.home_source === "bench", files: p.files && p.files["mission.waypoints"] ? { "mission.waypoints": p.files["mission.waypoints"] } : null };
}
function addChat(role, text, ts, extra) {
  chat.items.push(Object.assign({ role, text: String(text || ""), ts: ts || Date.now() / 1000 }, extra || {}));
  if (chat.items.length > CHAT_MAX) chat.items.splice(0, chat.items.length - CHAT_MAX);
  saveChat(); renderChat();
}
function onUserEcho(d) {
  const text = d.text || "", i = chat.pendingEcho.indexOf(text);
  if (i >= 0) {   // our own line coming back: mark it delivered instead of adding it twice
    chat.pendingEcho.splice(i, 1);
    const it = [...chat.items].reverse().find((x) => x.role === "user" && x.pending && x.text === text);
    if (it) { it.pending = false; if (d.ts) it.ts = d.ts; saveChat(); renderChat(); }
    return;
  }
  addChat("user", text, d.ts);   // typed on another phone
  setWaiting(true);
}
function setWaiting(v) {
  chat.waiting = v; clearTimeout(chat.waitTimer);
  if (v) chat.waitTimer = setTimeout(() => {
    if (!chat.waiting) return;
    chat.waiting = false; addChat("error", "No answer from Gemma within " + Math.round(CHAT_WAIT_MS / 1000) + " s.", null, { retry: chat.lastText });
  }, CHAT_WAIT_MS);
  renderChat();
}
function chatItemEl(it) {
  if (it.kind === "summary") return summaryCardEl(it);
  if (it.kind === "plan") return planCardEl(it);
  if (it.kind === "readback") return readbackCardEl(it);
  const m = el("div", "msg " + it.role + (it.pending ? " pending" : ""));
  m.appendChild(el("div", "bubble", it.text));
  if (it.role === "error" && it.retry) {
    const b = el("button", "secondary retry", "Retry"); b.type = "button";
    b.addEventListener("click", () => { it.retry = null; saveChat(); sendChat(b.dataset.text); });
    b.dataset.text = it.retry; m.appendChild(b);
  } else m.appendChild(el("div", "time", hhmm(it.ts)));
  return m;
}
function planCardEl(it) {
  const s = it.plan || {}, c = el("div", "plan-card");
  const head = el("div", "pc-head");
  head.appendChild(el("span", "pc-title", "Search plan"));
  head.appendChild(el("span", "tag " + (s.ok ? "ok" : "bad"), s.ok ? "Safety OK" : "BLOCKED"));
  c.appendChild(head);
  if (s.bench) c.appendChild(benchChip());
  c.appendChild(el("div", "pc-target", "Looking for: " + (s.target || "-")));
  const st = el("div", "pc-stats");
  [["Area", s.area == null ? "-" : G.fmtArea(s.area)], ["Lanes", s.lanes == null ? "-" : String(s.lanes)],
    ["Time", s.dur == null ? "-" : G.fmtDur(s.dur)], ["Alt", fmt(s.alt, 0, " m")]].forEach(([k, v]) => {
    const d = el("div"); d.appendChild(el("small", null, k)); d.appendChild(el("b", null, v)); st.appendChild(d);
  });
  c.appendChild(st);
  const b = el("button", "tonal pc-open", "Open in Preview"); b.type = "button";
  b.addEventListener("click", () => {
    if (!S.plan || S.plan.plan_id !== s.plan_id) { toast("This plan is no longer pending. Ask again or design a new route."); return; }
    switchTab("survey"); goStep(3);
  });
  c.appendChild(b);
  if (s.files) {
    const l = el("div", "pc-links"); Object.entries(s.files).forEach(([n, url]) => { const a = el("a", null, n + " (planned)"); a.href = url; a.setAttribute("download", ""); l.appendChild(a); });
    c.appendChild(l); c.appendChild(el("div", "pc-hint", "Mission Planner \u2192 Plan \u2192 Load WP File"));
  }
  c.appendChild(el("div", "time", hhmm(it.ts)));
  return c;
}
function readbackCardEl(it) {
  const s = it.info || {}, c = el("div", "plan-card");
  const head = el("div", "pc-head");
  head.appendChild(el("span", "pc-title", s.source === "upload" ? "Uploaded to the Pixhawk" : "Read from the Pixhawk"));
  if (s.compared) head.appendChild(el("span", "tag " + (s.ok ? "ok" : "bad"), s.ok ? "MATCH" : "DIFFERS"));
  c.appendChild(head);
  if (s.bench) c.appendChild(benchChip());
  c.appendChild(el("div", "pc-target", (s.compared ? (s.ok ? "\u2705 " : "\u274c ") : "") + s.text));
  const b = el("button", "tonal pc-open", "Show on map and table"); b.type = "button";
  b.addEventListener("click", () => { switchTab("survey"); goStep(3); const d = $("rb-details"); if (d && !d.hidden) d.open = true; });
  c.appendChild(b);
  const names = { "mission.waypoints": "mission.waypoints (planned)", "readback.waypoints": "readback.waypoints (on the Pixhawk)" };
  const l = el("div", "pc-links");
  Object.entries(s.files || {}).forEach(([n, url]) => { const a = el("a", null, names[n] || n); a.href = url; a.setAttribute("download", ""); l.appendChild(a); });
  c.appendChild(l); c.appendChild(el("div", "pc-hint", "Mission Planner \u2192 Plan \u2192 Load WP File"));
  c.appendChild(el("div", "time", hhmm(it.ts)));
  return c;
}
function renderChat() {
  const box = $("chat"); box.textContent = "";
  chat.items.forEach((it) => box.appendChild(chatItemEl(it)));
  if (chat.waiting) {
    const m = el("div", "msg assistant thinking"), bub = el("div", "bubble"), dots = el("span", "dots");
    dots.append(el("i"), el("i"), el("i")); bub.append(dots, document.createTextNode("Gemma is thinking…"));
    m.appendChild(bub); box.appendChild(m);
  }
  const empty = !chat.items.length && !chat.waiting;
  $("chat-empty").hidden = !empty; box.hidden = empty;
  updateSend(); scrollChat();
}
function scrollChat() { const box = $("chat"); requestAnimationFrame(() => { box.scrollTop = box.scrollHeight; }); }
function updateSend() { $("chat-send").disabled = chat.waiting || !chatInput.value.trim(); }
function autoGrow() { chatInput.style.height = "auto"; chatInput.style.height = Math.min(chatInput.scrollHeight + 2, 132) + "px"; }
async function sendChat(text) {
  text = String(text || "").trim(); if (!text || chat.waiting) return;
  chat.lastText = text;
  const viaWs = wsOpen();
  if (viaWs) chat.pendingEcho.push(text);
  addChat("user", text, null, { pending: viaWs });
  setWaiting(true);
  try { await send({ type: "chat", text, session: CHAT_SID }); }
  catch (err) {
    const i = chat.pendingEcho.indexOf(text); if (i >= 0) chat.pendingEcho.splice(i, 1);
    chat.waiting = false; clearTimeout(chat.waitTimer);
    addChat("error", "Not sent: " + err.message, null, { retry: text });
  }
}
function submitChat() {
  const text = chatInput.value.trim(); if (!text || chat.waiting) return;
  chatInput.value = ""; sstore("chat.draft", ""); autoGrow(); sendChat(text);
}
$("chat-form").addEventListener("submit", (e) => { e.preventDefault(); submitChat(); });
chatInput.addEventListener("input", () => { autoGrow(); updateSend(); sstore("chat.draft", chatInput.value); });
chatInput.addEventListener("keydown", (e) => {   // Enter sends, Shift+Enter = new line
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) { e.preventDefault(); submitChat(); }
});
chatInput.addEventListener("focus", () => setTimeout(scrollChat, 250));
document.querySelectorAll(".suggest").forEach((b) => b.addEventListener("click", () => {
  if (b.dataset.send) { sendChat(b.dataset.text); return; }
  chatInput.value = b.dataset.text; autoGrow(); updateSend(); chatInput.focus();
  try { chatInput.setSelectionRange(chatInput.value.length, chatInput.value.length); } catch (e) {}
}));
(function restoreChat() {
  try { const saved = JSON.parse(sstore(CHAT_KEY) || "[]"); if (Array.isArray(saved)) chat.items = saved.map((x) => Object.assign(x, { pending: false })); } catch (e) {}
  const draft = sstore("chat.draft"); if (draft) chatInput.value = draft;
  autoGrow(); renderChat();
})();

/* on-screen keyboard: size the page to the visual viewport so the composer and Abort stay above the keyboard */
let vvMax = 0;
function fitViewport() {
  const vv = window.visualViewport; if (!vv) return;
  const h = Math.round(vv.height); vvMax = Math.max(vvMax, h);
  document.documentElement.style.setProperty("--app-h", h + "px");
  const a = document.activeElement, typing = a && (a.tagName === "TEXTAREA" || (a.tagName === "INPUT" && a.type === "text"));
  document.documentElement.classList.toggle("kb-open", !!typing && h < vvMax * 0.8);
  if (vv.offsetTop > 0 && typing) window.scrollTo(0, 0);
  if (!$("panel-chat").hidden) scrollChat();
}
if (window.visualViewport) {
  visualViewport.addEventListener("resize", fitViewport); visualViewport.addEventListener("scroll", fitViewport);
  window.addEventListener("orientationchange", () => { vvMax = 0; setTimeout(fitViewport, 300); });
  document.addEventListener("focusin", () => setTimeout(fitViewport, 50)); document.addEventListener("focusout", () => setTimeout(fitViewport, 50));
  fitViewport();
}

/* ================= tabs / toast ================= */
function switchTab(t) {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.dataset.tab === t));
  $("panel-survey").hidden = t !== "survey"; $("panel-chat").hidden = t !== "chat"; $("panel-alerts").hidden = t !== "alerts";
  $("panel-vehicle").hidden = t !== "vehicle";
  if (t === "vehicle") { renderVehicleLive(); renderVehicleLog((S.status || {}).statustext); }
  if (t === "alerts") { unseen = 0; $("alert-count").hidden = true; }
  if (t === "survey") requestAnimationFrame(() => mv && mv.resize());
  if (t === "chat") scrollChat();
}
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
let toastTimer = null;
function toast(text, ok) {
  const t = $("toast"); t.textContent = text; t.hidden = false;
  t.classList.toggle("ok", !!ok);
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, ok ? 3000 : 6000);
}
const toast_ok = (t) => toast(t, true);
$("toast").addEventListener("click", () => { $("toast").hidden = true; });

/* ================= review (staged) + GO ================= */
const RV_ICON = { ok: "\u2713", warn: "!", fail: "\u2715", pending: "\u2026" };
let goTimer = null;
function goLabel() { return S.status && S.status.demo ? "GO \u00b7 start live search" : "GO"; }
function resetGo() { clearTimeout(goTimer); goTimer = null; const b = $("go-btn"); b.classList.remove("armed-confirm"); b.textContent = goLabel(); }
function renderReview(s) {
  const card = $("review-card"), staged = s.mission_state === "staged" && !["upload", "readonly"].includes(vmode());
  card.hidden = !staged;
  $("confirm-row").hidden = s.mission_state === "staged";
  if (!staged) { resetGo(); return; }
  const r = s.readiness, items = (r && r.items) || [];
  const list = $("rv-list"); list.textContent = "";
  if (!r) list.appendChild(el("li", "rv-item pending", "Checks starting\u2026"));
  // Problems first, in full; the passed checks fold into one tappable "N checks OK" line.
  const okN = items.filter((it) => it.status === "ok").length;
  items.forEach((it) => {
    const st = RV_ICON[it.status] ? it.status : "pending";
    if (st === "ok" && !S.rvOpen) return;
    const li = el("li", "rv-item " + st + (it.blocking && st === "fail" ? " blocking" : ""));
    li.appendChild(el("span", "rv-ic", RV_ICON[st]));
    const tx = el("div", "rv-tx"); tx.appendChild(el("b", null, it.label || it.id));
    if (it.detail) tx.appendChild(el("span", "rv-detail", it.detail));
    li.appendChild(tx); list.appendChild(li);
  });
  if (okN) {
    const li = el("li", "rv-item ok rv-fold");
    li.appendChild(el("span", "rv-ic", RV_ICON.ok));
    li.appendChild(el("div", "rv-tx", S.rvOpen ? "Hide passed checks" : okN + " checks OK (tap to show)"));
    li.addEventListener("click", () => { S.rvOpen = !S.rvOpen; renderReview(S.status); });
    list.appendChild(li);
  }
  const ts = r && r.ts ? new Date(r.ts * 1000) : null;
  $("rv-time").textContent = r ? (r.running ? "Checking\u2026" : (ts ? "Checked " + ts.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "")) : "";
  const ok = !!(r && r.go_allowed), g = $("go-btn");
  g.disabled = !ok;
  if (!goTimer) g.textContent = goLabel();
  if (!ok) resetGo();
  $("go-reason").textContent = ok ? "" : ((r && r.go_reason) || "Waiting for readiness checks\u2026");
  $("go-reason").hidden = ok;
}
$("rv-refresh").addEventListener("click", async () => {
  const b = $("rv-refresh"); b.disabled = true;
  try { const r = await fetch("/api/readiness/refresh", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }); if (!r.ok) throw new Error("HTTP " + r.status); }
  catch (err) { toast("Re-check failed: " + err.message); }
  b.disabled = false;
});
$("rv-cancel").addEventListener("click", async () => {
  try { await send({ type: "abort" }); } catch (err) { toast("Cancel failed: " + err.message); }
});
$("go-btn").addEventListener("click", async () => {
  const r = S.status.readiness, p = S.plan, b = $("go-btn");
  if (!(r && r.go_allowed) || S.status.mission_state !== "staged") return;
  if (!goTimer) {
    b.classList.add("armed-confirm"); b.textContent = S.status.demo ? "TAP AGAIN TO START THE SEARCH (no flight)" : "TAP AGAIN TO ARM AND TAKE OFF";
    goTimer = setTimeout(resetGo, 5000); return;
  }
  resetGo();
  const plan_id = (S.status.staged && S.status.staged.plan_id) || (S.status.readback_info && S.status.readback_info.plan_id) || (p && p.plan_id);
  try { await send({ type: "go", plan_id }); followDrone = true; switchTab("survey"); goStep(4); }
  catch (err) { addChat("error", "GO failed: " + err.message); toast("GO failed: " + err.message); }
});

/* ================= flight summary ================= */
function onFlightSummary(d, live) {
  if (!d || !d.text) return;
  const key = d.flight_dir || String(d.ts || d.text.slice(0, 40));
  const imgs = (d.images || []).length, prev = S.lastSummary;
  if (prev && S.sumKey === key && (prev.images || []).length >= imgs && prev.text === d.text) return;
  S.sumKey = key; S.lastSummary = d;
  const seen = Number(store("sumSeenTs") || 0), fresh = live || !d.ts || d.ts > seen;
  const old = chat.items.find((it) => it.kind === "summary" && it.key === key);
  if (old) { old.summary = d; saveChat(); renderChat(); }
  else if (fresh) addChat("plan", "", null, { kind: "summary", key, summary: d });
  if (fresh && d.ts) store("sumSeenTs", String(d.ts));
  renderFlightSummary();
  if (fresh && !old) toast_ok("Flight summary ready");
  if (mv) (d.images || []).forEach((im, i) => mv.addAlert({ level: "target_match", track_id: "sum" + key + i, ts: i, lat: im.lat, lon: im.lon }, () => { switchTab("survey"); goStep(4); }));
}
function summaryBody(d, cls) {
  const c = el("div", cls || "summary-card");
  const head = el("div", "pc-head");
  head.appendChild(el("span", "pc-title", "Flight summary"));
  if (d.source === "template") head.appendChild(el("span", "tag", "(template)"));
  c.appendChild(head); c.appendChild(el("div", "sum-text", d.text));
  const ims = d.images || [];
  if (ims.length) {
    const g = el("div", "sum-grid");
    ims.forEach((im) => {
      const f = el("figure", "sum-img"), a = el("a"); a.href = im.url; a.target = "_blank"; a.rel = "noopener";
      const i = el("img"); i.src = im.url; i.loading = "lazy"; i.alt = im.caption || "alert image"; a.appendChild(i); f.appendChild(a);
      const cap = (im.caption || "") + (im.conf != null ? " (" + Math.round(im.conf * 100) + "%)" : "");
      if (cap.trim()) f.appendChild(el("figcaption", null, cap));
      g.appendChild(f);
    });
    c.appendChild(g);
  }
  return c;
}
function summaryCardEl(it) { const c = summaryBody(it.summary || {}, "plan-card summary-card"); c.appendChild(el("div", "time", hhmm(it.ts))); return c; }
function renderFlightSummary() {
  const box = $("flight-summary"); if (!box) return;
  box.textContent = ""; box.hidden = !S.lastSummary;
  if (S.lastSummary) box.appendChild(summaryBody(S.lastSummary));
}

/* ================= abort (always visible, in-page two-step) ================= */
let abortTimer = null, abortLeft = 0;
const ABORT_LABEL = "ABORT / RTL", armedLabel = (n) => "TAP AGAIN TO CONFIRM RTL \u00b7 " + n;
function resetAbort() { clearInterval(abortTimer); abortTimer = null; const b = $("abort-btn"); b.classList.remove("confirming"); b.textContent = ABORT_LABEL; }
$("abort-btn").addEventListener("click", async () => {
  const b = $("abort-btn");
  if (!abortTimer) {
    abortLeft = 4; b.classList.remove("sent"); b.classList.add("confirming"); b.textContent = armedLabel(abortLeft);
    abortTimer = setInterval(() => { abortLeft--; if (abortLeft <= 0) resetAbort(); else b.textContent = armedLabel(abortLeft); }, 1000);
    return;
  }
  resetAbort(); b.textContent = "RTL SENT"; b.classList.add("sent");
  setTimeout(() => { if (!abortTimer) { b.textContent = ABORT_LABEL; b.classList.remove("sent"); } }, 2500);
  try { await send({ type: "abort" }); }
  catch (err) { toast("ABORT NOT SENT: " + err.message + " - use the RC transmitter!"); b.textContent = ABORT_LABEL; b.classList.remove("sent"); }
});

/* ================= map controls ================= */
const MODE_LABEL = { vector: "Map", satellite: "Sat", grid: "Grid" };
function setupMapControls() {
  const btn = $("layer-btn");
  const modes = mv.availableModes();
  btn.textContent = MODE_LABEL[mv.mode];
  btn.hidden = modes.length < 2;
  btn.addEventListener("click", () => {
    const i = modes.indexOf(mv.mode); const next = modes[(i + 1) % modes.length];
    mv.setMode(next); btn.textContent = MODE_LABEL[next]; store("mode", next); noMapBanner();
  });
  const saved = store("mode"); if (saved && modes.includes(saved) && saved !== mv.mode) { mv.setMode(saved); btn.textContent = MODE_LABEL[saved]; }
  $("center-btn").addEventListener("click", () => {
    followDrone = true;
    const t = S.tel, p = (t && t.lat != null) ? { lat: t.lat, lon: t.lon } : homePoint();
    if (p) mv.flyTo(p, Math.max(mv.map.getZoom(), 17)); else toast("No drone or home position yet.");
  });
  mv.map.on("dragstart", () => { followDrone = false; });
  noMapBanner();
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => mv.resize()).observe($("map-wrap"));
}
/* system-check: WebGL/map failure degrades to the no-map layout instead of crashing the tab */
function mapUnavailable(msg) {
  $("nomap").hidden = false; $("nomap").textContent = msg;
  $("layer-btn").hidden = true; $("center-btn").hidden = true;
  $("map").classList.add("failed");
}
function noMapBanner() {
  const n = $("nomap");
  if (!S.info.has_map) { n.hidden = false; n.textContent = "No map downloaded. Showing a plain grid; corners are still placed in lat/lon."; }
  else if (mv.mode === "grid") { n.hidden = false; n.textContent = "Grid view (no basemap)."; }
  else n.hidden = true;
}

/* ================= system-check: status-bar warnings + System panel ================= */
function renderSysNotes(s) {
  // One slim line ("Camera offline · Pixhawk not connected"); the raw backend text only after a tap on it.
  const v = s.vision || {}, cam = v.camera_problem || "", veh = s.vehicle_hint || "";
  $("sb-cam-item").hidden = !cam; $("sb-veh-item").hidden = !veh;
  document.body.classList.toggle("sb-flags", !!(cam || veh));
  const parts = [], details = [];
  if (cam) { parts.push("Camera offline"); details.push(cam); }
  if (veh) { parts.push("No Pixhawk link"); details.push(veh); }
  renderSlims($("sysnote"), parts.length ? [["warn", parts.join(" \u00b7 "), details]] : []);
}
const SYS_ICON = { ok: "\u2705", warn: "\u26a0\ufe0f", fail: "\u274c", skip: "\u23ed\ufe0f" };
function renderSystemReport(r) {
  const box = $("sys-report"); box.textContent = "";
  const sm = r.summary || {};
  box.appendChild(notice(sm.fail ? "err" : (sm.warn ? "warn" : "okc"),
    (sm.fail ? "PROBLEMS FOUND" : (sm.warn ? "OK with warnings" : "ALL GOOD")) + ": " + sm.ok + " ok, " + sm.warn + " warnings, " + sm.fail + " failed, "
    + sm.skip + " skipped (" + r.duration_s + " s, " + r.mode + ")"));
  // problems first, then the rest; one compact row per check (status dot, "group · name", detail)
  const order = { fail: 0, warn: 1, skip: 2, ok: 3 };
  const list = el("div", "sys-list");
  [...(r.checks || [])].sort((a, b) => (order[a.status] ?? 9) - (order[b.status] ?? 9)).forEach((c) => {
    const row = el("div", "sys-row " + (c.status || ""));
    row.appendChild(el("span", "sys-dot", SYS_ICON[c.status] ? "" : "?")).setAttribute("aria-label", c.status || "");
    const txt = el("div", "sys-txt");
    txt.appendChild(el("b", null, c.group + " \u00b7 " + c.name));
    txt.appendChild(el("span", null, (c.detail || "") + (c.via ? " [" + c.via + "]" : "")));
    row.appendChild(txt); list.appendChild(row);
  });
  box.appendChild(list);
  const fixes = (r.checks || []).filter((c) => (c.status === "fail" || c.status === "warn") && c.fix);
  if (fixes.length) {
    box.appendChild(el("h3", null, "What to do")); const ul = el("ul");
    fixes.forEach((c) => ul.appendChild(el("li", null, c.group + "/" + c.name + ": " + c.fix))); box.appendChild(ul);
  }
}
async function runSystemCheck(quick) {
  const bs = [$("sys-check-btn"), $("sys-quick-btn")]; bs.forEach((b) => { b.disabled = true; });
  $("sys-check-btn").textContent = "Checking (up to 1 min)...";
  try {
    const r = await fetch("/api/system/check?quick=" + (quick ? "true" : "false"));
    if (!r.ok) throw new Error("HTTP " + r.status);
    renderSystemReport(await r.json());
  } catch (e) { toast("System check failed: " + e.message); }
  bs.forEach((b) => { b.disabled = false; }); $("sys-check-btn").textContent = "Full check";
}
$("sys-check-btn").addEventListener("click", () => runSystemCheck(false));
$("sys-quick-btn").addEventListener("click", () => runSystemCheck(true));

/* ================= boot ================= */
async function getJSON(url, fallback) {
  try { const r = await fetch(url); if (!r.ok) throw new Error(r.status); return await r.json(); } catch (e) { return fallback; }
}
async function boot() {
  applyTheme(curTheme(), false);
  S.limits = Object.assign({}, DEFAULT_LIMITS, await getJSON("/api/limits", {}));
  S.info = await getJSON("/api/map/info", { layers: [], has_map: false });
  if (S.info.notice) { $("airspace").hidden = false; $("airspace-text").textContent = S.info.notice; }
  mv = new MapView($("map"), S.info, document.documentElement.dataset.theme, {
    onTap: addCorner,
    onCornerMove: (i, p, done, pts) => { S.corners = pts; refreshArea(); },
    onDegrade: (why) => mapUnavailable("Map stopped (" + why + "). Showing no map; use Paste coordinates to enter corners."),
  });
  if (mv.failed) {
    mapUnavailable("Map needs WebGL, which this browser does not provide. Use Paste coordinates to enter corners.");
  } else {
    setupMapControls();
    mv.map.on("load", () => { mv.setEditable(true); updateMapContext(); mv.setCorners(S.corners); });
  }
  setupQuestions();
  $("dronepos-btn").disabled = true;
  goStep(1);
  connect();
}
// test hook (Playwright page edge tests read/inject state through this)
window.__dbg = { S, applyStatus, addAlert, goStep, get mv() { return mv; } };
boot();
