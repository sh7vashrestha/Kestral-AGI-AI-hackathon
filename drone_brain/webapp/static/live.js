(() => {
'use strict';
const $ = (id) => document.getElementById(id);
const DEF = {view: 'both', fps: 10, w: 960, palette: 'inferno', range: 'auto', min: 15, max: 40, alpha: 50, boxes: false};
let S = Object.assign({}, DEF);
try { Object.assign(S, JSON.parse(localStorage.getItem('live.settings') || '{}')); } catch (e) { /* storage blocked */ }
const save = () => { try { localStorage.setItem('live.settings', JSON.stringify(S)); } catch (e) { /* ignore */ } };

let stats = null, probe = null, serverDown = false;

// ---------------------------------------------------------------- streams with auto-reconnect
class Feed {
  constructor(img, urlFn) { this.img = img; this.urlFn = urlFn; this.n = 0; this.backoff = 1000; this.timer = null; this.ok = false; this.on = false;
    img.addEventListener('load', () => { this.ok = true; this.backoff = 1000; });
    img.addEventListener('error', () => { if (!this.on) return; this.ok = false; this.retry(); }); }
  start() { this.on = true; this.connect(); }
  stop() { this.on = false; clearTimeout(this.timer); this.timer = null; this.ok = false; this.img.removeAttribute('src'); }
  connect() { clearTimeout(this.timer); this.timer = null; this.ok = false; this.img.src = this.urlFn() + '&_=' + (++this.n); }
  retry() { if (this.timer) return; this.timer = setTimeout(() => { this.timer = null; if (this.on) this.connect(); }, this.backoff);
    this.backoff = Math.min(this.backoff * 2, 10000); }
  watchdog() { if (this.on && !this.ok && !this.timer) this.retry(); }
}
const rangeQ = () => (S.range === 'fixed' && S.max > S.min) ? `&min=${S.min}&max=${S.max}` : '';
const feedRgb = new Feed($('iRgb'), () => `/stream/rgb.mjpg?w=${S.w}&fps=${S.fps}&boxes=${S.boxes ? 1 : 0}`);
const feedTh = new Feed($('iTh'), () => S.view === 'overlay'
  ? `/stream/overlay.mjpg?w=${S.w}&fps=${S.fps}&palette=${S.palette}&alpha=${(S.alpha / 100).toFixed(2)}${rangeQ()}`
  : `/stream/thermal.mjpg?fps=${S.fps}&palette=${S.palette}&boxes=${S.boxes ? 1 : 0}${rangeQ()}`);

let reapplyT = null;
function reapply() { clearTimeout(reapplyT); reapplyT = setTimeout(apply, 150); }
function apply() {
  const ov = S.view === 'overlay';
  $('grid').classList.toggle('overlay', ov);
  $('thTitle').textContent = ov ? 'Thermal overlay on RGB' : 'Thermal';
  $('alphaBox').hidden = !ov;
  $('fixedBox').hidden = S.range !== 'fixed';
  $('thHint').textContent = ov ? 'Overlay uses the current thermal-to-RGB calibration. Switch to Both to probe temperatures.'
                               : 'Tap the thermal image to read the temperature at that point.';
  $('thView').style.cursor = ov ? 'default' : 'crosshair';
  if (ov) { clearProbe(); feedRgb.stop(); } else if (!feedRgb.on) feedRgb.start(); else feedRgb.connect();
  $('sBar').src = `/api/live/palette.png?name=${S.palette}`;
  feedTh.start();
  syncControls(); save();
}

// ---------------------------------------------------------------- controls
function seg(id, key, cast) {
  $(id).addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return;
    S[key] = cast ? cast(b.dataset.v) : b.dataset.v; reapply(); });
}
function syncSeg(id, val) { $(id).querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === String(val))); }
function syncControls() {
  syncSeg('viewSeg', S.view); syncSeg('fpsSeg', S.fps); syncSeg('sizeSeg', S.w); syncSeg('rangeSeg', S.range);
  $('palette').value = S.palette; $('rMin').value = S.min; $('rMax').value = S.max;
  $('alpha').value = S.alpha; $('alphaOut').textContent = S.alpha + '%'; $('boxes').checked = S.boxes;
}
seg('viewSeg', 'view'); seg('fpsSeg', 'fps', Number); seg('sizeSeg', 'w', Number); seg('rangeSeg', 'range');
$('palette').addEventListener('change', (e) => { S.palette = e.target.value; reapply(); });
for (const id of ['rMin', 'rMax']) $(id).addEventListener('change', () => {
  const a = parseFloat($('rMin').value), b = parseFloat($('rMax').value);
  if (isFinite(a) && isFinite(b) && b > a) { S.min = a; S.max = b; reapply(); } });
$('alpha').addEventListener('input', () => { $('alphaOut').textContent = $('alpha').value + '%'; });
$('alpha').addEventListener('change', () => { S.alpha = Number($('alpha').value); reapply(); });
$('boxes').addEventListener('change', () => { S.boxes = $('boxes').checked; reapply(); });
document.querySelectorAll('.fs').forEach((b) => b.addEventListener('click', () => {
  const el = $(b.dataset.fs);
  if (document.fullscreenElement) document.exitFullscreen(); else if (el.requestFullscreen) el.requestFullscreen(); }));

$('snap').addEventListener('click', async () => {
  const btn = $('snap'), msg = $('snapMsg'); btn.disabled = true; msg.className = 'snapmsg muted'; msg.textContent = 'Saving...';
  try {
    const r = await fetch('/api/live/snapshot', {method: 'POST'});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    msg.className = 'snapmsg ok'; msg.textContent = 'Saved ' + d.dir + '  (' + Object.keys(d.files).join(', ') + ')';
  } catch (e) { msg.className = 'snapmsg err'; msg.textContent = 'Snapshot failed: ' + e.message; }
  btn.disabled = false;
});

// ---------------------------------------------------------------- orientation (rotate / flip preview + save as default)
let orientation = null;
const oMsg = (t, cls) => { const n = $('oNote'); n.textContent = t; n.className = cls || 'muted'; };
async function postOrient(target, patch) {
  try {
    const r = await fetch('/api/live/orientation', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(Object.assign({target}, patch))});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    orientation = d; renderOrient(); oMsg('Previewing. Press "Save as default" to keep it.');
    feedRgb.connect(); feedTh.connect();              // pick up the new orientation right away (overlay included)
  } catch (e) { oMsg('Could not change: ' + e.message, 'err'); }
}
function bindOrient(target, rotId, hId, vId) {
  $(rotId).addEventListener('click', () => { const o = orientation[target]; postOrient(target, {rotate: (o.rotate + 90) % 360}); });
  $(hId).addEventListener('click', () => { const o = orientation[target]; postOrient(target, {flip: o.flip === 'horizontal' ? 'none' : 'horizontal'}); });
  $(vId).addEventListener('click', () => { const o = orientation[target]; postOrient(target, {flip: o.flip === 'vertical' ? 'none' : 'vertical'}); });
}
bindOrient('camera', 'camRot', 'camFlipH', 'camFlipV'); bindOrient('thermal', 'thRot', 'thFlipH', 'thFlipV');
$('oSave').addEventListener('click', async () => {
  $('oSave').disabled = true;
  try {
    const r = await fetch('/api/live/orientation/save', {method: 'POST'});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    orientation = d.orientation; renderOrient();
    oMsg('Saved to .env. Restart main.py to apply it in the flight pipeline.', 'ok');
  } catch (e) { oMsg('Save failed: ' + e.message, 'err'); }
  renderOrient();
});
function renderOrient() {
  const o = orientation; if (!o) return;
  const ed = !!o.editable;
  for (const [t, lbl, ids] of [['camera', 'oCamLbl', ['camRot', 'camFlipH', 'camFlipV']], ['thermal', 'oThLbl', ['thRot', 'thFlipH', 'thFlipV']]]) {
    const c = o[t];
    $(lbl).textContent = c.label + (c.dirty ? ' (not saved)' : ' (saved)') + (t === 'camera' && c.out_size ? ' · ' + c.out_size[0] + '×' + c.out_size[1] : '');
    for (const id of ids) $(id).disabled = !ed;
  }
  $('oSave').disabled = !ed || !(o.camera.dirty || o.thermal.dirty);
  if (!ed) oMsg('Read-only here (main app is running). Preview with scripts/live_view.py, save, then restart main.py.');
}

// ---------------------------------------------------------------- thermal probe
function place(el, x, y, text) {
  el.hidden = false; el.style.left = (x * 100) + '%'; el.style.top = (y * 100) + '%';
  el.classList.toggle('flip', x > 0.7);
  el.querySelector('b').textContent = text;
}
function clearProbe() { probe = null; $('probe').hidden = true; }
$('probe').addEventListener('click', (e) => { e.stopPropagation(); clearProbe(); });
$('thView').addEventListener('click', (e) => {
  if (S.view === 'overlay') return;
  const r = $('iTh').getBoundingClientRect();
  probe = {x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))};
  place($('probe'), probe.x, probe.y, '...'); readProbe();
});
async function readProbe() {
  if (!probe) return;
  const p = probe;
  try {
    const r = await fetch(`/api/live/thermal_point?x=${p.x.toFixed(4)}&y=${p.y.toFixed(4)}`);
    if (!r.ok) throw new Error();
    const d = await r.json();
    if (probe === p) place($('probe'), p.x, p.y, d.c.toFixed(1) + ' °C');
  } catch (e) { if (probe === p) place($('probe'), p.x, p.y, 'n/a'); }
}

// ---------------------------------------------------------------- status polling
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
function feedUI(dot, metaEl, msgEl, d, name, fps, clientOk) {
  const st = d.state || 'unavailable', age = d.age_s;
  let cls = 'ok';
  if (st === 'unavailable') cls = 'bad'; else if (st === 'disabled') cls = ''; else if (st !== 'ok' || (age != null && age > 2)) cls = 'warn';
  dot.className = 'dot ' + cls;
  const bits = [];
  if (st === 'ok') { bits.push((fps || 0).toFixed(1) + ' fps'); if (age != null) bits.push(age.toFixed(1) + ' s old'); if (d.w) bits.push(d.w + '×' + d.h); }
  else bits.push(st);
  metaEl.textContent = bits.join(' · ');
  const show = st !== 'ok';
  msgEl.hidden = !show;
  if (show) {
    msgEl.replaceChildren();
    const title = {unavailable: name + ' unavailable', disabled: name + ' disabled', starting: name + ' starting...', stalled: name + ' stalled'}[st] || name + ' ' + st;
    msgEl.append(el('strong', title));
    if (d.error) msgEl.append(el('span', d.error));
    if (d.hint) msgEl.append(/sudo|run /.test(d.hint) ? el('code', d.hint.replace(/^.*try: /, '')) : el('span', d.hint));
  }
}
function render(d) {
  const ov = S.view === 'overlay';
  const rgb = d.rgb || {}, th = d.thermal || {};
  if (d.mission_url) { const a = $('back'); a.href = d.mission_url; a.hidden = false; }
  $('boxBox').hidden = !d.has_boxes;
  if (!$('palette').options.length) {
    for (const p of d.palettes || []) { const o = el('option', p[0].toUpperCase() + p.slice(1)); o.value = p; $('palette').append(o); }
    $('palette').value = S.palette;
  }
  feedUI($('dRgb'), $('mRgb'), $('xRgb'), rgb, 'RGB camera', rgb.fps || rgb.source_fps);
  const thFps = ov ? (d.overlay || {}).fps : th.fps;
  if (ov && rgb.state && rgb.state !== 'ok') feedUI($('dTh'), $('mTh'), $('xTh'), rgb, 'RGB camera', 0);
  else feedUI($('dTh'), $('mTh'), $('xTh'), ov ? Object.assign({}, th, {w: rgb.w, h: rgb.h}) : th, 'Thermal camera', thFps || th.source_fps);
  // scale bar
  const rng = (S.range === 'fixed' && S.max > S.min) ? [S.min, S.max] : th.range;
  $('scale').style.visibility = rng ? 'visible' : 'hidden';
  if (rng) { $('sMin').textContent = rng[0].toFixed(0) + ' °C'; $('sMax').textContent = rng[1].toFixed(0) + ' °C'; }
  // hottest point
  const hp = th.hottest;
  if (hp && th.state === 'ok') {
    const pos = ov ? hp.overlay : hp;
    if (pos && pos.x >= 0 && pos.x <= 1 && pos.y >= 0 && pos.y <= 1) place($('hotMark'), pos.x, pos.y, hp.c.toFixed(1) + ' °C'); else $('hotMark').hidden = true;
  } else $('hotMark').hidden = true;
  if (th.state !== 'ok') { $('probe').hidden = true; } else if (probe) $('probe').hidden = false;
  // numbers
  const ts = $('thStats'); ts.replaceChildren();
  if (th.min_c != null) {
    for (const [k, v] of [['min', th.min_c], ['median', th.median_c], ['max', th.max_c]]) { const s = el('span', k + ' '); s.append(el('b', v.toFixed(1) + '°')); ts.append(s); }
    const s = el('span', 'hottest '); s.append(el('b', th.hottest.c.toFixed(1) + '° @ ' + th.hottest.px.join(',')));
    ts.append(s);
  }
  if (d.orientation) { orientation = d.orientation; renderOrient(); }
  const sys = d.system || {};
  const parts = [];
  if (rgb.reconnects) parts.push('RGB reconnects ' + rgb.reconnects + (rgb.next_retry_s ? ' (retry in ' + rgb.next_retry_s + ' s)' : ''));
  if (th.reconnects) parts.push('thermal reconnects ' + th.reconnects);
  if (rgb.brightness != null) parts.push('RGB brightness ' + rgb.brightness);
  $('foot').textContent = parts.join(' · ');
  $('sysline').textContent = [sys.cpu_pct != null ? 'CPU ' + sys.cpu_pct + '%' : '', sys.ram_used_mb != null ? 'RAM ' + sys.ram_used_mb + '/' + sys.ram_total_mb + ' MB' : '',
    d.mode === 'pipeline' ? 'via main app' : ''].filter(Boolean).join(' · ');
}
async function poll() {
  try {
    const r = await fetch('/api/live/stats', {cache: 'no-store'});
    stats = await r.json(); serverDown = false; render(stats);
    readProbe();
  } catch (e) {
    serverDown = true;
    for (const [d, m] of [['dRgb', 'mRgb'], ['dTh', 'mTh']]) { $(d).className = 'dot bad'; $(m).textContent = 'server unreachable'; }
  }
  feedRgb.watchdog(); feedTh.watchdog();
  setTimeout(poll, 800);
}
// ---------------------------------------------------------------- search demo (scripts/demo_search.py only)
let searchOn = false, bestTs = 0;
function renderSearch(d) {
  const title = $('sTitle'), list = $('sList');
  const people = d.people || [];
  const best = people.find((p) => p.best);
  if (d.demo && !d.searching) { title.className = 'stitle'; title.textContent = (d.target ? `Target: "${d.target}". ` : '') + (d.hint || ''); }
  else if (!d.target) { title.className = 'stitle'; title.textContent = people.length ? `${people.length} ${people.length === 1 ? 'person' : 'people'} in view. Type who to look for.` : 'No one in view. Type who to look for.'; }
  else if (best && best.match) { title.className = 'stitle match'; title.textContent = `Found: "${d.target}" (${Math.round(best.score * 100)}%)`; }
  else if (best) { title.className = 'stitle close'; title.textContent = `Searching "${d.target}": closest ${Math.round(best.score * 100)}% (needs ${Math.round(d.cutoff * 100)}%)`; }
  else { title.className = 'stitle'; title.textContent = `Searching "${d.target}": no one in view`; }
  $('sMeta').textContent = (d.loop_ms ? `${d.loop_ms} ms per scan` : '') + (best && best.temp_c != null ? ` · best match ${best.temp_c.toFixed(1)} °C${best.warm ? ' (warm: live person)' : ' (not warm)'}` : '');
  list.replaceChildren();
  if (d.target) [...people].sort((a, b) => b.score - a.score).slice(0, 6).forEach((p, i) => {
    const li = el('li', `${Math.round(p.score * 100)}%` + (p.temp_c != null ? ` · ${p.temp_c.toFixed(1)} °C` : '') + (p.match ? ' · MATCH' : ''));
    if (p.best) li.className = 'best'; list.append(li);
  });
  const img = $('sBest');
  if (d.has_best_image && d.best && d.best.ts !== bestTs) { bestTs = d.best.ts; img.src = '/api/search/best.jpg?_=' + Date.now(); }
  img.hidden = !(d.target && d.has_best_image);
}
async function pollSearch() {
  try {
    const r = await fetch('/api/search', {cache: 'no-store'});
    if (!r.ok) return;                                    // the normal live view: no search here
    if (!searchOn) { searchOn = true; $('search').hidden = false; if (!S.boxes) { S.boxes = true; save(); syncControls(); apply(); } }
    renderSearch(await r.json());
  } catch (e) { /* server down: the main poll shows it */ }
  if (searchOn) setTimeout(pollSearch, 500);
}
async function sendSearch(text) {
  try {
    const r = await fetch('/api/search', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})});
    bestTs = 0; $('sBest').hidden = true; renderSearch(await r.json());
  } catch (e) { $('sMeta').textContent = 'search failed: ' + e.message; }
}
$('sForm').addEventListener('submit', (e) => { e.preventDefault(); sendSearch($('sText').value.trim()); $('sText').blur(); });
$('sClear').addEventListener('click', () => { $('sText').value = ''; sendSearch(''); });
pollSearch();

window.addEventListener('pagehide', () => { feedRgb.stop(); feedTh.stop(); });
syncControls(); apply(); poll();
})();
