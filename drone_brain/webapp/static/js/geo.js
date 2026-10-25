// Small flat-earth geometry helpers (same maths as survey.py). No dependencies.
export const R = 6378137.0;
const rad = (d) => (d * Math.PI) / 180;
const deg = (r) => (r * 180) / Math.PI;

export function toLocal(p, o) {           // -> {n, e} metres from origin o
  return { n: rad(p.lat - o.lat) * R, e: rad(p.lon - o.lon) * R * Math.cos(rad(o.lat)) };
}
export function toGeo(n, e, o) {
  return { lat: o.lat + deg(n / R), lon: o.lon + deg(e / (R * Math.cos(rad(o.lat)))) };
}
export function distM(a, b) { const l = toLocal(b, a); return Math.hypot(l.n, l.e); }

export function areaM2(corners) {
  if (corners.length < 3) return 0;
  const pts = corners.map((c) => toLocal(c, corners[0]));
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    a += p.e * q.n - q.e * p.n;
  }
  return Math.abs(a) / 2;
}

function cross(o, a, b) { return (a.e - o.e) * (b.n - o.n) - (a.n - o.n) * (b.e - o.e); }
export function selfIntersects(corners) {
  if (corners.length < 4) return false;
  const p = corners.map((c) => toLocal(c, corners[0]));
  const n = p.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (j === i || (j + 1) % n === i || (i + 1) % n === j) continue;
      const a = p[i], b = p[(i + 1) % n], c = p[j], d = p[(j + 1) % n];
      const d1 = cross(c, d, a), d2 = cross(c, d, b), d3 = cross(a, b, c), d4 = cross(a, b, d);
      if ((d1 > 0) !== (d2 > 0) && (d3 > 0) !== (d4 > 0) && d1 * d2 !== 0 && d3 * d4 !== 0) return true;
    }
  }
  return false;
}

export function centroid(pts) {
  const n = pts.length || 1;
  return { lat: pts.reduce((s, p) => s + p.lat, 0) / n, lon: pts.reduce((s, p) => s + p.lon, 0) / n };
}

export function circlePolygon(c, radiusM, steps = 72) {
  const ring = [];
  for (let i = 0; i <= steps; i++) {
    const b = (i / steps) * 2 * Math.PI;
    const g = toGeo(radiusM * Math.cos(b), radiusM * Math.sin(b), c);
    ring.push([g.lon, g.lat]);
  }
  return ring;
}

// Camera maths (matches survey.py)
export function footprintW(h, hfovDeg) { return 2 * h * Math.tan(rad(hfovDeg) / 2); }
export function footprintL(h, vfovDeg) { return 2 * h * Math.tan(rad(vfovDeg) / 2); }
export function gsd(h, L) { return (h * L.sensor_width_mm) / (L.focal_length_mm * L.image_width_px); }
export function personPx(h, L) { return L.person_height_m / gsd(h, L); }
export function maxAltForPx(px, L) { return ((L.person_height_m / px) * L.focal_length_mm * L.image_width_px) / L.sensor_width_mm; }

// Rectangle mission (chat planner) -> corner lat/lon, for drawing.
export function rectCorners(m, home) {
  const b = rad(m.direction_deg || 0);
  const an = Math.cos(b), ae = Math.sin(b), cn = Math.cos(b + Math.PI / 2), ce = Math.sin(b + Math.PI / 2);
  const o = m.centered ? -m.length_m / 2 : (m.offset_m || 0), hw = m.width_m / 2;   // centered: straddles home
  return [[o, -hw], [o, hw], [o + m.length_m, hw], [o + m.length_m, -hw]]
    .map(([u, v]) => toGeo(u * an + v * cn, u * ae + v * ce, home));
}

export function fmtArea(m2) {
  if (m2 >= 10000) return (m2 / 10000).toFixed(2) + " ha";
  return Math.round(m2) + " m²";
}
export function fmtDur(s) {
  if (s == null) return "-";
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return m + ":" + String(r).padStart(2, "0") + " min";
}
