// Offline MapLibre styles. No glyphs/sprites are referenced (no text layers), so nothing is fetched
// from the network; labels such as the grid coordinates are DOM elements.

// Muted basemaps so the overlays carry the meaning: route / corners in the UI accent, geofence amber, flown trail teal.
export const THEMES = {
  dark: {
    bg: "#121417", land: "#16191d", park: "#18221c", pitch: "#1b2b21", pitchLine: "#2f4a39", water: "#152235", building: "#22262c",
    buildingLine: "#2c3138", road: "#2a2f37", roadMajor: "#363c46", path: "#262b32", grid: "#2e333b",
    poly: "#5b9cf5", route: "#6aa6f8", fenceLine: "#d29922", trail: "#2fbfa8", casing: "#0b0d10", rb: "#f4f6f9",
  },
  light: {
    bg: "#eef0f3", land: "#eef0f3", park: "#dfe9dc", pitch: "#d3e4cf", pitchLine: "#a9c6a3", water: "#cddcec", building: "#e1e3e7",
    buildingLine: "#cdd1d7", road: "#ffffff", roadMajor: "#fbfbfc", path: "#d6d9de", grid: "#c9ced6",
    poly: "#1f62d0", route: "#1f62d0", fenceLine: "#b07800", trail: "#0f8a78", casing: "#ffffff", rb: "#15181c",
  },
};
THEMES.contrast = THEMES.light;   // old saved theme name


/** mode: "vector" | "satellite" | "grid" */
export function buildStyle(theme, mode, info) {
  const t = THEMES[theme] || THEMES.dark;
  const sources = {};
  const layers = [{ id: "bg", type: "background", paint: { "background-color": t.bg } }];
  const vec = (info.layers || []).find((l) => l.type === "pmtiles" && l.kind === "vector");
  const sat = (info.layers || []).find((l) => l.id === "satellite");

  if (mode === "satellite" && sat) {
    sources.sat = { type: "raster", tiles: sat.tiles.map((u) => location.origin + u), tileSize: 256,
      minzoom: sat.minzoom || 0, maxzoom: sat.maxzoom || 19 };
    layers.push({ id: "sat", type: "raster", source: "sat" });
  } else if (mode === "vector" && vec) {
    sources.pm = { type: "vector", url: "pmtiles://" + location.origin + vec.url, attribution: info.attribution || "" };
    const S = (id, type, sl, paint, filter, extra) => layers.push(Object.assign({
      id, type, source: "pm", "source-layer": sl, paint, ...(filter ? { filter } : {}) }, extra || {}));
    S("earth", "fill", "earth", { "fill-color": t.land });
    S("landuse", "fill", "landuse", {
      "fill-color": ["match", ["get", "kind"], ["park", "forest", "wood", "grassland", "grass", "meadow", "nature_reserve", "garden", "golf_course", "cemetery", "recreation_ground"], t.park,
        ["pitch", "playground", "track", "stadium"], t.pitch, t.land],
      "fill-opacity": 0.95,
    });
    S("pitch-line", "line", "landuse", { "line-color": t.pitchLine, "line-width": ["interpolate", ["linear"], ["zoom"], 14, 1, 19, 3] },
      ["==", ["get", "kind"], "pitch"]);
    S("natural", "fill", "natural", { "fill-color": ["match", ["get", "kind"], ["wood", "forest", "scrub", "grassland", "heath", "wetland"], t.park, t.land] });
    S("water", "fill", "water", { "fill-color": t.water });
    S("roads-casing", "line", "roads", { "line-color": t.bg, "line-width": ["interpolate", ["linear"], ["zoom"], 12, 1, 18, 14], "line-opacity": 0.6 },
      ["!=", ["get", "kind"], "path"], { layout: { "line-cap": "round", "line-join": "round" } });
    S("roads", "line", "roads", {
      "line-color": ["match", ["get", "kind"], ["highway", "major_road"], t.roadMajor, ["path"], t.path, t.road],
      "line-width": ["interpolate", ["linear"], ["zoom"], 12, ["match", ["get", "kind"], ["highway", "major_road"], 1.5, 0.6], 18,
        ["match", ["get", "kind"], ["highway", "major_road"], 12, ["path"], 2, 8]],
    }, null, { layout: { "line-cap": "round", "line-join": "round" } });
    S("buildings", "fill", "buildings", { "fill-color": t.building, "fill-outline-color": t.buildingLine }, null, { minzoom: 14 });
  }
  return { version: 8, name: "offline-" + mode, sources, layers };
}
