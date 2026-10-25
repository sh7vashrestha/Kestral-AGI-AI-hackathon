# Phase 2b: map UI + polygon survey

## Progress
- models.py: `LatLon`, `Mission` extended (shape rectangle|polygon, `corners`, length/width now Optional with a validator that requires them for rectangles), `AreaPlanRequest`, optional `SurveyPlan.area_m2/lane_count/estimated_flight_s/battery_note`.
- config.py: `max_area_m2` (200000), `min_area_m2` (100), `est_endurance_min` (15), `map_satellite_mbtiles`, property `map_dir` (= data_dir/maps).
- survey.py: `build_polygon_survey`, `PolygonError`, `flight_estimate` (rectangles get the estimate too).
- safety.py: polygon corners vs geofence, area min/max.
- webapp: `maps.py` (PMTiles Range, MBTiles, /api/map/info), `coords.py` (paste parser), `/api/plan_area`, WS `plan_area`, `/api/limits`, `/api/parse_coords`; Controller.plan_area; FakeController flies the confirmed route around the KSU soccer field.
- UI: `static/index.html`, `app.css`, `app.js` (ES module), `js/{mapview,style,geo}.js`; vendored MapLibre GL JS 4.7.1 + pmtiles 3.2.1 in `static/vendor/`.
- scripts/fetch_map.py: one-time online extract.
- Tests: tests/test_polygon.py, tests/test_webapp_map.py.

## Decisions
- Flying area: KSU Marietta campus soccer field "Nuesoft Technologies Field" (OSM way 180244662), centre 33.9371904, -84.5271293 (found via Overpass, inside the campus bbox of way 34833071). Fake home = this point. Dobbins ARB is close: `/api/map/info.notice` carries an airspace/LAANC/campus-approval reminder shown in the UI. No airspace logic.
- Mission model: kept one `Mission` class (not a union). length/width Optional; validator enforces them for rectangles so existing callers/tests are unchanged.
- Polygon: self-intersecting -> rejected (`PolygonError`, user-facing text). Concave -> replaced by the convex hull with a warning (hull vertices are a subset of the tapped corners so the geofence check still holds). Sweep along the longest hull edge; n = ceil(extent/s) lanes spread evenly (spacing <= s, first lane half a spacing inside), chords clipped to the polygon; of the 4 zig-zag traversals the shortest home->path->home is used (ties: first point nearest home) which starts at the corner nearest home.
- Flight time = path/speed + alt/2.5 (climb) + alt/1.5 (descent) + 2 s per waypoint + 20 s fixed; battery note assumes `est_endurance_min`.
- Map library: MapLibre GL JS (WebGL) + PMTiles protocol; raster satellite via MBTiles served from `/tiles/sat/{z}/{x}/{y}.ext` (TMS flip). Leaflet not needed: bundle is 800 KB, runs fine in Chromium; real phones all have WebGL. Without WebGL the map is replaced by a message and "Paste coordinates" still works.
- Vector style is built in JS (`js/style.js`, dark + contrast) for the Protomaps v4 schema (properties are `kind`, not `pmap:kind`). No text layers, so no glyph/sprite files are needed offline; grid coordinates are DOM labels.
- No-map mode: plain grid/graticule with home, drone, fence and corners.
- Satellite: only a user-supplied `.mbtiles` (`MAP_SATELLITE_MBTILES=file.mbtiles` in data/maps/); no bulk download. Layer button appears only if present.
- fetch_map.py downloads go-pmtiles v1.31.2 once into data/maps/bin if `pmtiles` is not on PATH, finds the newest Protomaps daily build (HEAD requests), runs `pmtiles extract --bbox --maxzoom`.
- Fetch result: `--center 33.9371904,-84.5271293 --radius-km 1.5 --maxzoom 18` -> data/maps/area.pmtiles 1.5 MB (source data tops out at z15; MapLibre overzooms to z20). data/ is git-ignored.
- Map `info.center` comes from the pmtiles header; the UI centres on drone position, else `snapshot.home`, else map centre.
- Optional snapshot keys `home` {lat,lon} and `route` are used by the UI when present.
- plan_area rejections (bow-tie etc.) are broadcast as `error` messages; safety failures still return a PlanProposal with `safety.ok=false` so the UI shows BLOCKED reasons.
- UI is tested with Playwright Chromium (software WebGL) in a scratch venv; headless Firefox here has no WebGL.
