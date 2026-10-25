# Hardening B: geometry, coordinates, limits, safety items, mission diff

Tests: `tests/test_geometry_edges.py` (122 cases + a 2 000-polygon seeded fuzz and a 300-rectangle fuzz; `hypothesis`
is not installed in the venv). Result column: OK = already correct, FIXED = bug found and fixed, DOC = documented, not changed.

| Edge case | Expected | Result | Fix |
|---|---|---|---|
| 3-8 corners | built and safe | OK | |
| 0,1,2,9 corners | PolygonError | OK | |
| repeated corner (adjacent or closing) | "same place" error | OK | |
| collinear / zero area | "in a line" error | OK | |
| bow-tie | "crosses itself" error | OK | |
| concave | hull + warning | OK | |
| collinear extra vertex on an edge | no false "not convex" warning | FIXED (warned falsely) | warn on hull area only |
| clockwise vs counter-clockwise, any rotation/order of corners | identical route | OK (hull is order independent) | |
| 1 m x 500 m | one lane, inside | OK | |
| < MIN_AREA, > MAX_AREA | safety blocks, clear reason | OK | |
| corner straddles geofence / area fully outside | each corner/waypoint named | OK | |
| home inside / far outside area | builds, start nearest home | OK | |
| vertex touching a non-adjacent edge | treated as concave -> hull | DOC | not flagged as self-intersecting; the hull makes it harmless |
| sharp wedge tips | waypoints stay inside the hull, so a very acute tip is covered only by the camera footprint (L/2 past the lane end) | DOC | deliberate: waypoints must not leave the polygon |
| absurd size (lanes > 2 000) | clear error, no memory blow-up | FIXED (unbounded loop) | `MAX_LANES` in survey.py, rect + polygon |
| decimal / space / tab / CRLF / `;` separated | parsed | OK | |
| `N33.9371 W84.5271` (letter before number) | lat 33.9, lon -84.5 | FIXED: returned lat 84.5, lon -33.9 (wrong place) | coords.py rewritten |
| DMS, degrees+decimal minutes, S/E, prefix letters | parsed | OK / FIXED (prefix) | |
| comma decimals `33,9371; -84,5271` | parsed | FIXED (error) | |
| Google Maps `(lat, lon)`, unicode minus, `@lat,lon`, `!3d..!4d..`, `?q=` links | parsed | FIXED (error) | |
| closed ring (first point repeated at end, KML/GeoJSON) | last point dropped | FIXED (duplicate error later) | |
| sign and letter disagree (`-33.9 S`, `33.9 N, 84.5 N`) | clear error | FIXED (silently accepted) | |
| NaN, inf, `1e1`, text, out of range, <3, >8, duplicates | specific message with line number | OK / FIXED (duplicates) | |
| lat > 90 (swapped) | rejected with "swapped?" hint, never silently swapped | OK | |
| swapped but both values <= 90 | rejected only when `home=` passed and swapped reading is within 5 km of home; otherwise the geofence rejects it | DOC | server.py should pass `home=` (see below) |
| round trip local<->geo, 10 latitudes incl. equator, S/E hemispheres, 88.9 deg, +-180 | < 1 cm | OK | |
| anti-meridian | lon wraps to [-180,180); east offset correct across the line | FIXED (lon exceeded 180) | `wrap_lon` |
| pole (|lat| > 89, NaN) | clear ValueError | FIXED (ZeroDivisionError at 90) | `_check_home_lat` |
| area > 2 km from home | flat-earth warning | FIXED (no warning) | |
| altitude exactly MAX_ALT / LAANC / obstacle clearance | allowed; epsilon beyond blocked | OK | |
| LAANC ceiling < obstacle clearance | blocked, "cannot be flown legally and safely" | OK | |
| MAX_ALT_M > 120 | config rejects | OK | |
| altitude 0, negative, NaN, inf | rejected | FIXED for NaN/inf in `build_polygon_survey` (NaN passed `<= 0`) | |
| overlap 0, 0.5, 0.89 | ok; >= 0.9, <0, NaN rejected | OK / FIXED (NaN message) | |
| capture_fps 0.2 / 5 | v_max = L(1-ov)fps | OK | |
| capture_fps 0 or negative | config rejects | FIXED (ZeroDivisionError at plan time) | `Field(gt=0, le=60)` in config.py |
| rotation 90/270/180 | W/L swap at 90/270 | OK | |
| speed exactly v_max | no warning | OK | |
| speed > v_max | clamped + warning | OK | |
| speed > MAX_SPEED, <= 0, NaN | safety blocks / PolygonError | OK / FIXED (negative speed accepted by polygon builder) | |
| path exactly MAX_PATH | allowed; 0.01 m over blocked | OK | |
| path over limit only because of home legs | blocked ("Total path") | OK | |
| > 600 mission items | blocked with message | FIXED (no cap) | `MAX_MISSION_ITEMS = 600` (ArduPilot Copter on a Pixhawk holds ~700); plan and uploaded list |
| item list: no TAKEOFF / TAKEOFF after first waypoint | blocked | FIXED | |
| RTL not last / RTL twice / items after RTL | blocked | FIXED for double RTL | |
| item 0 home placeholder | ignored whatever it holds | OK | |
| NaN/inf in lat, lon, alt, param1-4 | blocked | FIXED (params unchecked) | |
| positional item in AMSL / terrain frame | blocked (limits are relative to home) | FIXED | |
| WAYPOINT at lat=lon=0 | geofenced like any other | FIXED (skipped as "no position") | |
| mission_diff tolerance boundaries | strict `>` tolerance | OK | |
| mission_diff missing / extra / reordered | rows flagged | OK | |
| mission_diff NaN on either side | mismatch | FIXED (NaN compared equal) | |
| mission_diff duplicate seq | flagged, not silently overwritten | FIXED | |
| fuzz: 2 000 polygons x random home/settings | waypoints inside hull, finite, deterministic, order-independent, speed <= v_max, geofence respected when safety ok, hull sample points within W/2 across-track of a lane | all hold | |

## Needs a change in another module
- `webapp/server.py` `/api/parse_coords`: pass `home=(lat, lon)` to `parse_corners` so swapped-but-valid input gets the helpful message
  (the geofence already blocks the plan either way).
- `mission.py` upload path: nothing required; `validate_mission_items` now also checks frames/params, so callers that build items with
  AMSL positional frames or NaN params will be refused (the current builder uses frame 6 and finite values).
