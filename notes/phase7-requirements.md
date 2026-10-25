# Phase 7 requirements (collected from the user)

## Altitude and airspace (KSU Marietta soccer field, near Dobbins ARB)

All altitudes are **relative to home** (arming position), MAVLink frame
`MAV_FRAME_GLOBAL_RELATIVE_ALT_INT` (6). Already true in `mission.py`; keep it.
Arm on the field itself so home altitude = field ground level.

Effective altitude ceiling used by `safety.py` and the UI slider:

    effective_max_alt = min(MAX_ALT_M, LAANC_CEILING_M (if set), HARD_MAX_ALT_M = 120)

New settings (ADD to config.py / .env.example):
- `LAANC_CEILING_M` (optional): the altitude approved in the LAANC authorization
  (UAS facility map grid near Dobbins can be low). Human sets it per flight; UI shows it.
- `OBSTACLE_CLEARANCE_M` (default 15): minimum survey and RTL altitude above home,
  so lanes and RTL clear light poles, goals and trees around the field.
  `MIN_ALT_M` effectively becomes `max(MIN_ALT_M, OBSTACLE_CLEARANCE_M)`.
  Reject if `OBSTACLE_CLEARANCE_M > effective_max_alt` (cannot fly legally and safely).

## Pre-flight cross-check with the Pixhawk (read-only parameter reads)

`mission.preflight_check()` reads (PARAM_REQUEST_READ only, never PARAM_SET on real
hardware) and compares with .env; any mismatch blocks Confirm with a clear message:

| Pixhawk parameter | Requirement |
|---|---|
| `RTL_ALT` (cm) | ≥ OBSTACLE_CLEARANCE_M and ≤ effective_max_alt |
| `FENCE_ENABLE` | 1 |
| `FENCE_TYPE` | includes altitude (bit 0) and circle (bit 1) |
| `FENCE_ALT_MAX` (m) | ≤ effective_max_alt |
| `FENCE_RADIUS` (m) | ≤ GEOFENCE_RADIUS_M |
| `FENCE_ACTION` | RTL or Land (not 0) |
| `FS_BATT_ENABLE` / `BATT_FS_LOW_ACT` | RTL or Land |
| `FS_THR_ENABLE` | enabled (RC loss) |
| `WPNAV_SPEED` (cm/s) | ≥ plan speed (DO_CHANGE_SPEED overrides, but warn) |

The plan preview shows: effective ceiling and where it comes from (LAANC / MAX_ALT / 120 m),
RTL altitude, fence status, and the pre-flight result.

## Legal checklist (README, shown in the UI before the first Confirm of a session)

- LAANC authorization obtained for this location, time and altitude (Dobbins ARB Class D nearby)
- KSU campus UAS approval
- Part 107 (or recreational TRUST) rules: VLOS, no flight over non-participants
- Props-off bench test done after any code change
