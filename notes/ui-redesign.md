# Phone UI redesign (neutral theme, chat fix)

## Why
User complaint: the chat Send button covered the chat, and the colours were harsh. Root cause of the Send bug:
`.primary { width: 100% }` inside the flex `.inline-form`, which squeezed the text input to about 20 px. Palette was
saturated yellow fills on black with a big red bar.

## Palette (CSS custom properties in `webapp/static/app.css`)
| token | dark (default) | light (outdoor) | use |
|---|---|---|---|
| --bg | #121417 | #f2f4f7 | page |
| --surface | #1b1e23 | #ffffff | cards, sheet, bars |
| --raised | #23272e | #eef1f4 | tiles, chips, secondary buttons |
| --border | #2e333b | #d3d8df | hairlines |
| --text | #e7e9ec | #15181c | body text |
| --muted | #9aa2ad | #535c68 | labels, timestamps |
| --accent | #3574d6 | #1f62d0 | primary buttons, selected tab/step, corners, route |
| --accent-text | #8ab4f8 | #1a56b8 | links, accent-coloured text |
| --ok / --ok-text | #3fb950 / #5ccf7a | #1f883d / #1a7f37 | ready, GPS fix, safety passed |
| --warn / --warn-text | #d29922 / #e8b649 | #bf8700 / #8a5a00 | warnings, armed, geofence |
| --danger / --danger-text | #d1343a / #ff8078 | #cf222e / #b3121f | Abort, errors, BLOCKED, target match |
| --heat | #f0883e | #d4560f | heat-signature alerts |

Spacing 4/8/12/16/24, radii 8/12/16/pill, touch targets 44 px (primary 48, Abort 52). WCAG contrast (computed):
text 13.7-17.8, muted 5.8-7.2, accent text 6.8-7.9, semantic text 5.1-8.9, white on accent 4.5 (dark) / 5.7 (light),
white on danger 4.9 / 5.4. All AA.

## Decisions
- Theme names are `dark` / `light` (the old saved `contrast` maps to `light`). An inline script in `index.html` sets
  `data-theme` before first paint from localStorage, else `prefers-color-scheme`; the ◐ button toggles and saves; with
  no saved choice the page follows OS changes. CSS also has a `prefers-color-scheme: light` fallback.
- Map styles (`js/style.js`): muted land/roads, the pitch only slightly green with a subtle outline, polygon/route in
  the accent, geofence amber, flown trail teal, route casing per theme (`casing`). Markers are DOM and use the CSS
  tokens: corners accent, home = inverted neutral square "H", drone = text-coloured arrow, START green pill, alert pins
  grey (person) / red (target match) / orange (heat).
- START label overlapping the "Map" button: MapLibre owns the marker's `transform`, so the old CSS
  `translateY(-22px)` never applied. Now `anchor: "bottom", offset [0,-7]` via the Marker API; preview fit padding
  is `{top 76, bottom 44, left 40, right 96}` to keep the route clear of the area chip and tool column; `#map` has
  `z-index: 0` so markers always stay under the map tools.
- Tabs are a segmented control (selected = surface + accent underline); wizard steps are pills (current = accent tint,
  done = green). At <= 400 px only the current step keeps its number badge.
- Status bar: one compact scrolling row with a fade on the right; order STATE, CAM, VEHICLE, GPS, BATT, ALT, MODE,
  ARMED so state and problems are always visible at 360 px; tabular numbers; semantic colours via val-ok/warn/bad.
- Each wizard step's main actions (`.actions`) are sticky at the bottom of the sheet, so Confirm / Next / Design route
  never scroll out of reach.
- Abort: solid red, full width, flex footer under `main` (nothing can scroll under it), safe-area padding. Armed state:
  white button, red border/text, "TAP AGAIN TO CONFIRM RTL · 4" countdown plus a draining red bar (4 s). Sent state
  darker red. With the keyboard open it shrinks to 44 px but stays visible.
- Chat: thread (`#chat`) + empty state with suggestion chips (`#chat-empty`) + composer pinned above Abort.
  Auto-growing textarea (1 to 5 lines), 44x44 send icon button (disabled when empty or while waiting), Enter sends,
  Shift+Enter newline, "Gemma is thinking…" bubble, error bubble with Retry, client timeout 130 s. Own lines show at
  once (pending) and are matched to the server echo (no duplicates); lines typed on another phone still appear.
  A plan asked for in chat stays in chat as a compact card (target, area, lanes, time, altitude, Safety OK/BLOCKED)
  with "Open in Preview"; a plan from the Survey wizard still opens Preview directly. History + draft live in
  sessionStorage (`chat.v1`, `chat.draft`, try/catch).
- Keyboard: `interactive-widget=resizes-content` (Android Chrome) plus a `visualViewport` handler that sets `--app-h`
  (body height) and an `html.kb-open` class (iOS).
- System panel (from the system-check work): report is now a list sorted fail > warn > skip > ok with coloured status
  dots instead of a 3-column emoji table; `#sysnote` problem banner styled as a warning notice.
- Vehicle tab starts with a "Live cameras" link card to `/live` (page built separately).
- No WebSocket/REST changes; all test-referenced ids kept (`abort-btn`, `chat-input`, `chat-send`, `chat-form`, `chat`).
- Known, unchanged: the fake dev server has no `/api/legal` (404 on confirm; the UI treats that as acknowledged).

## Screenshots
`data/ui_screens/` (git-ignored), each as `<name>_<360|412>_<dark|light>.png` (64 files):
s1_empty, s1_area, s2_questions, s3_preview, s3_preview_scrolled, s3_confirm_armed, s4_flight, alerts (person,
target match, heat), vehicle (with Live link), vehicle_system (Quick check report + camera-unavailable status/banner,
report mocked from data/system_check.json, banner injected because the fake controller never reports one),
chat_empty, chat_waiting, chat_thread (plan card), chat_keyboard (viewport at 52 % height), abort_armed, abort_sent.
Taken with Playwright Chromium (swiftshader WebGL) against `python -m webapp.server --fake --port 8767`.
