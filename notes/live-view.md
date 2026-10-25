
## Camera orientation (rotate / flip) and camera robustness (2026-10-20)

**Settings** (`.env`, defaults in `config.py`): `CAMERA_ROTATE` (0|90|180|270 clockwise, default **180**), `CAMERA_FLIP`
(none|horizontal|vertical), `THERMAL_ROTATE` / `THERMAL_FLIP` (default 0 / none), `SENSOR_HEIGHT_MM` (3.348).
Rotation is applied first, then the flip (mirror of the rotated image). Applied at capture, so YOLO, CLIP crops, fusion,
storage and the live page all see the corrected image. `csi:` uses the hardware `nvvidconv flip-method=`
(`orient.gst_flip_method`, verified on this Jetson against cv2); `v4l2:`/`file:` use cv2.rotate/flip; thermal rotates the
160x120 degC array after the stuck-pixel fix (the stuck-pixel mask stays in sensor coordinates and is rotated with it).

| rotate, flip | flip-method | | rotate, flip | flip-method |
|---|---|---|---|---|
| 0, none | 0 | | 0, horizontal | 4 |
| 90, none | 3 (90 CW) | | 0, vertical | 6 |
| 180, none | 2 | | 90, vertical / 270, horizontal | 5 |
| 270, none | 1 (90 CCW) | | 90, horizontal / 270, vertical | 7 |
| | | | 180, horizontal = 0 vertical / 180, vertical = 0 horizontal | 6 / 4 |

**Which image axis is across-track:** the drone flies "up" the image. The image's HORIZONTAL axis is ACROSS-track (footprint
W, sets the lane spacing); the VERTICAL axis is ALONG-track (footprint L, sets max speed). With the default 180 the frame is
still 1920x1080 and W = 19.9 m, L = 11.2 m at 20 m. With 90/270 the frame is 1080x1920, the FOVs swap and W/L swap (mount the
camera accordingly). Calibration (`thermal_to_rgb.json`) now stores the orientations it was made with and is ignored (FOV-ratio
default used, warning logged) when they differ.

**Live page:** "Main camera" and "Thermal" rows show the current rotation/flip with `Rotate`, `Flip <->`, `Flip ^v` buttons
(standalone `live_view.py` only: previews immediately; for csi the pipeline restarts, ~3 s) and "Save as default", which
rewrites only CAMERA_ROTATE/FLIP and THERMAL_ROTATE/FLIP in `.env`. **main.py must be restarted to apply it.** On
`:8000/live` (pipeline mode) the controls are read-only. The Overlay view re-warps with the preview orientation at once
(default FOV-ratio homography assumes both corrected images are the same way up). API: `POST /api/live/orientation
{target: camera|thermal, rotate?, flip?}`, `POST /api/live/orientation/save`; `orientation` is in `/api/live/stats`.

**Bug fixes:** (1) RGB never retried: libargus keeps its daemon connection inside the process, so after an nvargus-daemon
restart ("Receive thread is not running") nothing in the same process can reopen the camera. CSI capture now runs in a
child process (`camera_worker.py`, frames over a pipe); every retry is a fresh process; `Camera._run` retries on every
failure (open, no first frame, 5 s stall, exception) with 0.5..30 s backoff; stats show `reconnects`, `attempts`,
`next_retry_s`. (2) SIGTERM/SIGINT: `shutdown.py` (live_view, main.py, system_check) -> stop uvicorn -> stop both cameras
(worker SIGTERM -> pipeline NULL, SIGKILL only after 2 s) -> `os._exit` at 2.8-3 s at the latest. Open MJPEG streams used to
block uvicorn's graceful shutdown. Measured on the real cameras: live_view exits in ~1.5 s, both devices free, the next
process opens immediately. nvargus-daemon still logs its usual enumeration noise ("V4L2Device not available", ...) each
time a client connects; no wedge, no new "Services are already stopped".
