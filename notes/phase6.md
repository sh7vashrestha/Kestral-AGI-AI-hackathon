# Phase 6 - storage, tracker, matcher

## Progress
- storage.py, tracker.py, matcher.py + tests (test_storage/test_tracker/test_matcher) done.
- Smoke test against the live llama-server: see end of this file if run.

## Decisions
- Scoring: C_match = match_confidence if (is_human and matches) else 0; C_final = C_yolo * C_match;
  C_avg = mean of last 5 (AVG_WINDOW) C_final for the track under the current target;
  target_match alert when n_results >= min_frames and C_avg >= match_threshold.
- "person" alert is immediate, once per track (rate limit is per level, so it never delays target_match).
- target_match: once per track per target description; re-alert only if C_avg improved >= 0.10 and
  alert_min_interval_s passed since the last target_match for that track.
- update_target(): resets every track's match history/alert state, drops pending jobs, discards the result of
  an in-flight VLM call (counted as "stale"). Person-alert state is kept (same person, new search).
- Queue: one pending job per track (highest c_yolo*bbox area), max vlm_queue_size jobs, oldest dropped. No jobs
  are queued while no target is set (person alerts still fire).
- Tracker time = Detection.ts; IoU match only if seen <= 3 s ago; GPS match against any live track; tracks
  expire after 30 s. Ground projection needs telemetry lat/lon/alt_rel (>0.5 m)/heading, nadir flat-ground.
- VLM Unavailable/Timeout: exponential backoff 0.5..15 s in the worker; job is dropped, never crashes.
- Storage never raises; flight dir has flight.json (plan, targets, start/end), crops/, frames/, *.jsonl.
- models.py not modified.
