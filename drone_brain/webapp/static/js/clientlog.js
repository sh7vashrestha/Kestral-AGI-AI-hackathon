// Classic script, loaded FIRST: forwards browser-side errors to the server (POST /api/client_log, rate-limited there)
// so a crashing tab / WebGL failure / reload loop leaves a trace in the drone's log. Never throws.
(function () {
  var sent = 0, seen = {};
  function post(level, message, source, stack) {
    try {
      var key = level + "|" + message + "|" + source;
      if (seen[key] || sent >= 20) return;          // dedupe + hard cap per page load
      seen[key] = 1; sent++;
      var body = JSON.stringify({
        level: level, message: String(message).slice(0, 900), source: String(source || "").slice(0, 280),
        stack: String(stack || "").slice(0, 2900), page: location.pathname + location.search, ua: navigator.userAgent,
      });
      if (navigator.sendBeacon) { navigator.sendBeacon("/api/client_log", new Blob([body], { type: "application/json" })); return; }
      fetch("/api/client_log", { method: "POST", headers: { "Content-Type": "application/json" }, body: body, keepalive: true });
    } catch (e) { /* logging must never break the page */ }
  }
  window.clientLog = post;
  window.addEventListener("error", function (e) {
    post("error", e.message || "script error", (e.filename || "") + ":" + (e.lineno || 0) + ":" + (e.colno || 0), e.error && e.error.stack);
  });
  window.addEventListener("unhandledrejection", function (e) {
    var r = e.reason; post("error", "unhandled rejection: " + (r && r.message || r), "", r && r.stack);
  });
  try {  // one line per page load: type 'reload' / 'navigate' / 'back_forward' tells a reload loop from first visits
    var nav = performance.getEntriesByType("navigation")[0];
    post("info", "page load (" + (nav ? nav.type : "?") + ")", document.referrer || "", "");
  } catch (e) {}
})();
