// =============================================
// CyberSatDetect | Session Inactivity Guard
// Auto-logout after 15 minutes of inactivity
// =============================================

(function () {
  const SESSION_TIMEOUT_MS = 15 * 60 * 1000; // 15 minutes
  const STORAGE_KEY = "csd_last_activity";
  const CHECK_INTERVAL_MS = 30 * 1000; // check every 30 seconds

  function updateActivity() {
    localStorage.setItem(STORAGE_KEY, Date.now().toString());
  }

  function getLastActivity() {
    const val = localStorage.getItem(STORAGE_KEY);
    return val ? parseInt(val, 10) : Date.now();
  }

  function isLoggedIn() {
    return !!localStorage.getItem("token");
  }

  function isPublicPage() {
    const p = (window.location.pathname || "").toLowerCase();
    return (
      p.endsWith("/login.html") ||
      p.endsWith("/signup.html") ||
      p.endsWith("/signup-simple.html") ||
      p.endsWith("/otp-verify.html") ||
      p.endsWith("/forgot-password.html") ||
      p.endsWith("/reset-password.html") ||
      p.endsWith("/index.html") ||
      p === "/" ||
      p.endsWith("/terms.html") ||
      p.endsWith("/about.html")
    );
  }

  function requireLogin() {
    if (isPublicPage()) return;
    if (isLoggedIn()) return;
    const loginUrl = new URL("login.html", window.location.href);
    loginUrl.searchParams.set("reason", "missing_token");
    window.location.assign(loginUrl.toString());
  }

  function logout() {
    console.warn("🔒 Session expired due to inactivity.");
    localStorage.removeItem("token");
    localStorage.removeItem("csd_run_id");
    localStorage.removeItem("csd_file_name");
    localStorage.removeItem("loginEmail");
    localStorage.removeItem(STORAGE_KEY);

    const loginUrl = new URL("login.html", window.location.href);
    loginUrl.searchParams.set("reason", "inactivity");
    window.location.assign(loginUrl.toString());
  }

  function checkTimeout() {
    if (!isLoggedIn()) return;

    const elapsed = Date.now() - getLastActivity();
    if (elapsed >= SESSION_TIMEOUT_MS) {
      logout();
    }
  }

  // Track user activity
  const ACTIVITY_EVENTS = ["mousedown", "keydown", "scroll", "touchstart"];
  ACTIVITY_EVENTS.forEach(function (evt) {
    document.addEventListener(evt, updateActivity, { passive: true });
  });

  // Initialize last activity
  if (isLoggedIn()) {
    updateActivity();
  }

  // Enforce login on protected pages
  requireLogin();

  // Periodic check
  setInterval(checkTimeout, CHECK_INTERVAL_MS);

  // Also check on focus (user returns to tab)
  window.addEventListener("focus", checkTimeout);
})();
