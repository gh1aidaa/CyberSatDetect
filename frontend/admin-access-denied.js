/**
 * Unified "admin only" access-denied UI (Security Center + CL Admin).
 */
function showAdminAccessDenied(container, role) {
  const el = typeof container === "string" ? document.querySelector(container) : container;
  if (!el) return;

  const roleLabel = role != null && String(role).trim() !== "" ? String(role) : "—";
  const safeRole = roleLabel
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  el.innerHTML = `
    <div class="access-denied">
      <div class="access-denied-icon" aria-hidden="true">⛔</div>
      <h2 class="access-denied-title">Access Denied</h2>
      <p class="access-denied-msg">This page is restricted to <strong>ADMIN</strong> users only.</p>
      <p class="access-denied-role">Your role: <strong class="access-denied-role-val">${safeRole}</strong></p>
      <a class="access-denied-back" href="dashboard.html">← Back to Dashboard</a>
    </div>
  `;
}
