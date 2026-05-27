// =====================================
// CyberSatDetect | Security Center
// =====================================

const API_BASE = window.location.origin;

function getToken() {
  return localStorage.getItem("token") || "";
}
function authHeaders() {
  return { Authorization: "Bearer " + getToken() };
}

// ---------- Access Control ----------
async function checkAccess() {
  try {
    const res = await fetch(API_BASE + "/auth/me", { headers: authHeaders() });
    if (!res.ok) { window.location.href = "login.html"; return false; }
    const user = await res.json();
    if (user.role !== "ADMIN") {
      showAdminAccessDenied(document.querySelector(".security-page"), user.role);
      return false;
    }
    return true;
  } catch { window.location.href = "login.html"; return false; }
}

// ---------- State ----------
let allIncidents = [];
let allUsers = [];
let auditData = { incidents: [], runs: [] };
let typeChart = null;
let timeChart = null;

// =====================================
// SPACE BACKGROUND (same as dashboard)
// =====================================
let bgScene, bgCamera, bgRenderer, bgSphere, bgStars;

function initBG() {
  const container = document.getElementById("canvas_container");
  if (!container) return;

  // force container style
  container.style.position = "fixed";
  container.style.top = "0";
  container.style.left = "0";
  container.style.width = "100vw";
  container.style.height = "100vh";
  container.style.zIndex = "-1";
  container.style.pointerEvents = "none";

  bgScene = new THREE.Scene();
  bgCamera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1200);
  bgCamera.position.set(0, 0, 260);

  bgRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  bgRenderer.setSize(window.innerWidth, window.innerHeight);
  bgRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(bgRenderer.domElement);

  const controls = new THREE.OrbitControls(bgCamera, bgRenderer.domElement);
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.7;
  controls.enablePan = false;
  controls.enableZoom = false;

  const loader = new THREE.TextureLoader();
  const bg = loader.load("https://i.ibb.co/HC0vxMw/sky2.jpg");
  bgSphere = new THREE.Mesh(
    new THREE.SphereGeometry(520, 48, 48),
    new THREE.MeshBasicMaterial({ side: THREE.BackSide, map: bg })
  );
  bgScene.add(bgSphere);

  const starsGeo = new THREE.BufferGeometry();
  const positions = [];
  for (let i = 0; i < 600; i++) {
    positions.push(
      (Math.random() - 0.5) * 1000,
      (Math.random() - 0.5) * 1000,
      (Math.random() - 0.5) * 1000
    );
  }
  starsGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  bgStars = new THREE.Points(starsGeo, new THREE.PointsMaterial({ color: "#ffffff", size: 1.6, opacity: 0.9, transparent: true }));
  bgScene.add(bgStars);

  animateBG();
}

function animateBG() {
  requestAnimationFrame(animateBG);
  if (bgSphere) bgSphere.rotation.y += 0.0007;
  if (bgStars) bgStars.rotation.y += 0.00035;
  bgRenderer.render(bgScene, bgCamera);
}

window.addEventListener("resize", () => {
  if (!bgCamera || !bgRenderer) return;
  bgCamera.aspect = window.innerWidth / window.innerHeight;
  bgCamera.updateProjectionMatrix();
  bgRenderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------- Tabs ----------
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ---------- Filters ----------
document.getElementById("filterType").addEventListener("change", renderIncidents);
document.getElementById("filterStatus").addEventListener("change", renderIncidents);
document.getElementById("auditFilter").addEventListener("change", renderAuditLogs);

// ---------- Init on Load ----------
document.addEventListener("DOMContentLoaded", async () => {
  initBG();
  const ok = await checkAccess();
  if (ok) loadAll();
});

// ---------- Fetch Data ----------
async function loadAll() {
  await Promise.all([loadIncidents(), loadUsers(), loadAudit()]);
  updateKPIs();
}

async function loadIncidents() {
  try {
    const res = await fetch(API_BASE + "/admin/incidents", { headers: authHeaders() });
    const data = await res.json();
    allIncidents = data.incidents || [];
    renderIncidents();
    renderCharts();
  } catch (e) { console.error("Incidents error:", e); }
}

async function loadUsers() {
  try {
    const res = await fetch(API_BASE + "/admin/users", { headers: authHeaders() });
    const data = await res.json();
    allUsers = data.users || [];
    renderUsers();
  } catch (e) { console.error("Users error:", e); }
}

async function loadAudit() {
  try {
    const res = await fetch(API_BASE + "/admin/audit-logs", { headers: authHeaders() });
    auditData = await res.json();
    renderAuditLogs();
  } catch (e) { console.error("Audit error:", e); }
}

// ---------- KPIs ----------
function updateKPIs() {
  const open = allIncidents.filter(i => i.status === "OPEN").length;
  const closed = allIncidents.filter(i => i.status === "CLOSED").length;
  const blocked = allUsers.filter(u => u.is_blocked).length;

  document.getElementById("kpiOpen").textContent = open;
  document.getElementById("kpiClosed").textContent = closed;
  document.getElementById("kpiUsers").textContent = allUsers.length;
  document.getElementById("kpiBlocked").textContent = blocked;
}

// ---------- Format Helpers ----------
function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString("en-GB") + " " + d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

function typeBadge(type) {
  const cls = "type-" + (type || "").toLowerCase();
  return `<span class="${cls}">${type}</span>`;
}

function statusBadge(status) {
  const cls = status === "OPEN" ? "badge-open" : "badge-closed";
  return `<span class="badge ${cls}">${status}</span>`;
}

// ---------- Incidents Table ----------
function renderIncidents() {
  const typeFilter = document.getElementById("filterType").value;
  const statusFilter = document.getElementById("filterStatus").value;

  let filtered = allIncidents;
  if (typeFilter) filtered = filtered.filter(i => i.type === typeFilter);
  if (statusFilter) filtered = filtered.filter(i => i.status === statusFilter);

  const tbody = document.getElementById("incidentTableBody");
  if (!filtered.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#666">No incidents found</td></tr>';
    return;
  }

  tbody.innerHTML = filtered.map(i => `
    <tr>
      <td>${fmtTime(i.created_at)}</td>
      <td>${typeBadge(i.type)}</td>
      <td>${i.user_id || "—"}</td>
      <td>${i.ip || "—"}</td>
      <td>${i.details || "—"}</td>
      <td>${statusBadge(i.status)}</td>
      <td>${i.status === "OPEN"
        ? `<button class="btn-sm btn-close" onclick="closeIncident('${i.id}')">Close</button>`
        : "—"}</td>
    </tr>
  `).join("");
}

// ---------- Close Incident ----------
window.closeIncident = async function(id) {
  try {
    await fetch(API_BASE + "/admin/incidents/" + id + "/close", {
      method: "POST",
      headers: authHeaders()
    });
    await loadIncidents();
    updateKPIs();
  } catch (e) { console.error("Close error:", e); }
};

// ---------- Charts ----------
function renderCharts() {
  // Type distribution (doughnut)
  const typeCounts = {};
  allIncidents.forEach(i => { typeCounts[i.type] = (typeCounts[i.type] || 0) + 1; });

  const typeLabels = Object.keys(typeCounts);
  const typeValues = Object.values(typeCounts);
  const typeColors = ["#ff4d4f", "#faad14", "#ff7875", "#b37feb", "#ffc069", "#69c0ff", "#52c41a"];

  if (typeChart) typeChart.destroy();
  typeChart = new Chart(document.getElementById("incidentTypeChart"), {
    type: "doughnut",
    data: {
      labels: typeLabels,
      datasets: [{ data: typeValues, backgroundColor: typeColors.slice(0, typeLabels.length), borderWidth: 0 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "right", labels: { color: "#ccc", font: { size: 11 } } },
        title: { display: true, text: "Incidents by Type", color: "#9ecfff", font: { size: 13 } }
      }
    }
  });

  // Timeline (bar — last 7 days)
  const dayCounts = {};
  const now = new Date();
  for (let d = 6; d >= 0; d--) {
    const day = new Date(now);
    day.setDate(day.getDate() - d);
    dayCounts[day.toISOString().slice(0, 10)] = 0;
  }
  allIncidents.forEach(i => {
    const day = (i.created_at || "").slice(0, 10);
    if (day in dayCounts) dayCounts[day]++;
  });

  const timeLabels = Object.keys(dayCounts).map(d => d.slice(5));
  const timeValues = Object.values(dayCounts);

  if (timeChart) timeChart.destroy();
  timeChart = new Chart(document.getElementById("incidentTimeChart"), {
    type: "bar",
    data: {
      labels: timeLabels,
      datasets: [{
        label: "Incidents",
        data: timeValues,
        backgroundColor: "rgba(255,77,79,0.5)",
        borderColor: "#ff4d4f",
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#8ab4d0" }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { ticks: { color: "#8ab4d0", stepSize: 1 }, grid: { color: "rgba(255,255,255,0.04)" }, beginAtZero: true }
      },
      plugins: {
        legend: { display: false },
        title: { display: true, text: "Incidents (Last 7 Days)", color: "#9ecfff", font: { size: 13 } }
      }
    }
  });
}

// ---------- Audit Logs ----------
function renderAuditLogs() {
  const filter = document.getElementById("auditFilter").value;
  const tbody = document.getElementById("auditTableBody");
  let rows = [];

  if (filter === "all" || filter === "incidents") {
    (auditData.incidents || []).forEach(i => {
      rows.push({
        time: i.created_at,
        cat: "Security",
        catClass: "cat-security",
        user: i.user_id || "—",
        event: i.type,
        details: i.details || "—"
      });
    });
  }

  if (filter === "all" || filter === "runs") {
    (auditData.runs || []).forEach(r => {
      rows.push({
        time: r.created_at,
        cat: "File Op",
        catClass: "cat-file",
        user: r.user_id || "—",
        event: r.status,
        details: r.filename || "—"
      });
    });
  }

  rows.sort((a, b) => (b.time || "").localeCompare(a.time || ""));

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#666">No logs found</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${fmtTime(r.time)}</td>
      <td><span class="${r.catClass}">${r.cat}</span></td>
      <td>${r.user}</td>
      <td>${r.event}</td>
      <td>${r.details}</td>
    </tr>
  `).join("");
}

// ---------- Users Table ----------
function renderUsers() {
  const tbody = document.getElementById("userTableBody");
  if (!allUsers.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#666">No users found</td></tr>';
    return;
  }

  tbody.innerHTML = allUsers.map(u => {
    const isBlocked = u.is_blocked === 1;
    const isLocked = u.locked_until && new Date(u.locked_until) > new Date();

    let statusHtml;
    if (isBlocked) statusHtml = '<span class="badge badge-blocked">Blocked</span>';
    else if (isLocked) statusHtml = '<span class="badge badge-locked">Locked</span>';
    else statusHtml = '<span class="badge badge-active">Active</span>';

    const roleBadge = u.role === "ADMIN"
      ? '<span class="badge badge-admin">ADMIN</span>'
      : '<span class="badge badge-user">USER</span>';

    const btnClass = isBlocked ? "btn-unblock" : "btn-block";
    const btnText = isBlocked ? "Unblock" : "Block";

    return `
      <tr>
        <td>${u.email}</td>
        <td>${roleBadge}</td>
        <td>${statusHtml}</td>
        <td>${u.failed_attempts || 0}</td>
        <td>${u.locked_until ? fmtTime(u.locked_until) : "—"}</td>
        <td>${fmtTime(u.created_at)}</td>
        <td><button class="btn-sm ${btnClass}" onclick="toggleBlock('${u.id}')">${btnText}</button></td>
      </tr>
    `;
  }).join("");
}

// ---------- Toggle Block ----------
window.toggleBlock = async function(userId) {
  try {
    await fetch(API_BASE + "/admin/users/" + userId + "/toggle-block", {
      method: "POST",
      headers: authHeaders()
    });
    await Promise.all([loadUsers(), loadIncidents()]);
    updateKPIs();
  } catch (e) { console.error("Block error:", e); }
};
