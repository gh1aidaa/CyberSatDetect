// =====================================
// CyberSatDetect | Dashboard (Final)
// =====================================

const API_BASE = window.location.origin;
let anomalyLineChart = null;
let anomalyPieChart = null;

// =====================================
// FORCE BACKGROUND CONTAINER STYLE (JS)
// =====================================
document.addEventListener("DOMContentLoaded", () => {
  const bgContainer = document.getElementById("canvas_container");
  if (bgContainer) {
    bgContainer.style.position = "fixed";
    bgContainer.style.top = "0";
    bgContainer.style.left = "0";
    bgContainer.style.width = "100vw";
    bgContainer.style.height = "100vh";
    bgContainer.style.zIndex = "-1";
    bgContainer.style.pointerEvents = "none";
  }

  initDashboard();
  initBG();
});

// =====================================
// AUTH HELPERS
// =====================================
function getToken() {
  return localStorage.getItem("token");
}

function authHeaders() {
  const t = getToken();
  return t ? { "Authorization": "Bearer " + t } : {};
}

// =====================================
// DASHBOARD DATA
// =====================================
async function initDashboard() {
  try {
    console.log("Dashboard API_BASE =", API_BASE);

    // Require login (avoid silent empty dashboard when token missing/expired)
    if (!getToken()) {
      const loginUrl = new URL("login.html", window.location.href);
      loginUrl.searchParams.set("reason", "missing_token");
      window.location.assign(loginUrl.toString());
      return;
    }

    const res = await fetch(`${API_BASE}/dashboard/runs`, {
      headers: authHeaders()
    });

    if (res.status === 401 || res.status === 403) {
      const loginUrl = new URL("login.html", window.location.href);
      loginUrl.searchParams.set("reason", "unauthorized");
      window.location.assign(loginUrl.toString());
      return;
    }

    const data = await res.json();
    if (!res.ok) throw new Error(data?.detail || "Failed to load dashboard");

    const runs = data.runs || [];
    const anomalyTypes = data.anomaly_types || [];
    const model = data.model || {};

    initKPIs(runs, model);
    initActivityFeed(runs);
    initTimeline(runs);
    initChart(runs);
    initPieChart(runs, anomalyTypes);
    initRunsTable(runs); // 🔥 الجديد
    initModel(model);

  } catch (e) {
    console.error("Dashboard load failed", e);
    // Surface failure in UI so it doesn't look like "no data"
    const msg = `Failed to load data from API (${API_BASE}).`;
    const filesEl = document.getElementById("filesAnalyzed");
    const statusEl = document.getElementById("modelStatus");
    const alertEl = document.getElementById("alertCount");
    const updEl = document.getElementById("lastUpdate");
    if (filesEl) filesEl.textContent = "—";
    if (statusEl) statusEl.textContent = "API error";
    if (alertEl) alertEl.textContent = "—";
    if (updEl) updEl.textContent = "—";

    const tbody = document.getElementById("runsTableBody");
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="5">${msg}</td></tr>`;
    }
  }
}

function getDisplayName(run) {
  const name = (run.filename || "").trim();
  return name || run.run_id;
}

function formatDateTime(value) {
  if (!value) return "—";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return "—";
  return dt.toLocaleString();
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  const num = Number(value);
  // إذا كانت القيمة بين 0 و 1 (decimal)، اضربها في 100
  const displayValue = num < 1 ? num * 100 : num;
  return `${displayValue.toFixed(1)}%`;
}

// =====================================
// KPIs
// =====================================
function initKPIs(runs, model) {
  const files = runs.length;
  const anomalies = runs.filter(r => r.anomaly_rate && r.anomaly_rate > 0).length;
  const lastUpdate = runs[0]?.created_at
    ? new Date(runs[0].created_at).toLocaleTimeString()
    : "—";

  document.getElementById("filesAnalyzed").textContent = files;
  document.getElementById("modelStatus").textContent = model.status || "—";
  document.getElementById("alertCount").textContent = anomalies;
  document.getElementById("lastUpdate").textContent = lastUpdate;
}

// =====================================
// ACTIVITY FEED
// =====================================
function initActivityFeed(runs) {
  const list = document.getElementById("activityList");
  list.innerHTML = "";

  if (!runs.length) {
    list.innerHTML = "<li>No recent activity</li>";
    return;
  }

  runs.slice(0, 5).forEach(run => {
    const li = document.createElement("li");
    li.textContent = `${getDisplayName(run)} → ${run.status}`;
    list.appendChild(li);
  });
}

// =====================================
// TIMELINE
// =====================================
function initTimeline(runs) {
  const bar = document.getElementById("timelineBar");
  bar.innerHTML = "";

  runs.slice(0, 24).forEach(r => {
    const div = document.createElement("div");
    div.className = "timeline-item";
    div.style.background = r.anomaly_rate ? "#E74C3C" : "#00BFFF";
    div.style.height = "30px";
    div.title = `${getDisplayName(r)} (${r.run_id})`;
    bar.appendChild(div);
  });
}

// =====================================
// CHART
// =====================================
function initChart(runs) {
  const canvas = document.getElementById("anomalyChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const labels = runs
    .slice(0, 10)
    .map(r => {
      const name = getDisplayName(r);
      return name.length > 12 ? `${name.slice(0, 12)}...` : name;
    });
  const values = runs.slice(0, 10).map(r => r.anomaly_rate || 0);

  if (anomalyLineChart) {
    anomalyLineChart.destroy();
  }

  anomalyLineChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Anomaly Rate",
        data: values,
        borderColor: "#00BFFF",
        backgroundColor: "rgba(0,191,255,0.2)",
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } }
    }
  });
}

function initPieChart(runs, anomalyTypes) {
  const canvas = document.getElementById("pieChart");
  const note = document.getElementById("anomalyTypesNote");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  if (anomalyPieChart) {
    anomalyPieChart.destroy();
    anomalyPieChart = null;
  }

  if (!anomalyTypes.length) {
    canvas.style.display = "none";
    if (note) note.style.display = "block";
    return;
  }

  canvas.style.display = "block";
  if (note) note.style.display = "none";

  const labels = anomalyTypes.map(a => a.type);
  const values = anomalyTypes.map(a => a.count || 0);

  anomalyPieChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: ["#E74C3C", "#00BFFF", "#F39C12", "#2ECC71", "#9B59B6"],
        borderWidth: 1,
        borderColor: "rgba(255,255,255,0.15)"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: "#EAF6FF"
          }
        }
      }
    }
  });
}

// =====================================
// EXECUTION RUNS TABLE  🔥 الجديد
// =====================================
function initRunsTable(runs) {
  const tbody = document.getElementById("runsTableBody");
  if (!tbody) return;

  tbody.innerHTML = "";

  if (!runs.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="5">No runs available</td>
      </tr>
    `;
    return;
  }

  runs.slice(0, 10).forEach(run => {
    const tr = document.createElement("tr");

    // 🔴 لون الصف إذا فيه anomaly
    if (run.anomaly_rate && run.anomaly_rate > 0) {
      tr.style.background = "rgba(231, 76, 60, 0.15)";
    }

    tr.innerHTML = `
      <td>
        <strong>${getDisplayName(run)}</strong>
        <div class="run-id">${run.run_id}</div>
      </td>
      <td>${run.status}</td>
      <td>${new Date(run.created_at).toLocaleString()}</td>
      <td>${(run.anomaly_rate || 0).toFixed(3)}</td>
      <td>
        <button onclick="openRun('${run.run_id}')">
          View
        </button>
      </td>
    `;

    tbody.appendChild(tr);
  });
}

// =====================================
// OPEN RUN → GO TO ANALYSIS
// =====================================
function openRun(runId) {
  localStorage.setItem("csd_run_id", runId);
  window.location.href = "analysis.html";
}

// =====================================
// MODEL SNAPSHOT
// =====================================
function initModel(model) {
  const healthPct = Math.max(0, Math.min(100, Number(model.health_pct || 0)));

  document.getElementById("mName").textContent = model.name || "—";
  document.getElementById("mVersion").textContent = model.version || "—";
  document.getElementById("mLastTrain").textContent = formatDateTime(model.last_training);
  document.getElementById("mAcc").textContent = model.accuracy == null
    ? "Not recorded"
    : formatPercent(model.accuracy);
  document.getElementById("predSuccess").textContent = formatPercent(model.prediction_success);

  document.getElementById("healthFill").style.width = `${healthPct}%`;
  document.getElementById("healthNote").textContent =
    model.note || "No model metadata available.";
}

// =====================================
// 3D BACKGROUND
// =====================================
let scene, camera, renderer, sphere, stars;

function initBG() {
  const container = document.getElementById("canvas_container");
  if (!container) return;

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    1200
  );
  camera.position.set(0, 0, 260);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.7;
  controls.enablePan = false;
  controls.enableZoom = false;

  const loader = new THREE.TextureLoader();
  const bg = loader.load("https://i.ibb.co/HC0vxMw/sky2.jpg");

  sphere = new THREE.Mesh(
    new THREE.SphereGeometry(520, 48, 48),
    new THREE.MeshBasicMaterial({
      side: THREE.BackSide,
      map: bg
    })
  );
  scene.add(sphere);

  const starsGeo = new THREE.BufferGeometry();
  const positions = [];
  for (let i = 0; i < 600; i++) {
    positions.push(
      (Math.random() - 0.5) * 1000,
      (Math.random() - 0.5) * 1000,
      (Math.random() - 0.5) * 1000
    );
  }

  starsGeo.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );

  stars = new THREE.Points(
    starsGeo,
    new THREE.PointsMaterial({
      color: "#ffffff",
      size: 1.6,
      opacity: 0.9,
      transparent: true
    })
  );
  scene.add(stars);

  animateBG();
}

function animateBG() {
  requestAnimationFrame(animateBG);
  if (sphere) sphere.rotation.y += 0.0007;
  if (stars) stars.rotation.y += 0.00035;
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  if (!camera || !renderer) return;
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
