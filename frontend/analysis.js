// =====================================
// CyberSatDetect | Analysis Page
// Compatible with NEW API (per-channel results)
// =====================================

const API_BASE = window.location.origin;

function getToken() {
  return localStorage.getItem("token");
}

function requireLogin() {
  window.location.href = "login.html";
}

const runId = localStorage.getItem("csd_run_id");
if (!getToken()) requireLogin();
if (!runId) {
  alert("No analysis run found. Please upload data first.");
  window.location.href = "data-upload.html";
}

// ---------- ELEMENTS ----------
const fileNameLabel = document.getElementById("fileNameLabel");
const runTimeLabel  = document.getElementById("runTimeLabel");

const onlyAnoms = document.getElementById("onlyAnomalies");
const segBody   = document.getElementById("segBody");
const tableNote = document.getElementById("tableNote");

const mMean     = document.getElementById("mMean");
const mStd      = document.getElementById("mStd");
const mMin      = document.getElementById("mMin");
const mMax      = document.getElementById("mMax");
const mScore    = document.getElementById("mScore");
const mFlag     = document.getElementById("mFlag");
const mSeverity = document.getElementById("mSeverity");
const mChannel  = document.getElementById("mChannel");

const chartHint = document.getElementById("chartHint");
const btnExport = document.getElementById("btnExport");

let segments = [];
let chart = null;

function setExportEnabled(ok) {
  if (btnExport) btnExport.disabled = !ok;
}

// ---------- LOAD ----------
async function loadAnalysis() {
  tableNote.textContent = "Loading analysis results…";
  setExportEnabled(false);

  const metaRes = await fetch(`${API_BASE}/runs/${runId}`, {
    headers: { "Authorization": `Bearer ${getToken()}` }
  });

  const meta = await metaRes.json();
  if (!metaRes.ok) {
    tableNote.textContent = "Failed to load run info.";
    return;
  }

  fileNameLabel.textContent =
    localStorage.getItem("csd_file_name") || "—";
  runTimeLabel.textContent = meta.created_at || "—";

  if (meta.status !== "DONE") {
    tableNote.textContent =
      `Results not ready. Current status: ${meta.status}`;
    return;
  }

  const res = await fetch(
    `${API_BASE}/runs/${runId}/results?preview_rows=500`,
    {
      headers: { Authorization: `Bearer ${getToken()}` },
    }
  );

  const data = await res.json();
  if (!res.ok) {
    tableNote.textContent = "Failed to load results.";
    return;
  }

  segments = [];
  let segCounter = 1;

  data.channels.forEach((channel) => {
    const prev = Array.isArray(channel.rows_preview)
      ? channel.rows_preview
      : [];
    prev.forEach((row, ord) => {
      const wi = Number(row.window_index);
      segments.push({
        segment: segCounter++,
        window_index: Number.isFinite(wi) ? wi : ord,
        score: Number(row.score),
        is_anomaly:
          row.is_anomaly == 1 ||
          row.is_anomaly === true ||
          row.is_anomaly === "1",
        severity: row.severity,
        channel: channel.channel_name,
      });
    });
  });

  setExportEnabled(true);
  renderTable();
}

function renderTable() {
  segBody.innerHTML = "";

  if (!segments.length) {
    tableNote.textContent = "No analysis data available for this run.";
    return;
  }

  const filtered = onlyAnoms.checked
    ? segments.filter(s => s.is_anomaly)
    : segments;

  filtered.forEach((seg) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${seg.segment}</td>
      <td>${seg.score.toFixed(4)}</td>
      <td>${seg.is_anomaly ? "Anomalous" : "Normal"}</td>
      <td>${seg.severity}</td>
      <td>${seg.channel}</td>
      <td><button type="button" data-seg="${seg.segment}">View</button></td>
    `;
    segBody.appendChild(tr);
  });

  tableNote.textContent = "";
}

onlyAnoms.addEventListener("change", renderTable);

segBody.addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-seg]");
  if (!btn) return;
  const id = Number(btn.dataset.seg);
  const seg = segments.find((s) => s.segment === id);
  showSegment(seg);
});

async function showSegment(seg) {
  if (!seg) return;

  mScore.textContent = seg.score.toFixed(4);
  mFlag.textContent = seg.is_anomaly ? "Anomalous" : "Normal";
  mSeverity.textContent = seg.severity;
  mChannel.textContent = seg.channel;

  mMean.textContent = "…";
  mStd.textContent = "…";
  mMin.textContent = "…";
  mMax.textContent = "…";
  chartHint.textContent = "Loading telemetry trace…";

  const canvas = document.getElementById("segChart");
  if (!canvas || typeof Chart === "undefined") {
    chartHint.textContent = "Chart library not available.";
    return;
  }
  const ctx = canvas.getContext("2d");
  if (chart) chart.destroy();

  try {
    const url = new URL(`${API_BASE}/runs/${runId}/segment-window`);
    url.searchParams.set("channel", seg.channel);
    url.searchParams.set("window_index", String(seg.window_index));

    const res = await fetch(url.toString(), {
      headers: { Authorization: `Bearer ${getToken()}` },
    });
    const payload = await res.json();
    if (!res.ok) {
      const d = payload.detail;
      throw new Error(typeof d === "string" ? d : "Failed to load segment window");
    }

    const vals = Array.isArray(payload.values) ? payload.values : [];

    if (
      payload.mean != null &&
      payload.std != null &&
      payload.min != null &&
      payload.max != null
    ) {
      mMean.textContent = Number(payload.mean).toFixed(6);
      mStd.textContent = Number(payload.std).toFixed(6);
      mMin.textContent = Number(payload.min).toFixed(6);
      mMax.textContent = Number(payload.max).toFixed(6);
    } else {
      mMean.textContent = "—";
      mStd.textContent = "—";
      mMin.textContent = "—";
      mMax.textContent = "—";
    }

    if (vals.length === 0) {
      chartHint.textContent = "No samples available for this window.";
      chart = new Chart(ctx, {
        type: "line",
        data: { labels: [], datasets: [] },
        options: { plugins: { legend: { display: false } } },
      });
      return;
    }

    if (payload.start != null && payload.end != null) {
      chartHint.textContent = `Series indices ${payload.start}–${payload.end} (${vals.length} samples)`;
    } else {
      chartHint.textContent = `Window ${seg.window_index} (${vals.length} samples)`;
    }

    const labels = vals.map((_, i) => String(i + 1));
    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Telemetry",
            data: vals,
            borderColor: "rgba(0,191,255,0.95)",
            backgroundColor: "rgba(0,191,255,0.12)",
            fill: true,
            tension: 0.12,
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: "#BFD6E8" },
          },
        },
        scales: {
          x: {
            ticks: { color: "#BFD6E8", maxTicksLimit: 12 },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
          y: {
            ticks: { color: "#BFD6E8" },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
        },
      },
    });
  } catch (e) {
    console.error(e);
    mMean.textContent = "—";
    mStd.textContent = "—";
    mMin.textContent = "—";
    mMax.textContent = "—";
    chartHint.textContent = e.message || "Could not load segment trace.";
    chart = new Chart(ctx, {
      type: "line",
      data: { labels: [], datasets: [] },
      options: { plugins: { legend: { display: false } } },
    });
  }
}

if (btnExport) {
  btnExport.addEventListener("click", () => {
    window.location.href = "reports.html";
  });
}

document.addEventListener("DOMContentLoaded", () => {
  loadAnalysis();
  initSpaceBackground();
});
// =====================================
// CyberSatDetect | Space Background
// =====================================

let scene, camera, renderer, stars, sphere;

function initSpaceBackground() {
  const container = document.getElementById("canvas_container");
  if (!container) return;

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    1500
  );
  camera.position.set(0, 0, 260);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;
  controls.enableZoom = false;
  controls.enablePan = false;

  const loader = new THREE.TextureLoader();
  const bgTexture = loader.load("https://i.ibb.co/HC0vxMw/sky2.jpg");

  sphere = new THREE.Mesh(
    new THREE.SphereGeometry(520, 48, 48),
    new THREE.MeshBasicMaterial({
      side: THREE.BackSide,
      map: bgTexture
    })
  );
  scene.add(sphere);

  const starsGeo = new THREE.BufferGeometry();
  const positions = [];

  for (let i = 0; i < 700; i++) {
    positions.push(
      (Math.random() - 0.5) * 1200,
      (Math.random() - 0.5) * 1200,
      (Math.random() - 0.5) * 1200
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

  animateSpace();
}

function animateSpace() {
  requestAnimationFrame(animateSpace);
  if (sphere) sphere.rotation.y += 0.0007;
  if (stars)  stars.rotation.y  += 0.00035;
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  if (!camera || !renderer) return;
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
