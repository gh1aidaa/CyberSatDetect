// =====================================
// CyberSatDetect | Anomaly Dashboard
// FINAL VERSION - MATCHES api.py LOGIC
// =====================================

// ---------- CONFIG ----------
const API_BASE = window.location.origin;
const POLL_INTERVAL = 3000;
const BAIL_AFTER_SEC = 90;

// ---------- ELEMENTS ----------
const fileNameLabel    = document.getElementById("fileNameLabel");
const cleanStatusLabel = document.getElementById("cleanStatusLabel");
const thresholdLabel   = document.getElementById("thresholdLabel");

const detStatus        = document.getElementById("detStatus");
const btnGoAnalysis    = document.getElementById("btnGoAnalysis");
const btnRunDetection  = document.getElementById("btnRun");
const btnBailAnalyze   = document.getElementById("btnBailAnalyze");

const sumTotal         = document.getElementById("sumTotal");
const sumAnom          = document.getElementById("sumAnom");
const sumHigh          = document.getElementById("sumHigh");
const sumMed           = document.getElementById("sumMed");

const anomBody         = document.getElementById("anomBody");
const anomNote         = document.getElementById("anomNote");

// ---------- CONSTANTS ----------
const SYSTEM_THRESHOLD = "System default";

/** Server run status (from GET /runs/{id}) — used to enable Run & avoid stale label checks */
let currentRunStatus = "";
let statusPollId = null;
let analyzingT0 = null;

// ---------- AUTH ----------
function getToken() {
  return localStorage.getItem("token");
}

function requireLogin() {
  window.location.href = "login.html";
}

// ---------- CONTEXT ----------
const runId    = localStorage.getItem("csd_run_id");
const fileName = localStorage.getItem("csd_file_name");

if (!getToken()) requireLogin();
if (!runId) {
  detStatus.textContent =
    "❌ No run found. Please upload telemetry data first.";
}

// ---------- INIT ----------
function initMeta() {
  fileNameLabel.textContent    = fileName || "—";
  cleanStatusLabel.textContent = "—";
  thresholdLabel.textContent   = SYSTEM_THRESHOLD;

  sumTotal.textContent = "—";
  sumAnom.textContent  = "—";
  sumHigh.textContent  = "—";
  sumMed.textContent   = "—";

  detStatus.textContent = "⏳ Loading run status…";

  btnRunDetection.disabled = true;
  btnGoAnalysis.disabled = true;
}

function stopStatusPolling() {
  if (statusPollId !== null) {
    clearInterval(statusPollId);
    statusPollId = null;
  }
}

// ---------- RUN DETECTION ----------
async function runDetection() {
  if (!btnRunDetection || !runId) return;

  if (btnRunDetection.disabled) return;

  btnRunDetection.disabled = true;
  detStatus.textContent = "🚀 Running anomaly detection…";

  try {
    const res = await fetch(`${API_BASE}/runs/${runId}/analyze`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${getToken()}`
      }
    });

    const text = await res.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch (_) {
      data = { _raw: text };
    }

    if (res.status === 401) {
      localStorage.removeItem("token");
      btnRunDetection.disabled = false;
      requireLogin();
      return;
    }

    if (!res.ok) {
      let msg = res.statusText || "Detection failed";
      if (typeof data.detail === "string") {
        msg = data.detail;
      } else if (Array.isArray(data.detail)) {
        msg = data.detail.map((e) => e.msg || JSON.stringify(e)).join("; ");
      } else if (typeof data._raw === "string" && data._raw.length && data._raw.length < 600) {
        msg = data._raw;
      }
      throw new Error(msg);
    }

    // Sync API completes analysis in this request — show results immediately
    currentRunStatus = "DONE";
    cleanStatusLabel.textContent = "DONE";
    detStatus.textContent = "✅ Detection completed.";
    stopStatusPolling();
    await loadResults();
    btnGoAnalysis.disabled = false;
    btnRunDetection.disabled = true;
  } catch (err) {
    console.error(err);
    detStatus.textContent = `❌ ${err.message || err}`;
    btnRunDetection.disabled = false;
    try {
      await pollStatus();
    } catch (_) {
      /* ignore */
    }
  }
}

// ---------- POLL STATUS ----------
async function pollStatus() {
  try {
    const res = await fetch(`${API_BASE}/runs/${runId}`, {
      headers: {
        "Authorization": `Bearer ${getToken()}`
      }
    });

    if (res.status === 401) {
      localStorage.removeItem("token");
      requireLogin();
      return;
    }

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);

    currentRunStatus = data.status;
    cleanStatusLabel.textContent = data.status;

    if (data.status !== "ANALYZING") {
      analyzingT0 = null;
      if (btnBailAnalyze) btnBailAnalyze.style.display = "none";
    }

    // ✅ تحليل انتهى (reload / عودة للصفحة)
    if (data.status === "DONE") {
      detStatus.textContent = "✅ Detection completed.";
      btnRunDetection.disabled = true;
      await loadResults();
      btnGoAnalysis.disabled = false;
      stopStatusPolling();
      return;
    }

    if (data.status === "ANALYZING") {
      btnRunDetection.disabled = true;
      btnGoAnalysis.disabled = true;
      if (analyzingT0 === null) analyzingT0 = Date.now();
      const sec = Math.floor((Date.now() - analyzingT0) / 1000);
      detStatus.textContent = `⏳ Detection is running… (${sec}s).`;
      if (sec >= BAIL_AFTER_SEC && btnBailAnalyze) {
        btnBailAnalyze.style.display = "inline-block";
      }
      return;
    }

    if (data.status === "PREPARING") {
      btnRunDetection.disabled = true;
      detStatus.textContent = "⏳ Cleaning telemetry…";
      return;
    }

    if (data.status === "FAILED") {
      btnRunDetection.disabled = true;
      btnGoAnalysis.disabled = true;
      detStatus.textContent =
        "❌ Last step failed. Upload again from Data Upload.";
      return;
    }

    // 🟢 جاهز للكشف: ملف مرفوع أو نظيف (الموديل لم يُشغَّل بعد)
    if (data.status === "UPLOADED" || data.status === "CLEANED") {
      btnRunDetection.disabled = false;
      btnGoAnalysis.disabled = true;
      detStatus.textContent =
        data.status === "CLEANED"
          ? "🟢 Data cleaned. Click Run Detection to start the model."
          : "🟢 Telemetry loaded. Click Run Detection to start the model.";
      return;
    }

    btnRunDetection.disabled = true;
    detStatus.textContent = `🧠 System status: ${data.status}`;
  } catch (err) {
    console.error(err);
    detStatus.textContent =
      "⚠️ Unable to fetch system status.";
  }
}

// ---------- LOAD RESULTS ----------
async function loadResults() {
  try {
    const res = await fetch(
      `${API_BASE}/runs/${runId}/results?preview_rows=200`,
      {
        headers: {
          Authorization: `Bearer ${getToken()}`,
        },
      }
    );

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);

    const channels = Array.isArray(data.channels) ? data.channels : [];

    let totalWindows = 0;
    let totalAnom = 0;
    for (const ch of channels) {
      totalWindows += Number(ch.num_windows) || 0;
      totalAnom += Number(ch.num_anomalies) || 0;
    }

    if (channels.length && channels[0].threshold != null) {
      const t = Number(channels[0].threshold);
      if (Number.isFinite(t)) {
        thresholdLabel.textContent = t.toFixed(6);
      }
    }

    const flat = [];
    for (const ch of channels) {
      const prev = Array.isArray(ch.rows_preview) ? ch.rows_preview : [];
      for (const r of prev) {
        flat.push({ ...r, _channel: ch.channel_name });
      }
    }

    let high = 0;
    let med = 0;
    anomBody.innerHTML = "";
    let rowNum = 0;

    flat.forEach((r) => {
      const score = Number(r.score ?? r.anomaly_score) || 0;
      const isA =
        r.is_anomaly === 1 ||
        r.is_anomaly === true ||
        r.is_anomaly === "1";
      if (!isA) return;

      rowNum++;
      const sevU = String(r.severity || "").toUpperCase();
      if (sevU === "HIGH") high++;
      else if (sevU === "MEDIUM") med++;

      const sevLabel =
        r.severity && String(r.severity).trim() !== ""
          ? String(r.severity)
          : "—";

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${rowNum}</td>
        <td>${score.toFixed(4)}</td>
        <td>${sevLabel}</td>
        <td>${r._channel || "—"}</td>
      `;
      anomBody.appendChild(tr);
    });

    sumTotal.textContent = totalWindows;
    sumAnom.textContent = totalAnom;
    sumHigh.textContent = high;
    sumMed.textContent = med;

    if (totalAnom > 0 && rowNum === 0) {
      anomNote.textContent =
        "⚠️ Anomalies are reported in the summary; first preview rows are normal-only. Open Analysis for the full segment list.";
    } else {
      anomNote.textContent =
        totalAnom > 0
          ? "⚠️ Anomalies detected. Review summary below."
          : "✅ No anomalies detected in this run.";
    }
  } catch (err) {
    console.error(err);
    anomNote.textContent = "❌ Failed to load anomaly results.";
  }
}

// ---------- BAIL (stuck ANALYZING) ----------
async function bailAnalyze() {
  if (!runId || !getToken()) return;
  try {
    if (btnBailAnalyze) btnBailAnalyze.disabled = true;
    detStatus.textContent = "⏳ Resetting stuck run…";
    const res = await fetch(`${API_BASE}/runs/${runId}/bail-analyze`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${getToken()}`,
      },
    });
    let data = {};
    try {
      data = await res.json();
    } catch (_) {
      data = {};
    }
    if (res.status === 401) {
      localStorage.removeItem("token");
      requireLogin();
      return;
    }
    if (!res.ok) {
      throw new Error(
        typeof data.detail === "string" ? data.detail : "Reset failed"
      );
    }
    analyzingT0 = null;
    if (btnBailAnalyze) {
      btnBailAnalyze.style.display = "none";
      btnBailAnalyze.disabled = false;
    }
    detStatus.textContent =
      "⚠️ Run marked as failed. Upload the file again from Data Upload, then return here.";
    currentRunStatus = "FAILED";
    cleanStatusLabel.textContent = "FAILED";
    btnRunDetection.disabled = true;
  } catch (e) {
    detStatus.textContent = `❌ ${e.message || e}`;
    if (btnBailAnalyze) btnBailAnalyze.disabled = false;
  }
}

// ---------- NAV ----------
btnGoAnalysis.addEventListener("click", () => {
  window.location.href = "analysis.html";
});

btnRunDetection.addEventListener("click", runDetection);
if (btnBailAnalyze) {
  btnBailAnalyze.addEventListener("click", bailAnalyze);
}

// ---------- START ----------
initMeta();
if (runId) {
  pollStatus();
  statusPollId = setInterval(pollStatus, POLL_INTERVAL);
}

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

document.addEventListener("DOMContentLoaded", initSpaceBackground);
