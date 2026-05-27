// =====================================
// CyberSatDetect | FULL FIXED JS
// =====================================

const API_BASE = window.location.origin;

let chart = null;
let currentFile = null;
let isTraining = false;

let anomChart = null;
let currentAnomFile = null;

function ui() {
  return {
    fileSelect: document.getElementById("normalFileSelect"),
    approveBtn: document.getElementById("approveBtn"),
    rejectBtn: document.getElementById("rejectBtn"),
    anomFileSelect: document.getElementById("anomalyFileSelect"),
    approveAnomBtn: document.getElementById("approveAnomBtn"),
    rejectAnomBtn: document.getElementById("rejectAnomBtn"),
    buildBtn: document.getElementById("buildBtn"),
    trainBtn: document.getElementById("trainBtn"),
    trainStatus: document.getElementById("trainStatus"),
    statMean: document.getElementById("statMean"),
    statStd: document.getElementById("statStd"),
    statMin: document.getElementById("statMin"),
    statMax: document.getElementById("statMax"),
    anomStatMean: document.getElementById("anomStatMean"),
    anomStatStd: document.getElementById("anomStatStd"),
    anomStatMin: document.getElementById("anomStatMin"),
    anomStatMax: document.getElementById("anomStatMax"),
  };
}

function setStatsEmpty() {
  const refs = ui();
  refs.statMean.innerText = "-";
  refs.statStd.innerText = "-";
  refs.statMin.innerText = "-";
  refs.statMax.innerText = "-";
}

function clearChart() {
  if (chart) {
    chart.destroy();
    chart = null;
  }
  const canvas = document.getElementById("dataChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function clearAnomChart() {
  if (anomChart) {
    anomChart.destroy();
    anomChart = null;
  }
  const canvas = document.getElementById("anomDataChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function setReviewButtonsEnabled(enabled) {
  const refs = ui();
  const canUse = enabled && !isTraining;
  refs.approveBtn.disabled = !canUse;
  refs.rejectBtn.disabled = !canUse;
}

function setAnomReviewButtonsEnabled(enabled) {
  const refs = ui();
  const canUse = enabled && !isTraining;
  if (refs.approveAnomBtn) refs.approveAnomBtn.disabled = !canUse;
  if (refs.rejectAnomBtn) refs.rejectAnomBtn.disabled = !canUse;
}

function setTrainStatus(message, type = "") {
  const refs = ui();
  refs.trainStatus.textContent = message || "";
  refs.trainStatus.className = "train-status";
  if (type) refs.trainStatus.classList.add(type);
}

function setTrainingState(running) {
  isTraining = running;
  const refs = ui();

  refs.buildBtn.disabled = running;
  refs.trainBtn.disabled = running;
  refs.fileSelect.disabled = running;
  if (refs.anomFileSelect) refs.anomFileSelect.disabled = running;

  setReviewButtonsEnabled(!!currentFile);
  setAnomReviewButtonsEnabled(!!currentAnomFile);

  const modelButtons = document.querySelectorAll("#modelTable button");
  modelButtons.forEach(btn => {
    btn.disabled = running;
  });
}

function setNoFilesState() {
  const refs = ui();
  currentFile = null;
  setStatsEmpty();
  clearChart();
  setReviewButtonsEnabled(false);
  refs.fileSelect.innerHTML = "";

  const opt = document.createElement("option");
  opt.value = "";
  opt.textContent = "No files available";
  refs.fileSelect.appendChild(opt);
}

// =====================
// AUTH
// =====================

function getToken() {
  return localStorage.getItem("token");
}

function requireLogin() {
  const loginUrl = new URL("login.html", window.location.href);
  const nextPath = window.location.pathname + window.location.search + window.location.hash;
  loginUrl.searchParams.set("next", nextPath);
  window.location.assign(loginUrl.toString());
}

// =====================
// BUILD DATASET
// =====================

window.buildDataset = async function () {
  if (!confirm("Build dataset now?")) return;

  try {
    const res = await fetch(`${API_BASE}/admin/continual/build-dataset`, {
      method: "POST",
      headers: { Authorization: `Bearer ${getToken()}` }
    });

    if (!res.ok) throw new Error();

    alert("Dataset built successfully ✅");
    await loadDatasets();

  } catch (err) {
    console.error(err);
    alert("Build failed ❌");
  }
};

// =====================
// LOAD USER
// =====================

async function loadUser() {
  const res = await fetch(`${API_BASE}/auth/me`, {
    headers: { Authorization: `Bearer ${getToken()}` }
  });

  if (!res.ok) {
    requireLogin();
    return false;
  }

  const user = await res.json();
  if (user.role !== "ADMIN") {
    showAdminAccessDenied(document.querySelector(".admin-page"), user.role);
    return false;
  }

  return true;
}

// =====================
// LOAD FILES
// =====================

async function loadNormalFiles() {

  const res = await fetch(`${API_BASE}/admin/continual/normal-files`, {
    headers: { Authorization: `Bearer ${getToken()}` }
  });

  const data = await res.json();

  const refs = ui();
  const select = refs.fileSelect;
  select.innerHTML = "";

  (data.files || []).forEach(file => {
    const opt = document.createElement("option");
    opt.value = file;
    opt.textContent = file;
    select.appendChild(opt);
  });

  if (data.files?.length > 0) {
    setReviewButtonsEnabled(true);
    currentFile = data.files[0];
    await loadNormalData(currentFile);
    return;
  }

  setNoFilesState();
}

// =====================
// LOAD DATA
// =====================

async function loadNormalData(file) {

  if (!file) {
    setNoFilesState();
    return;
  }

  currentFile = file;

  const res = await fetch(
    `${API_BASE}/admin/continual/normal-data?file=${encodeURIComponent(file)}`,
    { headers: { Authorization: `Bearer ${getToken()}` } }
  );

  if (!res.ok) {
    setNoFilesState();
    return;
  }

  const data = await res.json();
  const refs = ui();

  refs.statMean.innerText = data.mean?.toFixed?.(4) || "-";
  refs.statStd.innerText = data.std?.toFixed?.(4) || "-";
  refs.statMin.innerText = data.min?.toFixed?.(4) || "-";
  refs.statMax.innerText = data.max?.toFixed?.(4) || "-";

  let values = data.data || [];

  if (Array.isArray(values[0])) values = values[0];

  values = values.map(Number).filter(v => !isNaN(v));

  drawChart(values);
}

// =====================
// DRAW CHART
// =====================

function drawChart(values) {

  const ctx = document.getElementById("dataChart").getContext("2d");

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: values.map((_, i) => i),
      datasets: [{
        data: values,
        borderColor: "#2ecc71",
        borderWidth: 1,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } }
    }
  });
}

function drawAnomChart(values) {
  const ctx = document.getElementById("anomDataChart").getContext("2d");
  if (anomChart) anomChart.destroy();
  anomChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: values.map((_, i) => i),
      datasets: [{
        data: values,
        borderColor: "#e74c3c",
        borderWidth: 1,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } }
    }
  });
}

// =====================
// APPROVE / REJECT
// =====================

window.approveFile = async function () {
  if (!currentFile || isTraining) return;

  setReviewButtonsEnabled(false);

  try {
    const res = await fetch(
      `${API_BASE}/admin/continual/approve-normal?file=${encodeURIComponent(currentFile)}`,
      { method: "POST", headers: { Authorization: `Bearer ${getToken()}` } }
    );

    if (!res.ok) {
      if (res.status === 404) {
        alert("File already moved or not found.");
      } else {
        throw new Error("Approve failed");
      }
    }

    await loadNormalFiles();
    await loadDatasets();
  } catch (err) {
    console.error(err);
    alert("Approve failed ❌");
  }
};

window.rejectFile = async function () {
  if (!currentFile || isTraining) return;

  setReviewButtonsEnabled(false);

  try {
    const res = await fetch(
      `${API_BASE}/admin/continual/reject-normal?file=${encodeURIComponent(currentFile)}`,
      { method: "POST", headers: { Authorization: `Bearer ${getToken()}` } }
    );

    if (!res.ok) {
      if (res.status === 404) {
        alert("File already removed or not found.");
      } else {
        throw new Error("Reject failed");
      }
    }

    await loadNormalFiles();
    await loadDatasets();
  } catch (err) {
    console.error(err);
    alert("Reject failed ❌");
  }
};

// =====================
// ANOMALY: LOAD FILES / DATA
// =====================

async function loadAnomalyFiles() {
  const res = await fetch(`${API_BASE}/admin/continual/anomaly-files`, {
    headers: { Authorization: `Bearer ${getToken()}` }
  });
  const data = await res.json();
  const refs = ui();
  const select = refs.anomFileSelect;
  if (!select) return;
  select.innerHTML = "";

  (data.files || []).forEach(file => {
    const opt = document.createElement("option");
    opt.value = file;
    opt.textContent = file;
    select.appendChild(opt);
  });

  if (data.files?.length > 0) {
    setAnomReviewButtonsEnabled(true);
    currentAnomFile = data.files[0];
    await loadAnomalyData(currentAnomFile);
    return;
  }

  currentAnomFile = null;
  clearAnomChart();
  setAnomReviewButtonsEnabled(false);
  const opt = document.createElement("option");
  opt.value = "";
  opt.textContent = "No files available";
  select.appendChild(opt);
}

async function loadAnomalyData(file) {
  if (!file) return;
  currentAnomFile = file;
  const res = await fetch(
    `${API_BASE}/admin/continual/anomaly-data?file=${encodeURIComponent(file)}`,
    { headers: { Authorization: `Bearer ${getToken()}` } }
  );
  if (!res.ok) return;
  const data = await res.json();
  const refs = ui();
  if (refs.anomStatMean) refs.anomStatMean.innerText = data.mean?.toFixed?.(4) || "-";
  if (refs.anomStatStd) refs.anomStatStd.innerText = data.std?.toFixed?.(4) || "-";
  if (refs.anomStatMin) refs.anomStatMin.innerText = data.min?.toFixed?.(4) || "-";
  if (refs.anomStatMax) refs.anomStatMax.innerText = data.max?.toFixed?.(4) || "-";
  let values = data.data || [];
  if (Array.isArray(values[0])) values = values[0];
  values = values.map(Number).filter(v => !isNaN(v));
  drawAnomChart(values);
}

window.approveAnomaly = async function () {
  if (!currentAnomFile || isTraining) return;
  setAnomReviewButtonsEnabled(false);
  try {
    const res = await fetch(
      `${API_BASE}/admin/continual/approve-anomaly?file=${encodeURIComponent(currentAnomFile)}`,
      { method: "POST", headers: { Authorization: `Bearer ${getToken()}` } }
    );
    if (!res.ok) throw new Error("Approve anomaly failed");
    await loadAnomalyFiles();
    await loadAnomalyDatasets();
  } catch (err) {
    console.error(err);
    alert("Approve anomaly failed ❌");
  }
};

window.rejectAnomaly = async function () {
  if (!currentAnomFile || isTraining) return;
  setAnomReviewButtonsEnabled(false);
  try {
    const res = await fetch(
      `${API_BASE}/admin/continual/reject-anomaly?file=${encodeURIComponent(currentAnomFile)}`,
      { method: "POST", headers: { Authorization: `Bearer ${getToken()}` } }
    );
    if (!res.ok) throw new Error("Reject anomaly failed");
    await loadAnomalyFiles();
    await loadAnomalyDatasets();
  } catch (err) {
    console.error(err);
    alert("Reject anomaly failed ❌");
  }
};

// =====================
// DATASETS
// =====================

async function loadDatasets() {

  const res = await fetch(`${API_BASE}/admin/continual/datasets`, {
    headers: { Authorization: `Bearer ${getToken()}` }
  });

  const data = await res.json();

  const tbody = document.querySelector("#datasetTable tbody");
  tbody.innerHTML = "";

  (data.datasets || []).forEach(ds => {
    tbody.innerHTML += `<tr><td>${ds}</td><td>Ready</td></tr>`;
  });
}

async function loadAnomalyDatasets() {
  const res = await fetch(`${API_BASE}/admin/continual/anomaly-datasets`, {
    headers: { Authorization: `Bearer ${getToken()}` }
  });
  const data = await res.json();
  const tbody = document.querySelector("#anomDatasetTable tbody");
  if (!tbody) return;
  tbody.innerHTML = "";
  (data.datasets || []).forEach(ds => {
    tbody.innerHTML += `<tr><td>${ds}</td><td>Ready</td></tr>`;
  });
}

// =====================
// TRAIN
// =====================

window.startTraining = async function () {
  if (isTraining) return;

  setTrainingState(true);
  setTrainStatus("Training model in progress...", "running");

  try {
    const res = await fetch(`${API_BASE}/admin/continual/train`, {
      method: "POST",
      headers: { Authorization: `Bearer ${getToken()}` }
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || "Training failed");
    }

    setTrainStatus("Training completed successfully.", "success");
    await loadModels();
  } catch (err) {
    console.error(err);
    setTrainStatus("Training failed. Check backend logs.", "error");
  } finally {
    setTrainingState(false);
  }
};

// =====================
// MODELS
// =====================

async function loadModels() {

  const res = await fetch(`${API_BASE}/admin/continual/models`, {
    headers: { Authorization: `Bearer ${getToken()}` }
  });

  const data = await res.json();

  const tbody = document.querySelector("#modelTable tbody");
  tbody.innerHTML = "";

  (data.models || []).forEach(m => {

    tbody.innerHTML += `
      <tr>
        <td>${m.version}</td>
        <td>${m.status}</td>
        <td>${m.created_at || "-"}</td>
        <td>${m.approved_at || "-"}</td>
        <td>
          ${m.status === "PENDING" ? `<button onclick="approveModel('${m.version}')">Approve</button>` : ""}
          <button onclick="rollbackModel('${m.version}')">Rollback</button>
        </td>
      </tr>
    `;
  });
}

window.approveModel = async function (v) {
  if (isTraining) return;
  await fetch(`${API_BASE}/admin/continual/approve/${v}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${getToken()}` }
  });
  loadModels();
};

window.rollbackModel = async function (v) {
  if (isTraining) return;
  await fetch(`${API_BASE}/admin/continual/rollback/${v}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${getToken()}` }
  });
  loadModels();
};

// =====================================
// 🔥 FIXED 3D BACKGROUND
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

  // ✅ الحل النهائي
  renderer.domElement.style.position = "fixed";
  renderer.domElement.style.top = "0";
  renderer.domElement.style.left = "0";
  renderer.domElement.style.zIndex = "-2"; // 🔥 كان -1
  renderer.domElement.style.pointerEvents = "none";

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
  if (stars) stars.rotation.y += 0.00035;

  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  if (!camera || !renderer) return;

  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

document.addEventListener("DOMContentLoaded", async () => {

  initSpaceBackground(); // الخلفية

  const ok = await loadUser(); // تحقق تسجيل الدخول
  if (!ok) return;

  const refs = ui();
  refs.fileSelect.addEventListener("change", async (e) => {
    await loadNormalData(e.target.value);
  });
  if (refs.anomFileSelect) {
    refs.anomFileSelect.addEventListener("change", async (e) => {
      await loadAnomalyData(e.target.value);
    });
  }

  await loadNormalFiles(); // 🔥 هذا المهم
  await loadAnomalyFiles();
  await loadDatasets();
  await loadAnomalyDatasets();
  await loadModels();

  setTrainingState(false);
  setTrainStatus("");

});;