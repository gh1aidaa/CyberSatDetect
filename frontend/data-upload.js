// =====================================
// CyberSatDetect | Data Upload Script
// Upload RAW + prepare/clean (no model) → anomaly page runs detection
// =====================================

console.log("DATA-UPLOAD.JS LOADED");

// ========= CONFIG =========
const API_BASE = window.location.origin;
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB
const REDIRECT_DELAY = 1800; // 1.8 seconds

// ========= ELEMENTS =========
const fileInput  = document.getElementById("fileInput");
const fileLabel  = document.getElementById("fileLabelText");
const fileStatus = document.getElementById("fileStatus");
const btnStart   = document.getElementById("btnStart");

let selectedFile = null;
let redirectTimer = null;

// ========= AUTH HELPERS =========
function getToken() {
  return localStorage.getItem("token");
}

function redirectToLogin() {
  window.location.href = "login.html";
}

// ========= FILE VALIDATION =========
fileInput.addEventListener("change", () => {
  if (redirectTimer) {
    clearTimeout(redirectTimer);
    redirectTimer = null;
  }

  const file = fileInput.files[0];
  if (!file) return;

  const name = file.name.toLowerCase();

  if (file.size > MAX_FILE_SIZE) {
    showError("❌ File too large. Maximum allowed size is 50 MB.");
    return;
  }

  if (!name.endsWith(".csv") && !name.endsWith(".npy")) {
    showError("❌ Unsupported file type. Only CSV or NPY telemetry files are allowed.");
    return;
  }

  if (
    name.includes("..") ||
    name.endsWith(".exe") ||
    name.endsWith(".js") ||
    name.endsWith(".bat")
  ) {
    showError("🚫 Suspicious filename blocked. Please rename the file.");
    return;
  }

  selectedFile = file;
  fileLabel.textContent = file.name;
  fileStatus.textContent = "✔ File accepted. Ready to upload.";
  fileStatus.style.color = "#6cff9f";
  btnStart.disabled = false;
});

// ---------- ERROR HANDLER ----------
function showError(message) {
  selectedFile = null;
  btnStart.disabled = true;
  fileStatus.textContent = message;
  fileStatus.style.color = "#ff6b6b";
}

// ========= UPLOAD + PREPARE (clean) =========
btnStart.addEventListener("click", async () => {
  if (!selectedFile) return;

  const token = getToken();
  if (!token) {
    showError("🔒 Please login first.");
    redirectToLogin();
    return;
  }

  btnStart.disabled = true;
  fileStatus.style.color = "#ffd36c";

  try {
    // =========================
    // 1️⃣ UPLOAD RAW
    // =========================
    fileStatus.textContent = "📤 Uploading RAW data…";

    const formData = new FormData();
    formData.append("file", selectedFile);

    const uploadRes = await fetch(`${API_BASE}/runs/upload`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`
      },
      body: formData
    });

    let uploadData = {};
    try {
      uploadData = await uploadRes.json();
    } catch (_) {
      uploadData = {};
    }

    if (uploadRes.status === 401) {
      localStorage.removeItem("token");
      redirectToLogin();
      return;
    }

    if (!uploadRes.ok) {
      const detail = uploadData.detail;
      const msg =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((e) => e.msg || JSON.stringify(e)).join("; ")
            : uploadRes.statusText || "Upload failed";
      throw new Error(msg);
    }

    // store context (upload response already includes clean/prepare on server)
    localStorage.setItem("csd_run_id", uploadData.run_id);
    localStorage.setItem("csd_file_name", selectedFile.name);

    // =========================
    // 2️⃣ REDIRECT → anomaly page (Run detection = /analyze there)
    // =========================
    fileStatus.textContent =
      "✅ Data cleaned. Redirecting to anomaly detection…";
    fileStatus.style.color = "#6cff9f";

    redirectTimer = setTimeout(() => {
      window.location.assign("./anomaly.html");
    }, REDIRECT_DELAY);

  } catch (err) {
    console.error(err);
    const hint =
      err && err.message
        ? err.message
        : "Upload failed. Please try again.";
    showError(`❌ ${hint}`);
    btnStart.disabled = false;
  }
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

document.addEventListener("DOMContentLoaded", initSpaceBackground);