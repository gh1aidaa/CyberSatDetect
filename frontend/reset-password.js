// // ---------- RESET PASSWORD ----------
const form = document.getElementById("resetForm");
const msg = document.getElementById("message");
const btn = form.querySelector("button");

function setMessage(text, type = "") {
  msg.textContent = text;
  msg.className = "message" + (type ? " " + type : "");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (btn.disabled) return;

  const password = document.getElementById("password").value;
  const confirm  = document.getElementById("confirm").value;

  if (password.length < 8) {
    setMessage("Password must be at least 8 characters.", "error");
    return;
  }

  if (password !== confirm) {
    setMessage("Passwords do not match.", "error");
    return;
  }

  const params = new URLSearchParams(window.location.search);
  const token = params.get("token");

  if (!token) {
    setMessage("Invalid or expired reset link.", "error");
    return;
  }

  btn.disabled = true;

  try {
    const API_BASE = window.location.origin;
    const res = await fetch(`${API_BASE}/auth/reset-password`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        token: token,
        new_password: password
      })
    });

    const data = await res.json();

    if (!res.ok) {
      setMessage(data.detail || "Password reset failed.", "error");
      return;
    }

    // ✅ Success
    setMessage("Password updated successfully. Redirecting to login...", "success");

    setTimeout(() => {
      window.location.href = "login.html";
    }, 1500);

  } catch (err) {
    console.error("RESET ERROR:", err);
    setMessage("Server error. Please try again later.", "error");
  } finally {
    btn.disabled = false;
  }
});


// ---------- SPACE BACKGROUND (SAFE) ----------
document.addEventListener("DOMContentLoaded", () => {
  try {
    initBG();
  } catch (e) {
    console.warn("Three.js background disabled:", e);
  }
});


// ---------- THREE.JS ----------
let scene, camera, renderer, stars, sphere;
const containerBG = document.getElementById("canvas_container");

function initBG() {
  if (!containerBG || typeof THREE === "undefined") return;

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
  containerBG.appendChild(renderer.domElement);

  if (THREE.OrbitControls) {
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.7;
    controls.enablePan = false;
    controls.enableZoom = false;
  }

  const loader = new THREE.TextureLoader();
  const bg = loader.load("https://i.ibb.co/HC0vxMw/sky2.jpg");

  const geometry = new THREE.SphereGeometry(520, 48, 48);
  const material = new THREE.MeshBasicMaterial({
    side: THREE.BackSide,
    map: bg
  });

  sphere = new THREE.Mesh(geometry, material);
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

  starsGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));

  const starMat = new THREE.PointsMaterial({
    color: "#ffffff",
    size: 1.6,
    opacity: 0.9,
    transparent: true
  });

  stars = new THREE.Points(starsGeo, starMat);
  scene.add(stars);

  animateBG();
}

function animateBG() {
  requestAnimationFrame(animateBG);
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

