// ---------- FORGOT PASSWORD ----------
const form = document.getElementById("forgotForm");
const btn  = document.getElementById("resetBtn");
const msg  = document.getElementById("message");

function setMessage(text, type = "") {
  msg.textContent = text;
  msg.className = "message" + (type ? " " + type : "");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (btn.classList.contains("disabled")) return;

  const email = document.getElementById("email").value.trim().toLowerCase();
  const API_BASE = window.location.origin;

  if (!email) {
    setMessage("Please enter your email.", "error");
    return;
  }

  btn.classList.add("disabled");
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/auth/forgot-password`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ email })
    });

    const data = await res.json();

    // ⚠️ دايم نرجع نفس الرسالة (أمان)
    setMessage(
      "If the email exists, a reset link has been sent to your inbox.",
      "success"
    );

  } catch (error) {
    console.error(error);
    setMessage("Server error. Please try again later.", "error");
  } finally {
    btn.classList.remove("disabled");
    btn.disabled = false;
  }
});


// ---------- SPACE BACKGROUND (Three.js) ----------
let scene, camera, renderer, stars, sphere;
const container = document.getElementById("canvas_container");

function initBG() {
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

  starsGeo.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );

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
  if (stars) stars.rotation.y += 0.00035;
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

initBG();

