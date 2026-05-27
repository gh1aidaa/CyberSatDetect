// ==========================
// DOM READY
// ==========================
document.addEventListener("DOMContentLoaded", () => {

  const API_BASE = window.location.origin;

  console.log("Login Page Loaded 🔥");
  console.log("API_BASE =", API_BASE);

  const urlParams = new URLSearchParams(window.location.search);
  const nextTarget = urlParams.get("next");
  if (nextTarget) {
    localStorage.setItem("postLoginRedirect", nextTarget);
  }

  // Show message if redirected due to session inactivity
  const reason = urlParams.get("reason");
  if (reason === "inactivity") {
    const messageBox = document.getElementById("message");
    if (messageBox) {
      messageBox.textContent = "Session expired due to inactivity. Please login again.";
      messageBox.className = "message error";
    }
  }

  // =====================================
  // FORCE BACKGROUND CONTAINER STYLE (JS)
  // =====================================
  const bgContainer = document.getElementById("canvas_container");
  if (bgContainer) {
    bgContainer.style.position = "fixed";
    bgContainer.style.top = "0";
    bgContainer.style.left = "0";
    bgContainer.style.width = "100vw";
    bgContainer.style.height = "100vh";
    bgContainer.style.pointerEvents = "none";
  }

  // ==========================
  // Elements
  // ==========================
  const emailInput = document.getElementById("username");
  const passwordInput = document.getElementById("password");
  const messageBox = document.getElementById("message");
  const loginBtn = document.getElementById("loginBtn");

  if (!loginBtn) {
    console.error("❌ loginBtn NOT FOUND");
    return;
  }

  // ==========================
  // Helper
  // ==========================
  function setMessage(msg, type = "info") {
    if (!messageBox) return;
    messageBox.textContent = msg;
    messageBox.className = "message " + type;
  }

  // ==========================
  // LOGIN
  // ==========================
  async function handleLogin(e) {
    if (e) e.preventDefault();

    console.log("🔥🔥🔥 LOGIN CLICKED!");

    const email = emailInput?.value?.trim();
    const password = passwordInput?.value?.trim();

    console.log("📧 Email:", email);
    console.log("🔐 Password:", password ? "***" : "(empty)");

    if (!email || !password) {
      console.warn("❌ Email or password is empty");
      setMessage("Enter email and password", "error");
      return;
    }

    loginBtn.disabled = true;
    setMessage("🔄 Signing in...", "info");

    try {
      // Connectivity preflight (helps debug Live Server vs API port)
      try {
        const h = await fetch(`${API_BASE}/health`, { cache: "no-store" });
        console.log("API health:", h.status);
      } catch (healthErr) {
        console.error("API health check failed:", healthErr);
      }

      const requestPayload = { email, password };
      console.log("📤 Sending login request...");
      console.log("Request payload:", requestPayload);

      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(requestPayload)
      });

      console.log("📬 Received response. Status:", res.status, "OK:", res.ok);

      let data;
      try {
        data = await res.json();
      } catch (parseErr) {
        console.error("❌ Failed to parse JSON:", parseErr);
        throw parseErr;
      }

      console.log("📬 Response body:", data);
      console.log("   - data.message:", data?.message);
      console.log("   - data.detail:", data?.detail);
      console.log("   - res.ok:", res.ok);

      // ❌ Check for errors first
      if (!res.ok) {
        const errorMsg = data?.detail || `HTTP ${res.status}: ${res.statusText}`;
        console.error("❌ Login failed:", errorMsg);
        setMessage(errorMsg, "error");
        loginBtn.disabled = false;
        return;
      }

      // ✅ Any successful HTTP response means OTP was created and sent.
      console.log("✅ LOGIN SUCCESS!");
      const emailForOtp = data?.email || email;
      console.log("📧 Email for OTP:", emailForOtp);

      setMessage("✅ Code sent! Redirecting...", "success");

      // Keep a fallback copy in storage in case URL param is lost.
      localStorage.setItem("loginEmail", emailForOtp);

      // Redirect with a path-safe URL based on the current page location.
      const otpUrl = new URL("otp-verify.html", window.location.href);
      otpUrl.searchParams.set("email", emailForOtp);
      if (nextTarget) {
        otpUrl.searchParams.set("next", nextTarget);
      }
      console.log("🔄 Redirect URL:", otpUrl.toString());
      // Redirect immediately (setTimeout can be skipped/cancelled in some cases)
      window.location.assign(otpUrl.toString());
      return;

    } catch (err) {
      console.error("❌ Exception caught:", err);
      console.error("   Error type:", err.constructor.name);
      console.error("   Error message:", err.message);
      setMessage(`Server error: ${err?.message || err}`, "error");
      loginBtn.disabled = false;
    }
  }

  // ==========================
  // EVENT LISTENERS
  // ==========================
  loginBtn.addEventListener("click", handleLogin);

  // Allow Enter key to login
  passwordInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      handleLogin();
    }
  });

  // =====================================
  // CyberSatDetect | Space Background
  // =====================================

  let scene, camera, renderer, stars, sphere;

  function initSpaceBackground() {
    if (typeof THREE === "undefined") {
      console.error("Three.js failed to load on login page");
      return;
    }

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
    renderer.setClearColor(0x020617, 1);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.6;
    controls.enableZoom = false;
    controls.enablePan = false;

    const loader = new THREE.TextureLoader();
    const bgTexture = loader.load(
      "https://i.ibb.co/HC0vxMw/sky2.jpg",
      undefined,
      undefined,
      () => {
        // Keep stars visible even if remote texture fails.
        renderer.setClearColor(0x020617, 1);
      }
    );

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

  initSpaceBackground();
});