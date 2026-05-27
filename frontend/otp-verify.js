document.addEventListener("DOMContentLoaded", () => {

  const API_BASE = window.location.origin;

  console.log("OTP Verify Page Loaded 🔐");

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
  const otpInput = document.getElementById("otpInput");
  const verifyBtn = document.getElementById("verifyBtn");
  const resendBtn = document.getElementById("resendBtn");
  const messageBox = document.getElementById("message");
  const timerText = document.getElementById("timerText");

  // ==========================
  // Helper (موجود هنا قبل الاستخدام)
  // ==========================
  function setMessage(msg, type = "info") {
    if (!messageBox) return;
    messageBox.textContent = msg;
    messageBox.className = "message " + type;
  }

  // ==========================
  // Get Email from URL or Storage
  // ==========================
  const query = new URLSearchParams(window.location.search);
  let email = query.get("email");
  const nextTarget = query.get("next") || localStorage.getItem("postLoginRedirect");
  console.log("🔍 Email from URL:", email);

  function resolveSafeRedirect(rawTarget) {
    if (!rawTarget) return null;

    try {
      const baseDir = new URL("./", window.location.href);
      const target = new URL(rawTarget, baseDir);

      if (target.origin !== window.location.origin) return null;
      if (!target.pathname.endsWith(".html")) return null;

      return target;
    } catch {
      return null;
    }
  }
  
  if (!email) {
    email = localStorage.getItem("loginEmail");
    console.log("🔍 Email from localStorage:", email);
  }
  
  if (!email || email.trim() === "") {
    console.error("❌ Email not found");
    console.log("Available storage:", {
      urlParams: new URLSearchParams(window.location.search).toString(),
      localStorage: localStorage.length > 0 ? Object.keys(localStorage) : "empty"
    });
    setMessage("Session expired. Please login again.", "error");
    setTimeout(() => {
      const loginUrl = new URL("login.html", window.location.href);
      window.location.assign(loginUrl.toString());
    }, 3000);
  } else {
    console.log("✅ Email retrieved:", email);
  }

  // ==========================
  // Timer: 60 seconds
  // ==========================
  let timeLeft = 60;
  let timerInterval = null;

  function startTimer() {
    timeLeft = 60;
    resendBtn.disabled = true;
    timerText.classList.remove("warning");

    if (timerInterval) clearInterval(timerInterval);

    timerInterval = setInterval(() => {
      timeLeft--;

      // Format: MM:SS
      const minutes = Math.floor(timeLeft / 60);
      const seconds = timeLeft % 60;
      timerText.textContent = `0${minutes}:${seconds < 10 ? "0" : ""}${seconds}`;

      // Warning: last 10 seconds
      if (timeLeft <= 10) {
        timerText.classList.add("warning");
      }

      // Time's up
      if (timeLeft <= 0) {
        clearInterval(timerInterval);
        timerText.textContent = "00:00";
        otpInput.disabled = true;
        verifyBtn.disabled = true;
        resendBtn.disabled = false;
        setMessage("Code expired. Request a new one.", "error");
      }
    }, 1000);
  }

  // Start timer on page load
  startTimer();

  // ==========================
  // VERIFY OTP
  // ==========================
  async function handleVerifyOtp(e) {
    if (e) e.preventDefault();

    console.log("🔐 Verify OTP button clicked");

    const otp = otpInput?.value.trim();

    console.log("OTP entered:", otp);
    console.log("Email:", email);

    if (!otp) {
      setMessage("Enter the 6-digit code", "error");
      return;
    }

    if (otp.length !== 6 || !/^\d+$/.test(otp)) {
      setMessage("Code must be exactly 6 digits", "error");
      return;
    }

    verifyBtn.disabled = true;
    setMessage("Verifying...", "info");

    console.log("📤 Sending OTP verification request...");

    try {
      const requestBody = {
        email: email,
        otp: String(otp)
      };
      
      console.log("Request body:", requestBody);

      const res = await fetch(`${API_BASE}/auth/verify-otp`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(requestBody)
      });

      console.log("📩 Response status:", res.status);

      const data = await res.json();

      console.log("📩 Response data:", data);

      if (!res.ok) {
        console.error("❌ Verification failed:", data.detail);
        setMessage(data.detail || "❌ Wrong code. Try again.", "error");
        verifyBtn.disabled = false;
        return;
      }

      console.log("✅ OTP verified successfully!");
      console.log("Access token:", data.access_token);

      // ✅ Success
      setMessage("✅ Login successful! Redirecting...", "success");
      localStorage.setItem("token", data.access_token);
      
      // Clear session data
      localStorage.removeItem("loginEmail");
      localStorage.removeItem("postLoginRedirect");

      const meRes = await fetch(`${API_BASE}/auth/me`, {
        headers: {
          "Authorization": `Bearer ${data.access_token}`
        }
      });

      let role = "USER";
      if (meRes.ok) {
        const me = await meRes.json();
        role = me?.role || "USER";
      }

      let finalTarget = resolveSafeRedirect(nextTarget);

      if (finalTarget && finalTarget.pathname.endsWith("admin-cl.html") && role !== "ADMIN") {
        finalTarget = null;
      }

      if (!finalTarget) {
        finalTarget = new URL("dashboard.html", window.location.href);
      }
      
      // Redirect after 1 second
      setTimeout(() => {
        console.log("🔄 Redirecting to:", finalTarget.toString());
        window.location.assign(finalTarget.toString());
      }, 1000);

    } catch (err) {
      console.error("❌ Error during OTP verification:", err);
      setMessage("Server error. Try again.", "error");
      verifyBtn.disabled = false;
    }
  }

  // ==========================
  // RESEND OTP
  // ==========================
  async function handleResendOtp(e) {
    if (e) e.preventDefault();

    resendBtn.disabled = true;
    setMessage("Redirecting to Login...", "info");

    // To resend OTP, user needs to go back to login
    setTimeout(() => {
      console.log("🔄 Going back to login page...");
      localStorage.removeItem("loginEmail");
      const loginUrl = new URL("login.html", window.location.href);
      window.location.assign(loginUrl.toString());
    }, 1000);
  }

  // ==========================
  // EVENT LISTENERS
  // ==========================
  verifyBtn.addEventListener("click", handleVerifyOtp);
  resendBtn.addEventListener("click", handleResendOtp);

  // Allow Enter key to verify
  otpInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !verifyBtn.disabled) {
      handleVerifyOtp();
    }
  });

  // Auto-focus and format input
  otpInput.addEventListener("input", (e) => {
    e.target.value = e.target.value.replace(/[^0-9]/g, "");
  });

  // =====================================
  // CyberSatDetect | Space Background
  // =====================================

  let scene, camera, renderer, stars, sphere;

  function initSpaceBackground() {
    if (typeof THREE === "undefined") {
      console.error("Three.js failed to load on OTP page");
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
