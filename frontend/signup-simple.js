// ==========================
// SIGNUP PAGE
// ==========================
document.addEventListener("DOMContentLoaded", () => {

  console.log("Signup Page Loaded ✏️");
  const API_BASE = window.location.origin;

  // ==========================
  // Elements
  // ==========================
  const emailInput = document.getElementById("signupEmail") || document.getElementById("email");
  const passwordInput = document.getElementById("signupPassword") || document.getElementById("password");
  const confirmInput = document.getElementById("confirmPassword");
  const messageBox = document.getElementById("message") || document.getElementById("msg");
  const signupBtn = document.getElementById("signupBtn");

  if (!signupBtn) {
    console.error("❌ signupBtn NOT FOUND");
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
  // SIGNUP
  // ==========================
  async function handleSignup(e) {
    if (e) e.preventDefault();

    console.log("🔥 SIGNUP CLICKED");

    const email = emailInput?.value?.trim().toLowerCase();
    const password = passwordInput?.value?.trim();
    const confirm = confirmInput?.value?.trim();

    console.log("📧 Email:", email);
    console.log("🔐 Password:", password ? "***" : "(empty)");

    // Validation
    if (!email || !password) {
      setMessage("Enter email and password", "error");
      return;
    }

    if (password.length < 6) {
      setMessage("Password must be at least 6 characters", "error");
      return;
    }

    if (password !== confirm) {
      setMessage("Passwords do not match", "error");
      return;
    }

    signupBtn.disabled = true;
    setMessage("🔄 Creating account...", "info");

    try {
      const requestPayload = { email, password };
      console.log("📤 Sending signup request...");

      const res = await fetch(`${API_BASE}/auth/signup`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(requestPayload)
      });

      console.log("📬 Response status:", res.status);

      let data;
      try {
        data = await res.json();
      } catch (parseErr) {
        console.error("❌ Failed to parse JSON:", parseErr);
        throw parseErr;
      }

      console.log("📬 Response body:", data);

      // ❌ Check for errors first
      if (!res.ok) {
        const errorMsg = data?.detail || data?.message || `HTTP ${res.status}`;
        console.error("❌ Signup failed:", errorMsg);
        setMessage(errorMsg, "error");
        signupBtn.disabled = false;
        return;
      }

      // ✅ Success
      console.log("✅ SIGNUP SUCCESS!");
      setMessage("✅ Account created! Redirecting to login...", "success");

      setTimeout(() => {
        console.log("🔄 Redirecting to login.html...");
        window.location.href = "./login.html";
      }, 1500);

    } catch (err) {
      console.error("❌ Exception:", err);
      setMessage("Server error. Try again.", "error");
      signupBtn.disabled = false;
    }
  }

  // ==========================
  // EVENT LISTENERS
  // ==========================
  signupBtn.addEventListener("click", handleSignup);

  // Allow Enter key
  if (confirmInput) {
    confirmInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        handleSignup();
      }
    });
  }

  // ==========================
  // SPACE BACKGROUND (Three.js)
  // ==========================
  const container = document.getElementById("canvas_container");

  if (container && typeof THREE !== "undefined") {

    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer({ alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    const starsGeometry = new THREE.BufferGeometry();
    const starCount = 2000;
    const positions = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount * 3; i++) {
      positions[i] = (Math.random() - 0.5) * 1000;
    }

    starsGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3)
    );

    const starsMaterial = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 1
    });

    const stars = new THREE.Points(starsGeometry, starsMaterial);
    scene.add(stars);

    function animate() {
      requestAnimationFrame(animate);

      stars.rotation.x += 0.0005;
      stars.rotation.y += 0.0005;

      renderer.render(scene, camera);
    }

    animate();

    window.addEventListener("resize", () => {
      renderer.setSize(window.innerWidth, window.innerHeight);
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    });
  }

});
