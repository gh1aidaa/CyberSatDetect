// ---------------- SECURE SIGNUP (Frontend) ----------------
const signupForm = document.getElementById("signupForm");
const fullNameInput = document.getElementById("fullName");
const emailInput    = document.getElementById("email");
const passwordInput = document.getElementById("password");
const confirmInput  = document.getElementById("confirmPassword");
const roleSelect    = document.getElementById("role");
const agreeCheck    = document.getElementById("agree");

const errName  = document.getElementById("errName");
const errEmail = document.getElementById("errEmail");
const errPass  = document.getElementById("errPass");
const errConf  = document.getElementById("errConf");
const errAgree = document.getElementById("errAgree");
const msg      = document.getElementById("msg");
const signupBtn= document.getElementById("signupBtn");

function clearErrors(){
  [errName, errEmail, errPass, errConf, errAgree].forEach(e => e.textContent = "");
  msg.textContent = "";
  msg.className = "message";
}

function isStrongPassword(p){
  // 8+ chars, upper, lower, number
  return /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$/.test(p);
}

signupForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearErrors();

  const nameVal  = fullNameInput.value.trim();
  const emailVal = emailInput.value.trim().toLowerCase();
  const passVal  = passwordInput.value;
  const confVal  = confirmInput.value;
  const roleVal  = roleSelect?.value || "";

  let ok = true;

  if(!nameVal){ errName.textContent="Full name is required."; ok=false; }
  if(!emailVal){ errEmail.textContent="Email is required."; ok=false; }
  if(!passVal){ errPass.textContent="Password is required."; ok=false; }
  if(!confVal){ errConf.textContent="Confirm password."; ok=false; }

  if(passVal && !isStrongPassword(passVal)){
    errPass.textContent="Password must be 8+ chars with upper/lowercase and a number.";
    ok=false;
  }

  if(passVal && confVal && passVal !== confVal){
    errConf.textContent="Passwords do not match.";
    ok=false;
  }

  if(!agreeCheck.checked){
    errAgree.textContent="You must agree to Terms & Privacy.";
    ok=false;
  }

  if(!ok) return;

  signupBtn.classList.add("disabled");
  signupBtn.disabled = true;

  try{
    const API_BASE = window.location.origin;
    const res = await fetch(`${API_BASE}/auth/signup`, {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({
        fullName: nameVal,
        email: emailVal,
        password: passVal,
        role: roleVal
      })
    });

    const data = await res.json();
    if(!res.ok){
      msg.textContent = data.message || "Sign up failed.";
      msg.classList.add("error");
      return;
    }

    msg.textContent = "Account created successfully! Redirecting...";
    msg.classList.add("success");
    setTimeout(()=> window.location.href="login.html", 800);

  }catch(err){
    console.error(err);
    msg.textContent = "Server error. Try again later.";
    msg.classList.add("error");
  }finally{
    signupBtn.classList.remove("disabled");
    signupBtn.disabled = false;
  }
});


// ---------------- SPACE BACKGROUND (Three.js) ----------------
let scene, camera, renderer, stars, sphere;
const containerBG = document.getElementById("canvas_container");

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
  containerBG.appendChild(renderer.domElement);

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
  if (stars)  stars.rotation.y  += 0.00035;
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  if (!camera || !renderer) return;
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

initBG();
