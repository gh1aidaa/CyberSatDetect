// ---------- SPACE BACKGROUND (Three.js) ----------
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
